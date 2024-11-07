from collections import defaultdict
import json
import logging
import math
import multiprocessing
import os
import re
import subprocess
import yaml


logger = logging.getLogger(__name__)


def load_quantizations_from_config(config_file):
    """Load quantization types from a YAML configuration file."""
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data.get("quantizations", [])


def extract_from_config(config_file):
    """Extract parameters from a JSON configuration file."""
    with open(config_file, 'r') as file:
        config_data = json.load(file)
        
    param_mapping = {
        "max_position_embeddings": "ctx_size",
        "rope_theta": "rope_freq_base",
        "rope_scaling": "rope_scaling",
        "rope_scaling_type": "rope_scaling_type",
        "torch_dtype": "torch_dtype",
        "sampling.top_p": "top_p",
        "sampling.temperature": "temp",
        "sampling.repeat_penalty": "repeat_penalty",
        "sampling.repeat_last_n": "repeat_last_n",
        "sampling.min_p": "min_p",
        "sampling.top_k": "top_k",
        "sampling.presence_penalty": "presence_penalty",
        "sampling.frequency_penalty": "frequency_penalty",
        "sampling.mirostat": "mirostat",
        "sampling.mirostat_lr": "mirostat_lr",
        "sampling.mirostat_ent": "mirostat_ent",
        "sampling.tfs": "tfs",  
        "sampling.typical": "typical"
    }

    params = {param_mapping[key]: config_data.get(key) for key in param_mapping if key in config_data}

    return {k: v for k, v in params.items() if v is not None}


def apply_model_specific_overrides(args, config_params):
    """Apply user-specified model-specific arguments to override configuration parameters."""
    model_specific_args = [
        "temp", "top_k", "top_p", "min_p", "seed", "repeat_last_n",
        "repeat_penalty", "presence_penalty", "frequency_penalty",
        "tfs", "typical", "mirostat", "mirostat_lr", "mirostat_ent"
    ]

    for arg in model_specific_args:
        arg_name = arg.replace('-', '_')
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config_params[arg] = arg_value


def determine_base_precision(config_params):
    """Determine the base precision based on torch_dtype in config_params."""
    unquantized = defaultdict(lambda: "fp16")
    unquantized["float32"] = "fp32"
    unquantized["float16"] = "fp16"
    unquantized["bfloat16"] = "bf16"

    return unquantized[config_params.get("torch_dtype", "float16")]


def quantize(args, quantizations):
    """Quantize models for each specified quantization type."""
    # Load configuration parameters
    if args.config:
        config_params = extract_from_config(args.config)
    else:
        config_params = {}

    # Apply user overrides
    apply_model_specific_overrides(args, config_params)

    # Parameters relevant to quantization
    quantization_specific_params = ["ctx_size", "rope_freq_base", "rope_scaling", "rope_scaling_type"]

    # Extract quantization-specific parameters
    quantization_params = {k: v for k, v in config_params.items() if k in quantization_specific_params}

    command_parts = [
        os.path.join(args.path_to_llamacpp, "llama-quantize") if args.path_to_llamacpp else "llama-quantize"
    ]

    for quant_type in quantizations:
        output_model = os.path.join(args.output_dir, f"{args.model_name}_{quant_type}.gguf")
        
        if not args.overwrite and os.path.exists(output_model):
            logger.info(f"Quantized model {output_model} already exists. Skipping.")
            continue

        # Add quantization-specific parameters
        for param, value in quantization_params.items():
            if value is not None:
                command_parts.append(f"--{param.replace('_', '-')}")
                command_parts.append(str(value))

        if args.imatrix_path:
            command_parts.append(f"--imatrix {args.imatrix_path}")
        if args.use_leave_output_tensor:
            command_parts.append("--leave-output-tensor")

        # Base model, output model, and quantization type
        command_parts.extend([f"{args.base_model}", f"\"{output_model}\"", f"{quant_type}"])

        # Redirect output to a log file
        log_file = os.path.join(args.output_dir, f"{quant_type}_log.txt")
        command_parts.append(f"> \"{log_file}\" 2>&1")

        # Construct and execute command
        quantize_command = " ".join(command_parts)

        if args.dry_run:
            print(f"Dry-run (quantize): {quantize_command}")
            continue
        else:
            logger.info(f"Running quantization command: {quantize_command}")
            try:
                result = subprocess.run(quantize_command, shell=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Error during quantization to {quant_type}. Check {log_file} for details.")
                else:
                    logger.info(f"Successfully quantized model to {quant_type} and saved as {output_model}.")
            except Exception as e:
                logger.exception(f"Exception occurred while quantizing model to {quant_type}: {e}")


def measure_perplexity(args, quantizations):
    """Measure perplexity for each model."""
    # Load configuration parameters
    config_params = extract_from_config(args.config) if args.config else {}

    apply_model_specific_overrides(args, config_params)

    # Set default temperature to 0 if not specified
    if 'temp' not in config_params:
        config_params['temp'] = 0

    # Determine base precision
    base_precision = determine_base_precision(config_params)
    base_model = os.path.join(args.output_dir, f"{args.model_name}_{base_precision}.gguf")
    base_quant_type = base_precision  # Use base precision as the quant type for base model

    # Create a list of all models including the base model
    all_models = [(base_quant_type, base_model)] + [
        (quant_type, os.path.join(args.output_dir, f"{args.model_name}_{quant_type}.gguf")) for quant_type in quantizations
    ]

    perplexity_results = {}

    for quant_type, model in all_models:
        output_file = os.path.join(args.output_dir, f"perplexity_{quant_type}.txt")

        # Build the command
        command_parts = [
            os.path.join(args.path_to_llamacpp, 'llama-perplexity') if args.path_to_llamacpp else 'llama-perplexity',
            "-m", model,
            "-f", args.ppl_file,
            "--all-logits"
        ]

        # Add parameters from config_params
        for param, value in config_params.items():
            if value is not None:
                command_parts.append(f"--{param.replace('_', '-')}")
                command_parts.append(str(value))

        # Add fixed parameters
        command_parts.extend([
            f"--threads {args.threads}",
            f"--batch-size {args.batch_size}",
            f"--ubatch-size {args.ubatch_size}",
        ])

        # Redirect output to file
        command = " ".join(command_parts) + f" > \"{output_file}\" 2>&1"

        if args.dry_run:
            print(f"Dry-run (perplexity): {command}")
            continue

        else:
            logger.info(f"Running perplexity measurement for {quant_type}")
            try:
                result = subprocess.run(command, shell=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Error during perplexity measurement for {quant_type}")
                else:
                    # Read the output from the output file
                    with open(output_file, 'r') as f:
                        output = f.read()
                    perplexity = extract_perplexity(output)
                    if perplexity is not None:
                        perplexity_results[quant_type] = perplexity
                        logger.info(f"Perplexity for {quant_type}: {perplexity}")
                    else:
                        logger.warning(f"Could not extract perplexity for {quant_type}")
            except Exception as e:
                logger.exception(f"Exception occurred while measuring perplexity for {quant_type}: {e}")

    # After measurement, proceed to summarize results
    summarize_perplexity_results(args, perplexity_results, base_precision)


def extract_perplexity(output):
    """Extract perplexity from the output."""
    match = re.search(r"Final estimate: PPL = ([\d.]+)", output)

    return float(match.group(1)) if match else None


def summarize_perplexity_results(args, perplexity_results, base_precision):
    """Summarize perplexity results and display comparison table."""
    base_perplexity = perplexity_results.get(base_precision, None)

    if base_perplexity:
        print("\nPerplexity Comparison Table:")
        print(f"{'Quantization Type':<20} {'PPL(Q)':<15} {'ln(PPL(Q)/PPL(base))':<25}")
        print("=" * 65)
        for quant, ppl in perplexity_results.items():
            if ppl and base_perplexity:
                ln_ratio = round(math.log(ppl / base_perplexity), 6)
                print(f"{quant:<20} {ppl:<15} {ln_ratio:<25}")
    else:
        print("Base perplexity data missing; summary may be incomplete.")


def ppl_summary(args, quantizations):
    """Summarize perplexity results from existing files."""
    # Load configuration parameters
    config_params = extract_from_config(args.config) if args.config else {}

    # Determine base precision
    base_precision = determine_base_precision(config_params)

    perplexity_results = {}

    # Load perplexity results
    all_quant_types = [base_precision] + quantizations
    for quant_type in all_quant_types:
        output_file = os.path.join(args.output_dir, f"perplexity_{quant_type}.txt")
        try:
            with open(output_file, 'r') as file:
                output = file.read()
            perplexity = extract_perplexity(output)
            if perplexity:
                perplexity_results[quant_type] = perplexity
        except FileNotFoundError:
            logger.warning(f"Perplexity file {output_file} not found for {quant_type}.")

    # Summarize results
    summarize_perplexity_results(args, perplexity_results, base_precision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize models and measure perplexity using llama.cpp tools."
    )
    parser.add_argument("task", choices=["quantize", "perplexity", "ppl_summary"],
                        help="Task to perform: 'quantize', 'perplexity', 'ppl_summary'.")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist.")
    parser.add_argument("--verbosity", type=str, choices=["INFO", "DEBUG"], default="INFO", help="Logging verbosity level.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--path-to-llamacpp", type=str, default="", help="Path to the llama.cpp binaries directory.")

    quant_group = parser.add_mutually_exclusive_group(required=True)
    quant_group.add_argument("--config", type=str, help="Path to configuration file containing quantizations.")
    quant_group.add_argument("--quantizations", nargs="+", type=str, help="Specify quantization types directly.")

    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save quantized models and output files.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--base-model", type=str, help="Path to the base model file.")

    parser.add_argument("--imatrix-path", type=str, help="Path to the importance matrix file.")
    parser.add_argument("--use-leave-output-tensor", action="store_true", help="Use the --leave-output-tensor flag.")

    parser.add_argument("--dataset", type=str, default="ppl_test_data.txt", help="Path to the perplexity test data file.")

    parser.add_argument("--threads", type=int, default=max(multiprocessing.cpu_count() - 1, 1), help="Number of threads to use (default: one less than CPU cores).")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for perplexity computation (default: 512).")
    parser.add_argument("--ubatch-size", type=int, default=128, help="Micro-batch size for perplexity computation (default: 128).")

    # Add model-specific flags as optional arguments
    parser.add_argument("--temp", type=float, default=0, help="Temperature for sampling (default: 0).")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, help="Min-p sampling")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--repeat-last-n", type=int, help="Last n tokens to consider for penalization")
    parser.add_argument("--repeat-penalty", type=float, help="Penalize repeat sequence of tokens")
    parser.add_argument("--presence-penalty", type=float, help="Repeat alpha presence penalty")
    parser.add_argument("--frequency-penalty", type=float, help="Repeat alpha frequency penalty")
    parser.add_argument("--tfs", type=float, help="Tail Free Sampling value")
    parser.add_argument("--typical", type=float, help="Locally Typical Sampling value")
    parser.add_argument("--mirostat", type=int, help="Use Mirostat sampling")
    parser.add_argument("--mirostat-lr", type=float, help="Mirostat learning rate, parameter eta")
    parser.add_argument("--mirostat-ent", type=float, help="Mirostat target entropy, parameter tau")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.verbosity.upper()))

    # Enforce that only one of --quantizations or --config can be specified
    if args.quantizations and args.config:
        parser.error("Specify only one of --quantizations or --config, not both.")

    # Load quantizations from config file or command line argument
    if args.config:
        quantizations = load_quantizations_from_config(args.config)
    elif args.quantizations:
        quantizations = args.quantizations
    else:
        parser.error("One of --quantizations or --config must be specified.")

    # Ensure base_model is specified for quantize task
    if args.task == "quantize" and not args.base_model:
        parser.error("--base_model is required for quantize task.")

    # Execute the selected task
    if args.task == "quantize":
        quantize(args, quantizations)
    elif args.task == "perplexity":
        measure_perplexity(args, quantizations)
    elif args.task == "ppl_summary":
        ppl_summary(args, quantizations)
