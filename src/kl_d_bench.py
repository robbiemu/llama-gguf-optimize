import h5py
import logging
import numpy as np

from generate_logits import (
    generate_logits_with_llama_cpp, get_model, tokenize_dataset, 
    calculate_total_chunks, H5PY_SUPPORTED_COMPRESSIONS, SKIPPED_GENERATION
)
from compare_logits import (
    process_chunks, EARLY_STOPPING, EARLY_STOPPING_DEFAULTS,
    calculate_min_chunks_and_window_size, calculate_min_tokens
)
from gguf_optimize_logging import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_BASELINE_LOGITS_FILE = 'baseline_logits.h5'
DEFAULT_TARGET_LOGITS_FILE = 'target_logits.h5'


def reset_chunk_in_hdf5(hdf5_file_path, chunk_index):
    """
    Adds the chunk index to the freed chunks list instead of zeroing out data.
    """
    with h5py.File(hdf5_file_path, 'a') as h5f:
        if 'freed_chunks' not in h5f:
            # Create the dataset if it doesn't exist
            h5f.create_dataset('freed_chunks', data=[chunk_index], maxshape=(None,), dtype='int64')
            logger.info(f"Created 'freed_chunks' dataset and added chunk {chunk_index} in {hdf5_file_path}.")
        else:
            # Append the chunk index to the freed chunks list
            freed_chunks_dset = h5f['freed_chunks']
            freed_chunks_dset.resize((freed_chunks_dset.shape[0] + 1,))
            freed_chunks_dset[-1] = chunk_index
            logger.info(f"Added chunk {chunk_index} to freed chunks list in {hdf5_file_path}.")


def generate_logits_for_model(generate_args, chunk_index):
    """
    Generates logits for a single chunk using the provided arguments.
    """
    generate_args_chunk = generate_args.copy()
    generate_args_chunk['from_chunk'] = chunk_index
    generate_args_chunk['to_chunk'] = chunk_index
    logger.info(f"Generating logits for model, chunk {chunk_index}")

    return generate_logits_with_llama_cpp(**generate_args_chunk)


def compare_logits_for_chunk(compare_args, chunk_index):
    """
    Compares logits for a single chunk using the provided arguments.
    """
    compare_args['from_chunk'] = chunk_index
    compare_args['to_chunk'] = chunk_index
    logger.info(f"Comparing logits for chunk {chunk_index}")

    return process_chunks(**compare_args)


def calculate_and_update_compare_args(args, compare_args, sample_file=None):
    """
    Calculates min_chunks and min_tokens based on confidence level and sample_file,
    then updates compare_args accordingly.
    """
    if not args.early_stopping:
        return  # No action needed if early stopping is not enabled

    min_chunks, window_size = args.min_chunks, args.window_size
    if min_chunks is None or window_size is None:
        calculated_min_chunks, calculated_window_size = calculate_min_chunks_and_window_size(args.confidence)
        args.min_chunks = min_chunks or calculated_min_chunks
        compare_args['window_size'] = window_size or calculated_window_size
        logger.debug(f"Calculated at least one of min_chunks, window_size, from confidence: {args.min_chunks} min_chunks, {compare_args['window_size']} window_size")

    if not hasattr(args, 'min_tokens'):
        if sample_file is None:
            raise ValueError("sample_file must be provided to calculate min_tokens.")
        args.min_tokens = calculate_min_tokens(args.min_chunks, sample_file)

    # Update compare_args with the newly calculated parameters
    compare_args.update({
        'min_samples': args.min_tokens,
        'window_size': args.window_size
    })
    logger.debug(f"Updated compare_args with min_chunks and min_tokens.")


def reset_previous_chunk(args, skipped_baseline, skipped_target, chunk_index, keep):
    """
    Resets the previous chunk's data in the generated logits files if not skipped.
    """
    if chunk_index > 0:
        if not args.baseline_logits and not skipped_baseline:
            if keep > 0 and chunk_index < keep:
                logger.info(f"Chunk {chunk_index} is within the keep range ({keep}) and will not be reused.")
            else:
                reset_chunk_in_hdf5(args.baseline_logits_output, chunk_index - 1)
        if not args.target_logits and not skipped_target:
            if keep > 0 and chunk_index < keep:
                logger.info(f"Chunk {chunk_index} is within the keep range ({keep}) and will not be reused.")
            else:
                reset_chunk_in_hdf5(args.target_logits_output, chunk_index - 1)


def process_generate_both(args, total_chunks, generate_args_baseline, generate_args_target, compare_args):
    """
    Process when both logits files need to be generated.
    """
    # Set output file names from args
    generate_args_baseline = generate_args_baseline.copy()
    generate_args_target = generate_args_target.copy()
    generate_args_baseline['output'] = args.baseline_logits_output
    generate_args_target['output'] = args.target_logits_output

    logger.info("Neither logits file is supplied. Generating both baseline and target logits.")
    end = total_chunks if args.to_chunk is None else args.to_chunk + 1
    for chunk_index in range(args.from_chunk, end):
        logger.info(f"Processing chunk {chunk_index}")

        # Generate logits for baseline and target models
        generate_args_baseline['clobber'] = generate_args_target['clobber'] = (args.clobber and chunk_index == args.from_chunk)
        skipped_baseline = generate_logits_for_model(generate_args_baseline, chunk_index)
        skipped_target = generate_logits_for_model(generate_args_target, chunk_index)

        # Compare logits for this chunk
        # -- After generating the first chunk, calculate min_chunks and min_tokens
        if chunk_index == args.from_chunk and args.early_stopping:
            # Use the baseline logits output as the sample_file
            sample_file = args.baseline_logits_output
            calculate_and_update_compare_args(args, compare_args, sample_file=sample_file)
        compare_args['compute_overall'] = args.compute_overall and chunk_index == end - 1
        compare_args['clobber'] = (args.clobber and chunk_index == args.from_chunk) or chunk_index == 0
        
        comparison_result = compare_logits_for_chunk(compare_args, chunk_index)

        if comparison_result == EARLY_STOPPING:
            break

        # Reset previous chunk data in generated logits files
        reset_previous_chunk(args, skipped_baseline, skipped_target, chunk_index, args.keep)


def process_generate_one(args, total_chunks, generate_args_baseline, generate_args_target, compare_args):
    """
    Process when one logits file is supplied and the other needs to be generated.
    """
    # Copy and assign the output file name as needed
    generate_args_baseline = generate_args_baseline.copy()
    generate_args_target = generate_args_target.copy()
    if not args.baseline_logits:
        generate_args_baseline['output'] = args.baseline_logits_output
        logger.info("Baseline logits file is not supplied. Generating the missing logits.")
    if not args.target_logits:
        generate_args_target['output'] = args.target_logits_output
        logger.info("Target logits file is not supplied. Generating the missing logits.")

    end = total_chunks if args.to_chunk is None else args.to_chunk
    for chunk_index in range(args.from_chunk, end):
        logger.info(f"Processing chunk {chunk_index}")

        # Generate logits as needed
        skipped_baseline = skipped_target = None
        if not args.baseline_logits:
            generate_args_baseline['clobber'] = (args.clobber and chunk_index == args.from_chunk)
            skipped_baseline = generate_logits_for_model(generate_args_baseline, chunk_index)
        if not args.target_logits:
            generate_args_target['clobber'] = (args.clobber and chunk_index == args.from_chunk)
            skipped_target = generate_logits_for_model(generate_args_target, chunk_index)

        # Compare logits for this chunk
        # -- After generating the first chunk, calculate min_chunks and min_tokens
        if chunk_index == args.from_chunk and args.early_stopping:
            # Use the generated logits file as the sample_file
            sample_file = args.baseline_logits_output if not args.baseline_logits else args.target_logits_output
            calculate_and_update_compare_args(args, compare_args, sample_file=sample_file)
        compare_args['compute_overall'] = args.compute_overall and chunk_index == end - 1
        compare_args['clobber'] = (args.clobber and chunk_index == args.from_chunk) or chunk_index == 0
        comparison_result = compare_logits_for_chunk(compare_args, chunk_index)

        if comparison_result == EARLY_STOPPING:
            break

        # Reset previous chunk data in generated logits files
        reset_previous_chunk(args, skipped_baseline, skipped_target, chunk_index, args.keep)


def process_generate_neither(args, total_chunks, compare_args):
    """
    Process when both logits files are supplied and only comparison is needed.
    """
    logger.info("Both baseline and target logits files are supplied. Proceeding to compare directly.")
    end = total_chunks if args.to_chunk is None else args.to_chunk
    for chunk_index in range(args.from_chunk, end):
        # Compare logits for this chunk
        # -- After generating the first chunk, calculate min_chunks and min_tokens
        if chunk_index == args.from_chunk and args.early_stopping:
            # Use the baseline logits output as the sample_file
            sample_file = args.baseline_logits
            calculate_and_update_compare_args(args, compare_args, sample_file=sample_file)

        #logger.info(f"Comparing logits for chunk {chunk_index}")
        compare_args['compute_overall'] = args.compute_overall and chunk_index == end - 1
        compare_args['clobber'] = (args.clobber and chunk_index == args.from_chunk) or chunk_index == 0
        comparison_result = compare_logits_for_chunk(compare_args, chunk_index)

        if comparison_result == EARLY_STOPPING:
            break


def main(args):
    # Determine whether we have to generate logits or not
    baseline_logits_supplied = args.baseline_logits is not None
    target_logits_supplied = args.target_logits is not None

    # Initialize compare_args
    compare_args = {
        'baseline_path': args.baseline_logits or args.baseline_logits_output or DEFAULT_BASELINE_LOGITS_FILE,
        'target_path': args.target_logits or args.target_logits_output or DEFAULT_TARGET_LOGITS_FILE,
        'precision': args.kld_precision,
        'parts': args.parts,
        'output_path': args.output_file,  # KL-divergence cumulative file
    }

    # Conditionally add early_stopping arguments
    if args.early_stopping:
        compare_args.update({
            'early_stopping': args.early_stopping,
            'confidence_level': args.confidence,
            'margin_of_error': args.margin_of_error,
            'prior_learning_rate': args.learning_rate,
            'min_prior_weight': args.min_prior_weight,
            'decay_rate': args.decay_rate,
            'momentum': args.momentum,
            'theta_E': args.theta_E,
            'theta_P': args.theta_P,
            'dynamic_thresholds_enabled': args.dynamic_thresholds_enabled,
            'log_effect_sizes': args.log_effect_sizes,
        })
        logger.debug("Early stopping arguments added to compare_args.")

    # Prepare generation arguments
    generate_args_baseline = {}
    generate_args_target = {}

    if not baseline_logits_supplied:
        generate_args_baseline = {
            'model': args.baseline_model,
            'dataset': args.dataset,
            'context_size': args.context_size,
            'n_gpu_layers': args.n_gpu_layers,
            'threads': args.threads,
            'batch_size': args.batch_size,
            'ubatch_size': args.ubatch_size,
            'precision': args.model_precision,
            'compression': args.compression,
            'clobber': False,
            'rope_freq_base': args.rope_freq_base,
            'repeat_last_n': args.repeat_last_n,
            'repeat_penalty': args.repeat_penalty,
            'presence_penalty': args.presence_penalty,
            'frequency_penalty': args.frequency_penalty,
            'dynatemp_range': args.dynatemp_range,
            'dynatemp_exp': args.dynatemp_exp,
            'mirostat': args.mirostat,
            'mirostat_lr': args.mirostat_lr,
            'mirostat_ent': args.mirostat_ent,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'temp': args.temp,
            'seed': args.seed,
            'verbosity': args.verbosity,
            'output': DEFAULT_BASELINE_LOGITS_FILE
        }

    if not target_logits_supplied:
        generate_args_target = {
            'model': args.target_model,
            'dataset': args.dataset,
            'context_size': args.context_size,
            'n_gpu_layers': args.n_gpu_layers,
            'threads': args.threads,
            'batch_size': args.batch_size,
            'ubatch_size': args.ubatch_size,
            'precision': args.model_precision,
            'compression': args.compression,
            'clobber': False,
            'rope_freq_base': args.rope_freq_base,
            'repeat_last_n': args.repeat_last_n,
            'repeat_penalty': args.repeat_penalty,
            'presence_penalty': args.presence_penalty,
            'frequency_penalty': args.frequency_penalty,
            'dynatemp_range': args.dynatemp_range,
            'dynatemp_exp': args.dynatemp_exp,
            'mirostat': args.mirostat,
            'mirostat_lr': args.mirostat_lr,
            'mirostat_ent': args.mirostat_ent,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'temp': args.temp,
            'seed': args.seed,
            'verbosity': args.verbosity,
            'output': DEFAULT_TARGET_LOGITS_FILE
        }

    # Determine total_chunks and vocab_size
    if baseline_logits_supplied:
        with h5py.File(args.baseline_logits, 'r') as h5f:
            total_chunks = h5f.attrs.get('total_chunks', 0)
    elif target_logits_supplied:
        with h5py.File(args.target_logits, 'r') as h5f:
            total_chunks = h5f.attrs.get('total_chunks', 0)
    else:
        # Neither logits file is supplied; generate the first chunk to get total_chunks
        logger.info("Determining total number of chunks.")
        model = get_model(**generate_args_baseline)
        _tokens, total_tokens = tokenize_dataset(model, generate_args_baseline['dataset'])
        total_chunks = calculate_total_chunks(total_tokens, generate_args_baseline['context_size'], model)

        logger.info(f"Total chunks to process: {total_chunks}")

    # Decide which processing function to call based on supplied arguments
    if not baseline_logits_supplied and not target_logits_supplied:
        process_generate_both(
            args, total_chunks, generate_args_baseline, generate_args_target, 
            compare_args
        )
    elif not baseline_logits_supplied or not target_logits_supplied:
        process_generate_one(
            args, total_chunks, generate_args_baseline, generate_args_target, 
            compare_args
        )
    else:
        process_generate_neither(args, total_chunks, compare_args)

    logger.info(f"Completed processing {total_chunks} chunks.\nCumulative statistics stored in {args.output_file}.")


if __name__ == '__main__':
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and compare logits chunk by chunk.")

    baseline_group = parser.add_argument_group("Baseline parameters")
    baseline_required = baseline_group.add_mutually_exclusive_group(required=True)
    baseline_required.add_argument('--baseline-model', type=str, help='Path to the baseline GGUF model file.')
    baseline_required.add_argument('--baseline-logits', type=str, help='Path to the baseline logits HDF5 file.')
    baseline_group.add_argument('--baseline-logits-output', type=str, default=None,
        help=f"Output file for baseline logits when generating from model (default: {DEFAULT_BASELINE_LOGITS_FILE})")

    target_group = parser.add_argument_group("Target parameters")
    target_required = target_group.add_mutually_exclusive_group(required=True)
    target_required.add_argument('--target-model', type=str, help='Path to the target GGUF model file.')
    target_required.add_argument('--target-logits', type=str, help='Path to the target logits HDF5 file.')
    target_group.add_argument('--target-logits-output', type=str, default=None,
        help=f"Output file for target logits when generating from model (default: {DEFAULT_TARGET_LOGITS_FILE})")

    parser.add_argument('--dataset', type=str, help='Path to the dataset.txt file.')

    parser.add_argument('--output-file', type=str, default='kl_divergence.h5', help='Cumulative HDF5 output file for KL-divergence statistics.')
    parser.add_argument('--keep', type=int, default=0, 
        help="Number of chunks to save before reusing any (default: 0, which disables this feature).")
    parser.add_argument('--from-chunk', type=int, default=0, help="Starting chunk index for processing")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (exclusive)")
    parser.add_argument('--clobber', action='store_true', help="Overwrite existing output file")
    parser.add_argument('--compute-overall', action='store_true', help="Compute overall statistics even if early stopping occurs.")

    # Early stopping group
    early_stopping_group = parser.add_argument_group("Early stopping parameters")
    early_stopping_group.add_argument('--early-stopping', action='store_true',
                                      help="Enable early stopping based on confidence level.")
    early_stopping_group.add_argument('--confidence', type=float,
                                      help=f"Confidence level used in both the Beta distribution model and the Kuiper test to determine the threshold for stopping. A higher confidence level requires more stability in the pattern of effect sizes and p-values (default: {EARLY_STOPPING_DEFAULTS['confidence']}).")
    early_stopping_group.add_argument('--margin-of-error', type=float,
                                      help=f"Margin of error used in both the Beta distribution model and the Kuiper test to determine the precision required for stopping. A smaller margin of error requires more precision in the pattern of effect sizes and p-values (default: {EARLY_STOPPING_DEFAULTS['margin_of_error']}).")
    early_stopping_group.add_argument('--min-chunks', type=int,
                                      help="Minimum number of chunks required before early stopping. If not specified, it is calculated based on the confidence level. This parameter determines the window size for pattern analysis (default: calculated from confidence level).")


    # Bayesian update parameters group
    beta_group = parser.add_argument_group("Bayesian update parameters")
    beta_group.add_argument('--learning-rate', type=float,
                            help=f"Learning rate for Bayesian prior updates (default: {EARLY_STOPPING_DEFAULTS['learning_rate']}).")
    beta_group.add_argument('--min-prior-weight', type=float,
                            help=f"Minimum weight given to current prior in updates (default: {EARLY_STOPPING_DEFAULTS['min_prior_weight']}).")
    beta_group.add_argument('--momentum', type=float,
                            help=f"Momentum factor for smoothing learning rate updates (default: {EARLY_STOPPING_DEFAULTS['momentum']}).")
    beta_group.add_argument('--decay-rate', type=float,
                            help="Decay rate for dynamically adjusting the learning rate (default: calculated from total chunks).")
    beta_group.add_argument('--theta-E', type=float,
                            help=f"Maximum relative change in effect sizes across window of chunks, used in the Beta distribution model to determine pattern stability. A smaller θ_E requires more stability for effect sizes (default: {EARLY_STOPPING_DEFAULTS['theta_E']}, dynamic if theta-P also unset).")
    beta_group.add_argument('--theta-P', type=float,
                            help=f"Maximum std of p-values across window of chunks, used in the Beta distribution model to determine pattern stability. A smaller θ_P requires more stability for p-values (default: {EARLY_STOPPING_DEFAULTS['theta_P']}, dynamic if theta-P also unset).")
    beta_group.add_argument('--window-size', type=int,
                            help="Size of the sliding window for early stopping, used to analyze pattern stability in the Beta distribution model (default: calculated from confidence level).")

    parser.add_argument('--n-gpu-layers', type=int, help='Number of layers to store in VRAM.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing')
    parser.add_argument('--context-size', type=int, default=45056, help="The model's context size.")
    parser.add_argument('--batch-size', type=int, help='Logical maximum batch size (default: context size)')
    parser.add_argument('--ubatch-size', type=int, help='Physical maximum batch size (default: context size)')
    parser.add_argument('--model-precision', type=int, choices=[16,32], default=32, help='Model\'s activation precision (default: 32) note: currently llama.cpp only supports fp32 for processing the output weights, so this will be fp32')
    parser.add_argument('--kld-precision', type=int, choices=[32,64], default=None, help='Precision for calculating kl-divergence (default: twice model precision) note: memory intensive')
    parser.add_argument('--compression', type=str, choices=H5PY_SUPPORTED_COMPRESSIONS.keys(), default=None, help='Compression method to use for the output logits file. (Default: None)')
    parser.add_argument('--parts', type=int, default=1, help="Number of parts to split each chunk into for processing.")

    parser.add_argument('--rope-freq-base', type=float, help='ROPE frequency base')
    parser.add_argument('--repeat-last-n', type=int, default=64, help='Last n tokens for repeat penalty')
    parser.add_argument('--repeat-penalty', type=float, default=1.0, help='Repeat sequence penalty')
    parser.add_argument('--presence-penalty', type=float, default=0.0, help='Presence penalty for repeat tokens')
    parser.add_argument('--frequency-penalty', type=float, default=0.0, help='Frequency penalty for repeat tokens')
    parser.add_argument('--dynatemp-range', type=float, default=0.0, help='Dynamic temperature range')
    parser.add_argument('--dynatemp-exp', type=float, default=1.0, help='Dynamic temperature exponent')
    parser.add_argument('--mirostat', type=int, default=0, help='Mirostat sampling mode')
    parser.add_argument('--mirostat-lr', type=float, default=0.1, help='Mirostat learning rate')
    parser.add_argument('--mirostat-ent', type=float, default=5.0, help='Mirostat target entropy')
    parser.add_argument('--top-p', type=float, default=0, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k sampling threshold')
    parser.add_argument('--temp', type=float, default=0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Verbosity arguments
    early_stopping_group.add_argument('--log-effect-sizes', action='store_true',
                                      help="Enable logging of effect sizes for each chunk.")
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging verbosity level (default: INFO)")

    args = parser.parse_args()

    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))

    # Validate mutual exclusivity between logits files and dataset
    if not ((args.baseline_logits and args.target_logits) or args.dataset):
        raise ValueError("Either both baseline and target logits files must be provided, or --dataset must be specified.")
    if args.baseline_logits and args.target_logits and args.dataset:
        raise ValueError("If both baseline and target logits files are provided, --dataset should not be specified.")

    # Validation for mutually exclusive file output arguments
    if args.baseline_logits and args.baseline_logits_output:
        parser.error("Cannot use --baseline-logits-output when --baseline_logits is set.")
    if args.target_logits and args.target_logits_output:
        parser.error("Cannot use --target-logits-output when --target_logits is set.")

    if args.baseline_logits_output is None:
        args.baseline_logits_output = DEFAULT_BASELINE_LOGITS_FILE

    if args.target_logits_output is None:
        args.target_logits_output = DEFAULT_TARGET_LOGITS_FILE

    if args.batch_size is None:
        logger.debug("Setting batch size to context size: %s", args.context_size)
        args.batch_size = args.context_size
    if args.ubatch_size is None:
        logger.debug("Setting μbatch size to context size: %s", args.context_size)
        args.ubatch_size = args.context_size

    if args.early_stopping:
        setattr(args, "dynamic_thresholds_enabled", args.theta_E is None and args.theta_P is None)

        # Set defaults only for arguments that are None
        for arg, default in EARLY_STOPPING_DEFAULTS.items():
            if getattr(args, arg) is None:
                setattr(args, arg, default)

        min_chunks, window_size = args.min_chunks, args.window_size
        if min_chunks is None or window_size is None:
            calculated_min_chunks, calculated_window_size = calculate_min_chunks_and_window_size(args.confidence)
            args.min_chunks = min_chunks or calculated_min_chunks
            args.window_size = window_size or calculated_window_size

        # Validate early stopping arguments
        if args.confidence < args.margin_of_error:
            parser.error("--confidence must be greater than --margin-of-error.")
        if not (0 < args.learning_rate <= 1):
            parser.error("The prior learning rate must be between 0 and 1")
        if not (0 < args.min_prior_weight <= 1):
            parser.error("The minimum prior weight must be between 0 and 1")
        if not (0 <= args.theta_E <= 1):
            parser.error("--theta-E must be between 0 and 1.")
        if not (0 <= args.theta_P <= 1):
            parser.error("--theta-P must be between 0 and 1.")
        if args.min_chunks is not None and args.min_chunks <= 0:
            parser.error("--min-chunks must be a positive integer.")
    else:
        # Warn about ignored arguments only if explicitly set
        ignored_args = [
            f"--{arg.replace('_', '-')}" for arg in EARLY_STOPPING_DEFAULTS.keys()
            if getattr(args, arg) is not None
        ]
        if ignored_args:
            logger.warning(f"The following arguments are ignored because --early-stopping is not enabled: {', '.join(ignored_args)}")

    logger.info(f"kl_d_bench starting...")

    main(args)

    logger.info("kl_d_bench completed successfully.")
