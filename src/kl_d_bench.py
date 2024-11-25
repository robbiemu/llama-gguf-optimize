import h5py
import logging
import numpy as np

from generate_logits import (
    generate_logits_with_llama_cpp, get_model, tokenize_dataset, 
    calculate_total_chunks, H5PY_SUPPORTED_COMPRESSIONS
)
from compare_logits import process_chunks
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
    generate_args_chunk['clobber'] = args.clobber or chunk_index == 0 
    logger.info(f"Generating logits for model, chunk {chunk_index}")
    generate_logits_with_llama_cpp(**generate_args_chunk)


def compare_logits_for_chunk(compare_args, chunk_index):
    """
    Compares logits for a single chunk using the provided arguments.
    """
    compare_args['from_chunk'] = chunk_index
    compare_args['to_chunk'] = chunk_index
    compare_args['clobber'] = args.clobber or chunk_index == 0 
    logger.info(f"Comparing logits for chunk {chunk_index}")
    process_chunks(**compare_args)


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
    end = total_chunks if args.to_chunk is None else args.to_chunk
    for chunk_index in range(args.from_chunk, end):
        logger.info(f"Processing chunk {chunk_index}")

        # Generate logits for baseline and target models
        generate_logits_for_model(generate_args_baseline, chunk_index)
        generate_logits_for_model(generate_args_target, chunk_index)

        # Compare logits for this chunk
        compare_logits_for_chunk(compare_args, chunk_index)

        # Reset previous chunk data in generated logits files
        if chunk_index > 0:
            reset_chunk_in_hdf5(args.baseline_logits_output, chunk_index - 1)
            reset_chunk_in_hdf5(args.target_logits_output, chunk_index - 1)


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
        if not args.baseline_logits:
            generate_logits_for_model(generate_args_baseline, chunk_index)
        if not args.target_logits:
            generate_logits_for_model(generate_args_target, chunk_index)

        # Compare logits for this chunk
        compare_logits_for_chunk(compare_args, chunk_index)

        # Reset previous chunk data in generated logits files
        if chunk_index > 0:
            if not args.baseline_logits:
                reset_chunk_in_hdf5(args.baseline_logits_output, chunk_index - 1)
            if not args.target_logits:
                reset_chunk_in_hdf5(args.target_logits_output, chunk_index - 1)


def process_generate_neither(args, total_chunks, compare_args):
    """
    Process when both logits files are supplied and only comparison is needed.
    """
    logger.info("Both baseline and target logits files are supplied. Proceeding to compare directly.")
    end = total_chunks if args.to_chunk is None else args.to_chunk
    for chunk_index in range(args.from_chunk, end):
        logger.info(f"Comparing logits for chunk {chunk_index}")
        compare_logits_for_chunk(compare_args, chunk_index)


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
        'output_path': args.output_file  # KL-divergence cumulative file
    }

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
            total_chunks = h5f.attrs['total_chunks']
    elif target_logits_supplied:
        with h5py.File(args.target_logits, 'r') as h5f:
            total_chunks = h5f.attrs['total_chunks']
    else:
        # Neither logits file is supplied; generate the first chunk to get total_chunks
        logger.info("Determining total number of chunks.")
        model = get_model(**generate_args_baseline)
        _tokens, total_tokens = tokenize_dataset(model, generate_args_baseline['dataset'])
        total_chunks = calculate_total_chunks(total_tokens, generate_args_baseline['context_size'], model)

        logger.info(f"Total chunks to process: {total_chunks}")

    # Decide which processing function to call based on supplied arguments
    if not baseline_logits_supplied and not target_logits_supplied:
        process_generate_both(args, total_chunks, generate_args_baseline, generate_args_target, compare_args)
    elif not baseline_logits_supplied or not target_logits_supplied:
        process_generate_one(args, total_chunks, generate_args_baseline, generate_args_target, compare_args)
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
        help=f"Output file for target logits when generating from model (default: {DEFAULT_BASELINE_LOGITS_FILE})")
    parser.add_argument('--dataset', type=str, help='Path to the dataset.txt file.')

    parser.add_argument('--output-file', type=str, default='kl_divergence.h5', help='Cumulative HDF5 output file for KL-divergence statistics.')
    parser.add_argument('--from-chunk', type=int, default=0, help="Starting chunk index for processing")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (exclusive)")
    parser.add_argument('--clobber', action='store_true', help="Overwrite existing output file")

    parser.add_argument('--n-gpu-layers', type=int, help='Number of layers to store in VRAM.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing')
    parser.add_argument('--context-size', type=int, default=2048, help="The model's context size.")
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

    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging verbosity level (default: INFO)")

    args = parser.parse_args()

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


    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))

    main(args)
