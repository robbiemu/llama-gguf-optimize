import h5py
import logging
import os

from generate_logits import generate_logits_with_llama_cpp
from compare_logits import process_chunks
from gguf_optimize_logging import setup_logging

def delete_chunk_from_hdf5(hdf5_file_path, chunk_index):
    with h5py.File(hdf5_file_path, 'a') as h5f:
        group_name = f'chunk_{chunk_index}'
        if group_name in h5f:
            del h5f[group_name]
            logging.info(f"Deleted chunk data for chunk {chunk_index} from cumulative HDF5 file.")

def get_total_chunks_from_hdf5(hdf5_file_path):
    """Reads the total number of chunks from the HDF5 file attributes."""
    with h5py.File(hdf5_file_path, 'r') as h5f:
        total_chunks = h5f.attrs['total_chunks']
    return total_chunks

def process_chunk(chunk_index, generate_args_baseline, generate_args_target, compare_args):
    # Update chunk indices
    generate_args_baseline['from_chunk'] = chunk_index
    generate_args_baseline['to_chunk'] = chunk_index
    generate_args_baseline['output'] = f"baseline_logits_chunk_{chunk_index}.h5"
    generate_args_baseline['clobber'] = True

    generate_args_target['from_chunk'] = chunk_index
    generate_args_target['to_chunk'] = chunk_index
    generate_args_target['output'] = f"target_logits_chunk_{chunk_index}.h5"
    generate_args_target['clobber'] = True

    compare_args['baseline_path'] = generate_args_baseline['output']
    compare_args['target_path'] = generate_args_target['output']
    compare_args['from_chunk'] = 0  # Each file contains only one chunk at index 0
    compare_args['to_chunk'] = 1
    compare_args['clobber'] = False  # Append to cumulative HDF5 output

    # Generate logits for baseline model
    logging.info(f"Generating logits for baseline model, chunk {chunk_index + 1}")
    generate_logits_with_llama_cpp(**generate_args_baseline)

    # Generate logits for target model
    logging.info(f"Generating logits for target model, chunk {chunk_index + 1}")
    generate_logits_with_llama_cpp(**generate_args_target)

    # Compare logits for this chunk and accumulate in output HDF5 file
    logging.info(f"Comparing logits for chunk {chunk_index + 1}")
    process_chunks(**compare_args)

    if chunk_index - 1 >= 0:
        cumulative_hdf5_file = compare_args['output_path']
        delete_chunk_from_hdf5(cumulative_hdf5_file, chunk_index - 1)

    # Delete temporary logits files to save space
    os.remove(generate_args_baseline['output'])
    os.remove(generate_args_target['output'])
    logging.info(f"Deleted temporary logits files for chunk {chunk_index + 1}")

def main(args):
    # Define arguments for generating logits and comparing logits
    generate_args_baseline = {
        'model': args.baseline_model,
        'dataset': args.dataset,
        'context_size': args.context_size,
        'n_gpu_layers': args.n_gpu_layers,
        'threads': args.threads,
        'batch_size': args.batch_size,
        'ubatch_size': args.ubatch_size,
        'precision': args.precision,
        'from_chunk': args.from_chunk,
        'to_chunk': args.to_chunk,
        'clobber': args.clobber,
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
    }

    generate_args_target = generate_args_baseline.copy()
    generate_args_target['model'] = args.target_model  # Update model for target

    compare_args = {
        'output_path': args.output_file
    }

    # Generate the first chunk to obtain total number of chunks
    logging.info("Generating first chunk to determine total number of chunks.")
    generate_args_baseline_first_chunk = generate_args_baseline.copy()
    generate_args_baseline_first_chunk['from_chunk'] = 0
    generate_args_baseline_first_chunk['to_chunk'] = 0
    generate_args_baseline_first_chunk['output'] = f"baseline_logits_chunk_{0}.h5"
    generate_args_baseline_first_chunk['clobber'] = True
    generate_logits_with_llama_cpp(**generate_args_baseline_first_chunk)

    # Read total number of chunks from the HDF5 file
    total_chunks = get_total_chunks_from_hdf5(generate_args_baseline_first_chunk['output'])
    logging.info(f"Total chunks to process: {total_chunks}")

    # Process the first chunk (we have already generated logits for the baseline model)
    process_chunk(0, generate_args_baseline_first_chunk, generate_args_target.copy(), compare_args)

    # Process remaining chunks
    for chunk_index in range(1, total_chunks):
        process_chunk(chunk_index, generate_args_baseline.copy(), generate_args_target.copy(), compare_args)

    logging.info(f"Completed processing {total_chunks} chunks. Cumulative statistics stored in {args.output_file}.")

if __name__ == '__main__':
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and compare logits chunk by chunk.")
    parser.add_argument('--baseline_model', type=str, required=True, help='Path to the baseline GGUF model file.')
    parser.add_argument('--target_model', type=str, required=True, help='Path to the target GGUF model file.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.txt file.')
    parser.add_argument('--context-size', type=int, default=2048, help="The model's context size.")
    parser.add_argument('--output_file', type=str, default='kl_divergence_overall.h5', help='Cumulative HDF5 output file for KL-divergence statistics.')
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging verbosity level (default: INFO)")
    parser.add_argument('--n-gpu-layers', type=int, help='Number of layers to store in VRAM.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing')
    parser.add_argument('--batch-size', type=int, default=2048, help='Logical maximum batch size')
    parser.add_argument('--ubatch-size', type=int, default=512, help='Physical maximum batch size')
    parser.add_argument('--precision', type=int, choices=[16,32], help='Model weight and activation precision')
    parser.add_argument('--from-chunk', type=int, default=0, help="Starting chunk index for processing")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (exclusive)")
    parser.add_argument('--clobber', action='store_true', help="Overwrite existing output file")
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
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k sampling threshold')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))
    main(args)
