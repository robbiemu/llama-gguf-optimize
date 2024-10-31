import contextlib
import gc
import h5py
import llama_cpp
import logging
import numpy as np
import os
import time

from version import __version__
from gguf_optimize_model_fns import estimate_model_precision


logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def prepare_llama_args(kwargs):
    llama_args = {
        'model_path': kwargs.get('model'),
        'n_threads': kwargs.get('threads'),
        'n_gpu_layers': kwargs.get('n_gpu_layers'),
        'seed': kwargs.get('seed'),
        'n_ctx': kwargs.get('context_size'),
        'n_batch': kwargs.get('batch_size'),
        'n_ubatch': kwargs.get('ubatch_size'),
        'rope_freq_base': kwargs.get('rope_freq_base'),
        'temp': kwargs.get('temp'),
        'top_k': kwargs.get('top_k'),
        'top_p': kwargs.get('top_p'),
        'min_p': kwargs.get('min_p'),
        'repeat_last_n': kwargs.get('repeat_last_n'),
        'repeat_penalty': kwargs.get('repeat_penalty'),
        'presence_penalty': kwargs.get('presence_penalty'),
        'frequency_penalty': kwargs.get('frequency_penalty'),
        'dynatemp_range': kwargs.get('dynatemp_range'),
        'dynatemp_exp': kwargs.get('dynatemp_exp'),
        'mirostat': kwargs.get('mirostat'),
        'mirostat_lr': kwargs.get('mirostat_lr'),
        'mirostat_ent': kwargs.get('mirostat_ent'),
        'logits_all': True
    }

    # Remove any None values from the dictionary
    llama_args = {k: v for k, v in llama_args.items() if v is not None}

    return llama_args
    

def estimate_disk_size(total_chunks, context_size, vocab_size, precision):
    """Estimates the total disk size based on number of chunks and vocab size."""
    bytes_per_logit = max(2, precision // 8)  # float32 is 4 bytes
    # One logit vector per chunk
    total_bytes = total_chunks * context_size * vocab_size * bytes_per_logit
    estimated_total_disk_size = total_bytes / (1024 ** 3)  # Convert to GB
    logger.info(f"Estimated total disk size (before compression): {estimated_total_disk_size:.2f} GB")


def write_header(h5f, context_size, vocab_size):
    """Writes metadata as attributes to the HDF5 file."""
    h5f.attrs['format'] = f"generate_logits_v{__version__}"
    h5f.attrs['n_ctx'] = context_size
    h5f.attrs['n_vocab'] = vocab_size
    logger.debug(f"Header written with context size: {context_size} and vocab size: {vocab_size}")


def create_processed_chunks_dataset(h5f, total_chunks):
    """Creates the 'processed_chunks' dataset in the HDF5 file."""
    processed_chunks_dset = h5f.create_dataset(
        'processed_chunks',
        shape=(total_chunks,),
        dtype=bool,
        maxshape=(None,),
        chunks=True
    )
    # Initialize processed_chunks to False
    processed_chunks_dset[...] = False

    return processed_chunks_dset

def create_hdf5_dataset(output_file, total_chunks, vocab_size, context_size, precision, resume=False):
    """Creates and returns an HDF5 file and datasets for storing logits and processed chunk flags."""
    global TARGET_CHUNK_SIZE_BYTES

    if resume and os.path.exists(output_file):
        logger.debug(f"Resuming with existing HDF5 file: {output_file}")

        h5f = h5py.File(output_file, 'a')
        dset = h5f['logits']
        if 'processed_chunks' in h5f:
            processed_chunks_dset = h5f['processed_chunks']
        else:
            logger.debug("Creating missing 'processed_chunks' dataset for resumable processing")
            processed_chunks_dset = create_processed_chunks_dataset(h5f, total_chunks)
    else:
        logger.debug(f"Creating HDF5 dataset with vocab_size: {vocab_size}")

        vocab_size = int(vocab_size)  # Ensure vocab_size is an integer
        dtype = 'float16' if precision <= 16 else 'float32'
        BYTES_PER_FLOAT = 4 if dtype == 'float32' else 2

        # Calculate max allowable `context_size` to keep chunk size < TARGET_CHUNK_SIZE_BYTES
        max_context_size = min(context_size, TARGET_CHUNK_SIZE_BYTES // (vocab_size * BYTES_PER_FLOAT))

        # Open the file and write the header before creating the dataset
        h5f = h5py.File(output_file, 'w')

        # Write the header to store metadata
        write_header(h5f, context_size, vocab_size)

        # Create the logits dataset
        dset = h5f.create_dataset(
            'logits',
            shape=(total_chunks, context_size, vocab_size),
            maxshape=(None, context_size, vocab_size),
            dtype=dtype,
            chunks=(1, max_context_size, vocab_size),
            compression="gzip"
        )

        # Create the 'processed_chunks' dataset
        processed_chunks_dset = create_processed_chunks_dataset(h5f, total_chunks)

    return h5f, dset, processed_chunks_dset


def process_tokens_chunk(model, tokens_chunk, dset, chunk_index):
    """
    Processes a single chunk of tokens and captures timing for the full chunk processing.
    """
    start_time = time.time()

    # Insert stream tokens if required
    bool_map = {"true": True, "false": False}

    bos = model.token_bos()
    add_bos_token = model.metadata.get("tokenizer.ggml.add_bos_token", "true")
    require_bos = bool_map.get(add_bos_token, True)
    logger.debug("require_bos: add_bos_token=%s, require_bos=%s", add_bos_token, require_bos)
    if require_bos and bos is not None:
        tokens_chunk.insert(0, bos)

    eos = model.token_eos()
    add_eos_token = model.metadata.get("tokenizer.ggml.add_eos_token", "true")
    require_eos = bool_map.get(add_eos_token, True)
    logger.debug("require_eos: add_eos_token=%s, require_eos=%s", add_eos_token, require_eos)
    if require_eos and eos is not None:
        tokens_chunk.append(eos)

    # Measure model inference time
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        _ = model(tokens_chunk)
    inference_time = (time.time() - start_time) * 1000  # ms
    logger.debug("Inference time: %.2f ms", inference_time)

    start_hdf5_time = time.time()

    errors = 0

    BYTES_PER_FLOAT = 4 if dset.dtype == np.float32 else 2

    # Calculate the number of logits that fit in the target memory limit
    vocab_size = dset.shape[2]
    buffer_size = TARGET_CHUNK_SIZE_BYTES // (vocab_size * BYTES_PER_FLOAT)
    logit_count = model.n_tokens  # Directly using `n_tokens` instead of len(eval_logits)
    logger.debug(f"Logits shape {model.scores.shape} dtype {model.scores.dtype}")

    # Iterate through `self.scores` in chunks, avoiding intermediate list creation
    for i in range(0, logit_count, buffer_size):
        # Access a batch of logits directly from `self.scores`
        logits_buffer = model.scores[i : i + buffer_size, :]

        # Check for NaNs in the current buffer
        if np.any(np.isnan(logits_buffer)):
            logger.warning(f"NaN detected in logits at chunk {chunk_index}, batch starting at token index {i}.")
            errors = 1

        # Write the buffered logits directly to HDF5 without converting to a list
        dset[chunk_index, i : i + logits_buffer.shape[0], :] = logits_buffer

    # Measure HDF5 write time (if needed)
    hdf5_time = (time.time() - start_hdf5_time) * 1000  # ms

    # Measure garbage collection time
    start_gc_time = time.time()
    gc.collect()
    gc_time = (time.time() - start_gc_time) * 1000  # ms

    total_time = (time.time() - start_time) * 1000  # ms
    accounted_time = inference_time + hdf5_time + gc_time
    unaccounted_time = total_time - accounted_time

    return {
        'chunk_index': chunk_index + 1,
        'total_time': total_time,
        'inference_time': inference_time,
        'hdf5_time': hdf5_time,
        'gc_time': gc_time,
        'unaccounted_time': unaccounted_time,
        'errors': errors
    }


def generate_logits_with_llama_cpp(**kwargs):
    """Main function with corrected chunk processing and context size handling."""
    errors = 0

    # Handle `--clobber` flag
    if kwargs.get('clobber', False) and os.path.exists(kwargs['output']):
        os.remove(kwargs['output'])
        logger.info(f"Existing output file {kwargs['output']} removed due to --clobber flag.")

    resume = os.path.exists(kwargs['output']) and not kwargs.get('clobber', False)

    model = llama_cpp.Llama(**prepare_llama_args(kwargs))
    
    # Get vocab_size
    vocab_size = model.n_vocab() if callable(getattr(model, 'n_vocab', None)) else model.n_vocab
    assert isinstance(vocab_size, int)
    logger.debug(f"Number of logits: {vocab_size}.")

    # Read the entire text
    with open(kwargs['dataset'], 'r', encoding='utf-8') as f:
        text_data = f.read()
    encoded_text = text_data.encode('utf-8')

    # Load or tokenize tokens
    tokens_file = kwargs['dataset'] + '.tokens.npy'
    if os.path.exists(tokens_file):
        tokens = np.load(tokens_file)
        total_tokens = len(tokens)
        logger.info(f"Loaded precomputed tokens from {tokens_file}")
    else:
        # Tokenize the entire text
        tokens = model.tokenize(encoded_text)
        total_tokens = len(tokens)
        logger.info(f"Tokenized dataset, total tokens: {total_tokens}")
        np.save(tokens_file, tokens)
        logger.info(f"Saved tokens to {tokens_file}")

    # Calculate total chunks
    bos = model.token_bos()
    b = 1 if bos is not None else 0
    eos = model.token_eos()
    b += 1 if eos is not None else 0

    # Create chunks, dynamically adjusting size
    chunk_size = kwargs['context_size'] - b
    total_chunks = (total_tokens + chunk_size - 1) // chunk_size  # Round up
    assert(total_chunks > 0)

    # Adjust `from` and `to` chunk indices
    start_chunk = kwargs.get('from_chunk', 0)
    end_chunk = kwargs.get('to_chunk', total_chunks - 1)
    if end_chunk > total_chunks - 1:
        end_chunk = total_chunks - 1
    if start_chunk > end_chunk:
        logger.error(f"Invalid chunk range: from {start_chunk} to {end_chunk}")
        return
    logger.info(f"Processing chunks from {start_chunk} to {end_chunk}")

    precision = kwargs['precision'] if kwargs.get('precision') is not None \
        else estimate_model_precision(kwargs['model'])

    # Create HDF5 dataset

    h5f, dset, processed_chunks_dset = create_hdf5_dataset(kwargs['output'], total_chunks, vocab_size, kwargs['context_size'], precision, resume=resume)

    try:
        total_chunks_processed = 0
        errors = 0
        timing_logs = []

        # Process the chunks
        for chunk_index in range(start_chunk, end_chunk + 1):
            if processed_chunks_dset[chunk_index]:
                logger.info(f"Skipping chunk {chunk_index} as it has already been processed.")
                continue
            start_index = chunk_index * chunk_size
            end_index = min((chunk_index + 1) * chunk_size, total_tokens)
            tokens_chunk = tokens[start_index:end_index]

            # Process the chunk and collect timing information
            timing_info = process_tokens_chunk(model, tokens_chunk, dset, chunk_index)
            total_chunks_processed += 1
            errors += timing_info['errors']

            # Mark the chunk as processed
            processed_chunks_dset[chunk_index] = True
            h5f.flush()  # Ensure data is written to disk

            timing_logs.append(timing_info)

            if logger.level == logging.DEBUG:
                logger.info(f"[{timing_info['chunk_index']}] {timing_info['total_time']:.2f} ms (inference time: {timing_info['inference_time']:.2f} ms, HDF5 time: {timing_info['hdf5_time']:.2f} ms, GC time: {timing_info['gc_time']:.2f} ms, unaccounted: {timing_info['unaccounted_time']:.2f} ms)")
            else:
                print(f"[{timing_info['chunk_index']}] {timing_info['total_time']:.2f} ms", end=' ', flush=True)

            # Estimate runtime after processing the first chunk
            if total_chunks_processed == 1:
                avg_chunk_time = timing_info['total_time']
                remaining_chunks = (end_chunk - chunk_index)
                estimated_runtime = (avg_chunk_time * remaining_chunks) / (60 * 1000)  # Convert ms to minutes
                logger.info(f"\nEstimated runtime: {estimated_runtime:.2f} minutes for {remaining_chunks} remaining chunks")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")

    finally:
        # Ensure the HDF5 file is closed if it was opened
        if total_chunks_processed == 0:
            logger.info(f"No new chunks were processed. All chunks in the specified range have been processed.")
        else:
            logger.info(f"\nProcessed {total_chunks_processed} chunks")
            if errors > 0:
                logger.warning(f"Total errors detected during logit generation: {errors}")
            logger.info(f"Final file size: {os.path.getsize(kwargs['output']) / (1024 * 1024):.2f} MB")
        h5f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate logits and save them to an HDF5 file.")
    parser.add_argument('--model', type=str, required=True, help='Path to the GGUF model file.')
    parser.add_argument('--context-size', type=int, required=False, help="The model's context size.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.txt file.')
    parser.add_argument('--output', type=str, default="logits.h5", help='Output file for logits.')
    parser.add_argument('--n-gpu-layers', type=int, default=None, help='Number of layers to store in VRAM.')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() - 1), help='Number of threads to use for parallel processing (default: system threads - 1)')
    parser.add_argument('--batch-size', type=int, default=2048, help='Logical maximum batch size (default: 2048)')
    parser.add_argument('--ubatch-size', type=int, default=512, help='Physical maximum batch size (default: 512)')
    parser.add_argument('--precision', type=int, choices=[16,32], default=None, help='Precision of the model weights and activations (default: auto-estimated, POSSIBLY INCORRECT).')

    parser.add_argument('--from', dest='from_chunk', type=int, default=0, help="Optional starting chunk index for processing (default: 0)")
    parser.add_argument('--to', dest='to_chunk', type=int, help="Optional ending chunk index for processing (default: last chunk)")
    parser.add_argument('--clobber', action='store_true', help="Overwrite existing output file")

    parser.add_argument('--rope-freq-base', type=float, default=None, help='ROPE frequency base. (default: automatically assigned)')
    parser.add_argument('--repeat-last-n', type=int, default=64, help='Last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)')
    parser.add_argument('--repeat-penalty', type=float, default=1.0, help='Penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)')
    parser.add_argument('--presence-penalty', type=float, default=0.0, help='Repeat alpha presence penalty (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--frequency-penalty', type=float, default=0.0, help='Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--dynatemp-range', type=float, default=0.0, help='Dynamic temperature range (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--dynatemp-exp', type=float, default=1.0, help='Dynamic temperature exponent (default: 1.0)')
    parser.add_argument('--mirostat', type=int, default=0, help='Use Mirostat sampling. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)')
    parser.add_argument('--mirostat-lr', type=float, default=0.1, help='Mirostat learning rate, parameter eta (default: 0.1)')
    parser.add_argument('--mirostat-ent', type=float, default=5.0, help='Mirostat target entropy, parameter tau (default: 5.0)')

    parser.add_argument(
        '--verbosity',
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level (default: INFO)"
    )

    args = parser.parse_args()
    args_dict = vars(args)

    logger.setLevel(getattr(logging, args.verbosity.upper(), logging.INFO))
    logging.info(f"generate_logits starting (version {__version__})")

    generate_logits_with_llama_cpp(**args_dict)
