import contextlib
import gc
import h5py
import llama_cpp
import logging
import numpy as np
import os
import time

from version import __version__
from gguf_optimize_logging import setup_logging


logger = logging.getLogger(__name__)

SKIPPED_GENERATION = "skipped"
TARGET_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
H5PY_SUPPORTED_COMPRESSIONS = {
    'gzip': {'level': 4},  # Default is 4, range is 0 (no compression) to 9 (maximum compression)
    'lzf': {},            # No parameters required
    #'scaleoffset': {'scale_factor': 4, 'min_bits': 0},  # Default scale factor and min bits,
    'none': {}
}

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


def prepare_call_args(kwargs):
    """
    Prepare a dictionary of arguments for the Llama __call__ method from a given kwargs dictionary,
    including only those defined in __main__ and accepted by Llama's __call__ method.
    """
    call_args = {
        'temperature': kwargs.get('temp'),
        'top_k': kwargs.get('top_k'),
        'top_p': kwargs.get('top_p'),
        'min_p': kwargs.get('min_p'),
        'repeat_penalty': kwargs.get('repeat_penalty'),
        'presence_penalty': kwargs.get('presence_penalty'),
        'frequency_penalty': kwargs.get('frequency_penalty'),
        'seed': kwargs.get('seed'),
        'mirostat_mode': kwargs.get('mirostat'),
        'mirostat_tau': kwargs.get('mirostat_ent'),
        'mirostat_eta': kwargs.get('mirostat_lr'),
    }

    # Remove any None values to avoid passing undefined arguments
    call_args = {k: v for k, v in call_args.items() if v is not None}
    
    return call_args


def verify_model_context_size(model, sample_text="This is a test.", padding=2):
    """
    Verify the actual context size needed for a model by analyzing a test generation.
    
    Args:
        model: llama_cpp.Llama instance
        sample_text: Short text to use for verification
        padding: Extra tokens to account for (e.g., BOS+EOS tokens)
        
    Returns:
        tuple: (actual_context_size, requires_bos, requires_eos)
    """
    # Get tokenizer metadata
    bool_map = {"true": True, "false": False}
    bos = model.token_bos()
    eos = model.token_eos()
    
    # Check tokenizer settings
    add_bos_token = model.metadata.get("tokenizer.ggml.add_bos_token", "true")
    add_eos_token = model.metadata.get("tokenizer.ggml.add_eos_token", "true")
    require_bos = bool_map.get(add_bos_token, True)
    require_eos = bool_map.get(add_eos_token, True)
    
    # Tokenize sample text
    encoded_text = sample_text.encode('utf-8')
    tokens = model.tokenize(encoded_text)
    
    # Add special tokens if required
    if require_bos and bos is not None:
        tokens.insert(0, bos)
    if require_eos and eos is not None:
        tokens.append(eos)
    
    # Generate and check actual logit size
    _ = model(tokens)  # Generate logits
    actual_logit_size = model.scores.shape[0]
    logger.debug(f"Generated logits shape {model.scores.shape}, dtype {model.scores.dtype}")
    
    # Calculate effective context size
    special_tokens = (1 if require_bos and bos is not None else 0) + \
                    (1 if require_eos and eos is not None else 0)
    effective_context_size = actual_logit_size + padding
    
    return {
        'effective_context_size': effective_context_size,
        'requires_bos': require_bos and bos is not None,
        'requires_eos': require_eos and eos is not None,
        'special_tokens_count': special_tokens,
        'logit_size': actual_logit_size
    }


def calculate_chunk_parameters(model, requested_context_size=None):
    """
    Calculate the optimal chunk parameters for logit generation.
    
    Args:
        model: llama_cpp.Llama instance
        requested_context_size: Optional user-specified context size
        
    Returns:
        dict: Chunk parameters including sizes and token handling info
    """
    # First verify the model's actual behavior
    verification = verify_model_context_size(model)
    
    # If no context size specified, use the verified effective size
    if requested_context_size is None:
        context_size = verification['effective_context_size']
    else:
        context_size = min(requested_context_size, verification['effective_context_size'])
    
    # Calculate the actual chunk size for tokens
    chunk_size = context_size - verification['special_tokens_count']
    
    return {
        'context_size': context_size,
        'chunk_size': chunk_size,
        'requires_bos': verification['requires_bos'],
        'requires_eos': verification['requires_eos'],
        'special_tokens_count': verification['special_tokens_count'],
        'verified_logit_size': verification['logit_size']
    }


def write_header(h5f, context_size, vocab_size, total_chunks):
    """Writes metadata as attributes to the HDF5 file."""
    h5f.attrs['format'] = f"generate_logits_v{__version__}"
    h5f.attrs['n_ctx'] = context_size
    h5f.attrs['n_vocab'] = vocab_size
    h5f.attrs['total_chunks'] = total_chunks

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


def create_hdf5_datasets(h5f, total_chunks, vocab_size, context_size, precision, compression, resume=False):
    """Creates and returns datasets for storing logits and processed chunk flags."""
    global TARGET_CHUNK_SIZE_BYTES

    if resume:
        logger.debug(f"Resuming with existing HDF5 file: {h5f.filename}")
        dset = h5f['logits']
        
        if 'processed_chunks' in h5f:
            processed_chunks_dset = h5f['processed_chunks']
        else:
            logger.debug("Creating missing 'processed_chunks' dataset for resumable processing")
            processed_chunks_dset = create_processed_chunks_dataset(h5f, total_chunks)

        if 'freed_chunks' in h5f:
            freed_chunks_dset = h5f['freed_chunks']
        else:
            logger.debug("Creating missing 'freed_chunks' dataset for resumable processing")
            freed_chunks_dset = h5f.create_dataset(
                'freed_chunks',
                shape=(0,),
                dtype=np.int64,
                maxshape=(None,),
                chunks=True
            )

        # Create chunk_index dataset if it doesn't already exist
        if 'chunk_index' not in h5f:
            logger.debug("Creating missing 'freed_chunks' dataset for resumable processing")
            chunk_index_dset = h5f.create_dataset(
                'chunk_index',
                shape=(total_chunks,),
                dtype=np.int64,
                chunks=True
            )
            chunk_index_dset[...] = -1  
        else:
            chunk_index_dset = h5f['chunk_index']

    else:
        logger.debug(f"Creating HDF5 dataset with vocab_size: {vocab_size}")
        
        vocab_size = int(vocab_size)  
        dtype = 'float16' if precision <= 16 else 'float32'
        BYTES_PER_FLOAT = 4 if dtype == 'float32' else 2

        max_context_size = min(context_size, TARGET_CHUNK_SIZE_BYTES // (vocab_size * BYTES_PER_FLOAT))

        write_header(h5f, context_size, vocab_size, total_chunks)

        # Create the logits dataset    
        dset = h5f.create_dataset(
            'logits',
            shape=(total_chunks, context_size, vocab_size),
            maxshape=(None, context_size, vocab_size),
            dtype=dtype,
            chunks=(1, max_context_size, vocab_size),
            compression=compression if compression != "none" else None
        )

        # Create the freed chunks list dataset
        freed_chunks_dset = h5f.create_dataset(
            'freed_chunks',
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
            chunks=True
        )

        processed_chunks_dset = create_processed_chunks_dataset(h5f, total_chunks)

        chunk_index_dset = h5f.create_dataset(
            'chunk_index',
            shape=(total_chunks,),
            dtype=np.int64,
            chunks=True
        )
        chunk_index_dset[...] = -1  # Initialize with -1 to indicate empty slots


    return dset, processed_chunks_dset, freed_chunks_dset, chunk_index_dset


def calculate_special_token_requirements(model):
    """
    Determine if the model requires BOS and EOS tokens based on its metadata.
    """
    bool_map = {"true": True, "false": False}
    require_bos = bool_map.get(model.metadata.get("tokenizer.ggml.add_bos_token", "true"), True)
    require_eos = bool_map.get(model.metadata.get("tokenizer.ggml.add_eos_token", "true"), True)

    return require_bos, require_eos


def calculate_total_chunks(total_tokens, context_size, model):
    """
    Calculates the total number of chunks for a given dataset and context size.
    Adjusts for special tokens (BOS and EOS) based on model requirements.
    """
    require_bos, require_eos = calculate_special_token_requirements(model)

    bos = model.token_bos()
    eos = model.token_eos()

    special_token_count = (1 if require_bos and bos is not None else 0) + \
                          (1 if require_eos and eos is not None else 0)

    chunk_size = context_size - special_token_count
    total_chunks = (total_tokens + chunk_size - 1) // chunk_size
    return total_chunks


def get_total_chunks(model_path, dataset_path, context_size):
    """
    Returns the total number of chunks for the given model and dataset without processing them.
    """
    model_args = {'model_path': model_path, 'logits_all': True}
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        model = llama_cpp.Llama(**model_args)

    tokens, total_tokens = tokenize_dataset(model, dataset_path)
    total_chunks = calculate_total_chunks(total_tokens, context_size, model)

    return total_chunks


def tokenize_dataset(model, dataset_path):
    """
    Tokenizes the dataset and returns the tokens along with their count.
    If precomputed tokens exist, they are loaded instead.
    """
    tokens_file = dataset_path + '.tokens.npy'
    if os.path.exists(tokens_file):
        tokens = np.load(tokens_file).tolist()
        total_tokens = len(tokens)
        logger.info(f"Loaded precomputed tokens from {tokens_file}")
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        encoded_text = text_data.encode('utf-8')

        tokens = model.tokenize(encoded_text)
        total_tokens = len(tokens)
        logger.info(f"Tokenized dataset, total tokens: {total_tokens}")
        np.save(tokens_file, tokens)
        logger.info(f"Saved tokens to {tokens_file}")

    return tokens, total_tokens


def process_single_chunk(model, call_args, tokens_chunk, dset, chunk_index, freed_chunks_dset, chunk_index_dset):
    """
    Processes a single chunk of tokens and captures timing for the full chunk processing.
    """
    start_time = time.time()

    bool_map = {"true": True, "false": False}
    bos = model.token_bos()
    add_bos_token = model.metadata.get("tokenizer.ggml.add_bos_token", "true")
    require_bos = bool_map.get(add_bos_token, True)
    logger.debug("require_bos: add_bos_token=%s (require_bos=%s)", add_bos_token, require_bos)
    if require_bos and bos is not None:
        tokens_chunk.insert(0, bos)

    eos = model.token_eos()
    add_eos_token = model.metadata.get("tokenizer.ggml.add_eos_token", "true")
    require_eos = bool_map.get(add_eos_token, True)
    logger.debug("require_eos: add_eos_token=%s (require_eos=%s)", add_eos_token, require_eos)
    if require_eos and eos is not None:
        tokens_chunk.append(eos)

    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        llama_cpp.llama_cpp.llama_kv_cache_clear(model.ctx)
        _ = model(tokens_chunk, **call_args)

    inference_time = (time.time() - start_time) * 1000  # ms
    logger.debug("Inference time: %.2f ms", inference_time)

    start_hdf5_time = time.time()

    errors = 0

    bytes_per_float = 4 if dset.dtype == np.float32 else 2

    vocab_size = dset.shape[2]
    buffer_size = TARGET_CHUNK_SIZE_BYTES // (vocab_size * bytes_per_float)
    logit_count = model.n_tokens  # Directly using `n_tokens` instead of len(eval_logits)
    logger.debug(f"Logits shape {model.scores.shape} dtype {model.scores.dtype}")

    # Update chunk_index with the logical index of this chunk
    if freed_chunks_dset.size > 0:  # Reuse freed chunk
        freed_chunk_index = freed_chunks_dset[0]
        physical_index = chunk_index_dset[freed_chunk_index]
        freed_chunks_dset.resize(freed_chunks_dset.shape[0] - 1, axis=0)
        logger.debug(f"Reusing freed chunk {physical_index} for chunk {chunk_index}.")
    else:
        physical_index = chunk_index
        logger.debug(f"No freed chunks available, using new chunk index {physical_index}.")

    chunk_index_dset[chunk_index] = physical_index  # Store the logical index in the dataset
    logger.debug(f"Written chunk {chunk_index} at physical slot {physical_index}")

    for i in range(0, logit_count, buffer_size):
        logits_buffer = model.scores[i : i + buffer_size, :]

        if np.any(np.isnan(logits_buffer)):
            logger.warning(
                f"NaN detected in logits at chunk {chunk_index}, batch starting at token index {i}."
            )
            errors = 1

        dset[physical_index, i : i + logits_buffer.shape[0], :] = logits_buffer

    hdf5_time = (time.time() - start_hdf5_time) * 1000  # ms

    start_gc_time = time.time()
    gc.collect()
    gc_time = (time.time() - start_gc_time) * 1000  # ms

    total_time = (time.time() - start_time) * 1000  # ms
    accounted_time = inference_time + hdf5_time + gc_time
    unaccounted_time = total_time - accounted_time

    return {
        'chunk_index': chunk_index,
        'total_time': total_time,
        'inference_time': inference_time,
        'hdf5_time': hdf5_time,
        'gc_time': gc_time,
        'unaccounted_time': unaccounted_time,
        'errors': errors
    }


def process_all_chunks(
        h5f, model, call_args, tokens, total_tokens, start_chunk, end_chunk, chunk_size, timing_logs):
    total_chunks_processed = 0
    errors = 0

    processed_chunks_dset = h5f['processed_chunks']
    freed_chunks_dset = h5f['freed_chunks']
    chunk_index_dset = h5f['chunk_index']
    logits_dset = h5f['logits']

    for chunk_index in range(start_chunk, end_chunk + 1):
        if processed_chunks_dset[chunk_index]:
            logger.info(f"Skipping chunk {chunk_index} as it has already been processed.")
            continue

        # Define the chunk boundaries
        tokens_chunk = tokens[chunk_index * chunk_size : min((chunk_index + 1) * chunk_size, total_tokens)]

        timing_info = process_single_chunk(model, call_args, tokens_chunk, logits_dset, chunk_index, freed_chunks_dset, chunk_index_dset)
        errors += timing_info['errors']
        total_chunks_processed += 1

        processed_chunks_dset[chunk_index] = True
        h5f.flush()
        timing_logs.append(timing_info)

        if total_chunks_processed == 1:
            avg_chunk_time = timing_info['total_time']
            remaining_chunks = end_chunk - chunk_index
            estimated_runtime = (avg_chunk_time * remaining_chunks) / (60 * 1000)
            logger.info(f"Estimated runtime: {estimated_runtime:.2f} minutes for {remaining_chunks} remaining chunks")

        if logger.level == logging.DEBUG:
            logger.info(
                f"[{timing_info['chunk_index']}] {timing_info['total_time']:.2f} ms "
                f"(inference time: {timing_info['inference_time']:.2f} ms, "
                f"HDF5 time: {timing_info['hdf5_time']:.2f} ms, "
                f"GC time: {timing_info['gc_time']:.2f} ms, "
                f"unaccounted: {timing_info['unaccounted_time']:.2f} ms)"
            )
        else:
            print(f"[{timing_info['chunk_index']}] {timing_info['total_time']:.2f} ms", end=' ', flush=True)

    return total_chunks_processed, errors


def get_model(**kwargs):
    model_args = prepare_llama_args(kwargs)
    if logger.level == logging.DEBUG:
        model = llama_cpp.Llama(**model_args)
    else:
        with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f): 
            model = llama_cpp.Llama(**model_args)

    return model

def generate_logits_with_llama_cpp(**kwargs):
    errors = 0

    # Handle `--clobber` flag
    if kwargs.get('clobber', False) and os.path.exists(kwargs['output']):
        os.remove(kwargs['output'])
        logger.info(f"Existing output file {kwargs['output']} removed due to --clobber flag.")

    resume = os.path.exists(kwargs['output']) and not kwargs.get('clobber', False)

    model_args = prepare_llama_args(kwargs)
    if logger.level == logging.DEBUG:
        model = llama_cpp.Llama(**model_args)
    else:
        with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f): 
            model = llama_cpp.Llama(**model_args)

    vocab_size = model.n_vocab() if callable(getattr(model, 'n_vocab', None)) else model.n_vocab
    assert isinstance(vocab_size, int), "vocab_size should be an integer"
    logger.debug(f"Number of logits: {vocab_size}.")

    with open(kwargs['dataset'], 'r', encoding='utf-8') as f:
        text_data = f.read()
    encoded_text = text_data.encode('utf-8')

    tokens_file = kwargs['dataset'] + '.tokens.npy'
    if os.path.exists(tokens_file):
        tokens = np.load(tokens_file).tolist()
        total_tokens = len(tokens)
        logger.info(f"Loaded precomputed tokens from {tokens_file}")
    else:
        tokens = model.tokenize(encoded_text)
        total_tokens = len(tokens)
        logger.info(f"Tokenized dataset, total tokens: {total_tokens}")
        np.save(tokens_file, tokens)
        logger.info(f"Saved tokens to {tokens_file}")

    bos = model.token_bos()
    eos = model.token_eos()
    chunk_size = kwargs['context_size'] - (1 if bos is not None else 0) - (1 if eos is not None else 0)
    tokens, total_tokens = tokenize_dataset(model, kwargs['dataset'])
    total_chunks = calculate_total_chunks(total_tokens, kwargs['context_size'], model)

    start_chunk = kwargs.get('from_chunk', 0)
    end_chunk = kwargs.get('to_chunk', total_chunks - 1)
    if start_chunk > end_chunk:
        logger.error(f"Invalid chunk range: from {start_chunk} to {end_chunk}")
        return
    logger.info(f"Processing chunks from {start_chunk} to {end_chunk}")

    precision = kwargs['precision']
    total_chunks_processed = 0
    timing_logs = []

    try:
        # Open file in a main try block, with all interrupts handled here
        with h5py.File(kwargs['output'], 'a' if resume else 'w') as h5f:
            # Create datasets and process chunks
            create_hdf5_datasets(
                h5f, total_chunks, vocab_size, kwargs['context_size'], precision, kwargs['compression'], resume=resume
            )

            # List to collect timing logs
            timing_logs = []
            call_args = prepare_call_args(kwargs)
            total_chunks_processed, errors = process_all_chunks(
                h5f, model, call_args, tokens, total_tokens, start_chunk, 
                end_chunk, chunk_size, timing_logs
            )

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user. Saving progress and exiting.")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")

    finally:
        # Ensure that the HDF5 file is flushed and closed properly
        if 'h5f' in locals() and h5f:
            try:
                h5f.flush()
                h5f.close()
            except Exception as e:
                logger.error(f"Error closing HDF5 file: {e}")

    # Final logging
    if total_chunks_processed == 0:
        logger.info("No new chunks were processed. All chunks in the specified range have been processed.")
        return SKIPPED_GENERATION
    else:
        print("") # ensure INFO starts on a new line after the [chunk number] <duration>
        logger.info(f"Processed {total_chunks_processed} chunks")
        if errors > 0:
            logger.warning(f"Total errors detected during logit generation: {errors}")
        logger.info(f"Final file size: {os.path.getsize(kwargs['output']) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate logits and save them to an HDF5 file.")
    parser.add_argument('--model', type=str, required=True, help='Path to the GGUF model file.')
    parser.add_argument('--context-size', type=int, required=False, help="The model's context size.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.txt file.')
    parser.add_argument('--output', type=str, default="logits.h5", help='Output file for logits.')
    parser.add_argument('--n-gpu-layers', type=int, default=None, help='Number of layers to store in VRAM.')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() - 1), help='Number of threads to use for parallel processing (default: system threads - 1)')
    parser.add_argument('--batch-size', type=int, help='Logical maximum batch size (default: context size)')
    parser.add_argument('--ubatch-size', type=int, help='Physical maximum batch size (default: context size)')
    parser.add_argument('--precision', type=int, choices=[16,32], default=32, help='Model\'s activation precision  (default: 32) note: currently llama.cpp only supports fp32 for processing the output weights, so this will be fp32.')
    parser.add_argument('--compression', type=str, choices=H5PY_SUPPORTED_COMPRESSIONS.keys(), default=None, help='Compression method to use for the output logits file. (Default: None)')

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
    parser.add_argument('--top-p', type=float, default=0, help='Top-p sampling threshold')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k sampling threshold')
    parser.add_argument('--temp', type=float, default=0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    parser.add_argument(
        '--verbosity',
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level (default: INFO)"
    )

    args = parser.parse_args()
    args_dict = vars(args)

    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))
    logging.info(f"generate_logits starting (version {__version__})")

    if args_dict['batch_size'] is None:
        logger.debug("Setting batch size to context size: %s", args_dict['context_size'])
        args_dict['batch_size'] = args.context_size
    if args_dict['ubatch_size'] is None:
        logger.debug("Setting μbatch size to context size: %s", args_dict['context_size'])
        args_dict['ubatch_size'] = args.context_size

    generate_logits_with_llama_cpp(**args_dict)
