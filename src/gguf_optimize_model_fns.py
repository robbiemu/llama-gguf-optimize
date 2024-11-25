import contextlib
import llama_cpp
import os
import logging

from version import __version__


logger = logging.getLogger(__name__)

def estimate_model_parameters(metadata):
    # Extract and validate metadata, converting to int if necessary
    try:
        vocab_size, embedding_length, feed_forward_length, num_layers = [
            int(metadata.get(key)) for key in [
                "llama.vocab_size", "llama.embedding_length", 
                "llama.feed_forward_length", "llama.block_count"
            ]
        ]
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid metadata for parameter estimation: {e}")
        return None

    # Embedding parameters
    embedding_params = vocab_size * embedding_length

    # Self-attention and feed-forward parameters
    layer_params_per_layer = 4 * embedding_length**2 + 4 * embedding_length * feed_forward_length

    # Total parameters = embedding parameters + layer parameters across all layers
    total_params = embedding_params + (num_layers * layer_params_per_layer)
    logger.debug(f"Estimated number of parameters: {total_params}")

    return total_params


def estimate_model_precision(model_path=None, model=None):
    """ DEPRECATED v0.5.3 """
    try:
        if model is None:
            with open(os.devnull, 'w') as f_null, contextlib.redirect_stderr(f_null), contextlib.redirect_stdout(f_null):
                model = llama_cpp.Llama(model_path)

        # Estimate number of parameters based on the architecture metadata
        num_params = estimate_model_parameters(model.metadata)

        if num_params is None or num_params == 0:
            logger.warning("Unable to estimate number of parameters. Defaulting to 32.0 bits.")
            return 32

        # Get file size in bytes
        file_size_bytes = os.path.getsize(model_path)

        # Calculate bits per weight
        bits_per_weight = (file_size_bytes * 8) / num_params
        logger.info(f"Estimated Model Precision: {bits_per_weight} bits per weight")

        return bits_per_weight

    except FileNotFoundError:
        logger.error(f"GGUF file not found at path: {model_path}. Defaulting to 32.0 bits.")

        return 32
    except Exception as e:
        logger.error(f"An error occurred while processing the GGUF file: {e}. Defaulting to 32.0 bits.")

        return 32
