from collections import namedtuple
import contextlib
import json
import llama_cpp
import logging
import math
import multiprocessing as mp
import numpy as np
import optuna
import os
import psutil
import random
from scipy.stats import norm
import string
import time
import torch

from version import __version__
from gguf_optimize_model_fns import estimate_model_precision


logger = logging.getLogger(__name__)

ExponentRange = namedtuple('ExponentRange', ['min', 'max'])

DEFAULT_BATCH_EXPONENT = 11  # 2^11 = 2048
DEFAULT_UBATCH_EXPONENT = 9  # 2^9 = 512
PROBABILITY_THRESHOLD = 0.95

trial_cache = {}
near_best_trials = []
prior_mean = None
prior_variance = None
epsilon = 1e-6

if torch.cuda.is_available():
    torch.cuda.init()  
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()  
    torch.mps.synchronize()
else:
    device = torch.device("cpu")


def update_bayesian_mean_variance(prior_mean, prior_variance, new_data):
    global epsilon

    likelihood_mean = np.mean(new_data)
    n = len(new_data)
    if n > 1:
        sample_variance = np.var(new_data, ddof=1)
        likelihood_variance = sample_variance / n
    else:
        likelihood_variance = epsilon

    # Ensure prior_variance is numeric and positive
    if prior_variance is None or not np.issubdtype(type(prior_variance), np.number) or not np.isfinite(prior_variance) or prior_variance <= 0:
        prior_variance = epsilon

    # Calculate posterior mean and variance
    posterior_mean = (prior_variance * likelihood_mean + likelihood_variance * prior_mean) / (prior_variance + likelihood_variance)
    posterior_variance = (prior_variance * likelihood_variance) / (prior_variance + likelihood_variance)

    return posterior_mean, posterior_variance


def calculate_probability_of_superiority(current_best_mean, current_best_variance, trial_mean, trial_variance):
    global epsilon

    diff_mean = trial_mean - current_best_mean
    diff_variance = trial_variance + current_best_variance

    if not np.isfinite(diff_variance) or diff_variance <= 0:
        logger.warning("Variance is zero or negative; adjusting variance.")
        diff_variance = epsilon

    prob_superiority = norm.cdf(0, loc=diff_mean, scale=np.sqrt(diff_variance))
    return prob_superiority


def update_best_chunk_time_with_probability(trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch):
    global near_best_trials, prior_mean, prior_variance

    # Filter out np.inf values caused by early termination
    trial_chunk_times = [t for t in trial_chunk_times if t != np.inf]
    best_chunk_times = [t for t in best_chunk_times if t != np.inf] if best_chunk_times else []

    # Calculate average chunk times
    trial_avg_chunk_time = np.mean(trial_chunk_times) if trial_chunk_times else float('inf')
    best_avg_chunk_time = np.mean(best_chunk_times) if best_chunk_times else float('inf')

    # Calculate sample variances
    n_trial = len(trial_chunk_times)
    n_best = len(best_chunk_times)

    trial_variance = (np.var(trial_chunk_times, ddof=1) / n_trial) if n_trial > 1 else 1e6
    best_variance = (np.var(best_chunk_times, ddof=1) / n_best) if n_best > 1 else 1e6

    # Initialize prior_mean and prior_variance if they are None
    if prior_mean is None or prior_variance is None:
        prior_mean = trial_avg_chunk_time
        prior_variance = trial_variance
        logger.info(f"Initialized prior with mean {prior_mean:.2f} ms and variance {prior_variance:.2f}")

    # Perform Bayesian update with the current trial data
    prior_mean, prior_variance = update_bayesian_mean_variance(prior_mean, prior_variance, trial_chunk_times)

    # Calculate Probability of Superiority against the best configuration
    prob_superiority = calculate_probability_of_superiority(
        best_avg_chunk_time, best_variance, trial_avg_chunk_time, trial_variance
    )

    if trial_avg_chunk_time < best_avg_chunk_time:
        # Trial is better than the current best
        if prob_superiority >= PROBABILITY_THRESHOLD:
            # Significant improvement found, update best values
            best_chunk_times, best_batch, best_ubatch = trial_chunk_times, n_batch, n_ubatch
            near_best_trials = []  # Clear near-best trials as we have a new best
            logger.info(f"New best found with probability of superiority: {prob_superiority:.3f}")
        else:
            # Trial is better but not with high confidence
            logger.warning(
                f"Trial with avg chunk time {trial_avg_chunk_time:.2f} ms is better than the best "
                f"({best_avg_chunk_time:.2f} ms) but probability of superiority is {prob_superiority:.3f} (below threshold)."
            )
            near_best_trials.append({
                "chunk_time": trial_avg_chunk_time,
                "params": {"n_batch": n_batch, "n_ubatch": n_ubatch},
                "prob_superiority": prob_superiority,
                "p_value": prob_superiority if prob_superiority < PROBABILITY_THRESHOLD else None
            })
    else:
        # Trial is worse than the current best
        logger.debug(
            f"Trial with avg chunk time {trial_avg_chunk_time:.2f} ms is worse than the best "
            f"({best_avg_chunk_time:.2f} ms). Probability of superiority: {prob_superiority:.3f}"
        )
        near_best_trials.append({
            "chunk_time": trial_avg_chunk_time,
            "params": {"n_batch": n_batch, "n_ubatch": n_ubatch},
            "prob_superiority": prob_superiority,
            "p_value": prob_superiority if prob_superiority < PROBABILITY_THRESHOLD else None
        })

    return best_chunk_times, best_batch, best_ubatch


def evaluate_trial(trial_time, best_time, trial_params, p_value, margin=0.05):
    """
    Evaluates whether a trial is within the margin of error. Resets near_best_trials
    when a new best trial is found to keep only relevant near-best trials.
    """
    global near_best_trials
    
    if trial_time < best_time:
        # New best trial found, reset near_best_trials
        near_best_trials = []
        if p_value < margin:
            logging.info("New best trial found.")

            return "best"
        else:
            logging.warning(
                "Trial with avg chunk time %.2f ms is better than the best (%.2f ms) but not beyond the margin of error (p=%.3f).",
                trial_time, best_time, p_value
            )
            near_best_trials.append({"params": trial_params, "chunk_time": trial_time, "p_value": p_value})

            return "near_best"
        
    return "worse"


def get_model_size_gb(model_path):
    """Get the model size in GB by checking the file size on disk."""
    model_size_bytes = os.path.getsize(model_path)
    model_size_gb = model_size_bytes / (1024 ** 3)  # Convert to GB

    return model_size_gb


def get_available_memory_gb():
    """Get available memory in GB based on the platform."""
    if torch.cuda.is_available():
        # If CUDA is available, get GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory

        # Use full available memory
        return total_memory / (1024 ** 3)
    else:
        # For CPU or non-CUDA environments, use system memory
        total_memory = psutil.virtual_memory().total

        return total_memory / (1024 ** 3)


def estimate_max_batch_size(model_size_gb, hidden_size, num_layers, precision_bits, sequence_length, available_memory_gb):
    """Estimate the maximum batch size based on GPU memory usage patterns."""
    # TODO why is sequence_length inessential to valid estimates??
    
    # Subtract model size from available memory
    available_memory_bytes = available_memory_gb * (1024 ** 3)
    model_size_bytes = model_size_gb * (1024 ** 3)
    remaining_memory = max(0, available_memory_bytes - model_size_bytes)

    # Approximate memory usage per token (scaled down further)
    bytes_per_token = hidden_size * num_layers * precision_bits / 8

    # Calculate the max batch size
    max_batch_size = remaining_memory // bytes_per_token
    
    logger.info(f"Available memory: {available_memory_gb:.2f} GB")
    logger.info(f"Model size: {model_size_gb:.2f} GB")
    logger.info(f"Max batch size calculated: {max_batch_size}")
    
    return max_batch_size


def estimate_number_of_trials(ubatch_exponent_range, batch_exponent_range):
    num_ubatch_values = ubatch_exponent_range.max - ubatch_exponent_range.min + 1
    num_batch_values = batch_exponent_range.max - batch_exponent_range.min + 1

    # Calculate approximate valid combinations, assuming about half the batch values divide each ubatch evenly
    # This rough estimate assumes that for each ubatch size, about half the batch sizes will be divisible
    estimated_valid_combinations = num_ubatch_values * (num_batch_values // 2)

    # Estimate the number of trials needed by TPESampler
    # Using a heuristic based on the logarithm of the total valid combinations
    c = 5  # Complexity factor
    estimated_trials = int(c * math.log2(max(1, estimated_valid_combinations)))

    # Cap trials by total valid combinations and set a minimum threshold
    estimated_trials = min(estimated_trials, estimated_valid_combinations)
    estimated_trials = max(min(10, estimated_valid_combinations), estimated_trials)

    return estimated_trials


def setup_study():
    """Sets up the Optuna TPE study with MedianPruner."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=1,        # Number of trials to wait before pruning
        interval_steps=1           # Interval between pruning checks
    )
    
    return optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner
    )


def generate_random_text(target_num_tokens, model):
    """Generates random text that tokenizes to approximately the target number of tokens."""
    generated_text = []
    total_tokens = 0

    # Define a simple vocabulary of random words
    vocabulary = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(1000)]

    while total_tokens < target_num_tokens:
        # Generate a random sentence
        sentence = ' '.join(random.choices(vocabulary, k=100))
        generated_text.append(sentence)

        # Concatenate the generated text
        text_so_far = ' '.join(generated_text)

        # Tokenize the current text (encode as UTF-8)
        encoded_text = text_so_far.encode("utf-8")
        tokens = model.tokenize(encoded_text)
        total_tokens = len(tokens)

    return text_so_far


def objective_wrapper(trial, pre_chunked_text, kwargs, best_chunk_time=None):
    n_batch = trial.params.get('n_batch', trial.user_attrs.get('n_batch'))
    n_ubatch = trial.params.get('n_ubatch', trial.user_attrs.get('n_ubatch'))

    trial_key = (n_batch, n_ubatch)

    # Check for a cached result or exception
    if trial_key in trial_cache:
        cached_result = trial_cache[trial_key]
        cached_result['read_count'] += 1  # Increment read count on access

        if isinstance(cached_result['result'], tuple) and cached_result['result'][0] == 'exception':
            logger.debug(f"Re-raising cached exception for n_batch={n_batch}, n_ubatch={n_ubatch}")
            raise cached_result['result'][1]
        elif cached_result['result'] is not None:
            logger.debug(f"Using cached result for n_batch={n_batch}, n_ubatch={n_ubatch}")

            return cached_result['result']

    # Proceed with trial execution as usual
    queue = mp.Queue()
    process = mp.Process(target=objective, args=(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time))
    process.start()
    chunk_times = []

    try:
        # Initialize start_time within the loop for each chunk processing phase
        start_time = time.time()

        # Run trial and handle results as usual
        while process.is_alive() or not queue.empty():
            if best_chunk_time and (time.time() - start_time) * 1000 > best_chunk_time:
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                process.join()
                raise optuna.TrialPruned("Chunk time exceeded best_chunk_time threshold.")

            if not queue.empty():
                result = queue.get_nowait()
                if isinstance(result, tuple):
                    chunk_num, chunk_time = result
                    if chunk_num == "done":
                        process.join()
                        break  # Exit if done
                    else:
                        chunk_times.append(chunk_time)
                        logger.debug(f"Got chunk {chunk_num} for trial {trial.number}, {chunk_time} ms")

                        trial.report(chunk_time, step=chunk_num)
                        if trial.should_prune():
                            process.terminate()
                            process.join(timeout=1)
                            if process.is_alive():
                                process.kill()
                            process.join()
                            raise optuna.TrialPruned()

                # Reset start_time after each completed chunk
                start_time = time.time()

        # Cache the result after successful completion
        if chunk_times:
            trial_cache[trial_key] = {'result': chunk_times, 'read_count': 0}  # Cache result with initial read_count

            return chunk_times

        raise optuna.TrialPruned("No result returned from trial process.")

    except optuna.TrialPruned as e:
        logger.debug(f"Trial {trial.number} was pruned")
        trial_cache[trial_key] = {'result': ('exception', e), 'read_count': 0}  # Cache the pruned exception

        raise  # Re-raise for consistent behavior

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) or 'OOM' in str(e):
            logger.warning(f"Trial {trial.number} pruned due to OOM error: {e}")
            trial_cache[trial_key] = {'result': ('exception', optuna.TrialPruned("OOM")), 'read_count': 0}  # Cache OOM as pruned
            raise optuna.TrialPruned("OOM")  # Raise pruned for consistent behavior
        else:
            logger.error(f"Trial {trial.number} failed with exception: {e}")
            trial_cache[trial_key] = {'result': ('exception', e), 'read_count': 0}  # Cache other runtime errors
            raise

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with unexpected exception: {e}")
        trial_cache[trial_key] = {'result': ('exception', e), 'read_count': 0}  # Cache unexpected exceptions
        raise


def objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time=None):
    """Objective function for optimization inside subprocess, reporting each chunk time via the queue."""
    logger.info(f"Testing with batch size (n_batch): {n_batch}, micro batch size (n_ubatch): {n_ubatch}")

    try:
        args = kwargs.copy()
        args['n_batch'] = n_batch
        args['n_ubatch'] = n_ubatch
        args = prepare_llama_args(args)
        logger.debug(f"Initializing model")
        with open(os.devnull, 'w') as f_null, contextlib.redirect_stderr(f_null), contextlib.redirect_stdout(f_null):
            model = llama_cpp.Llama(**args)
        logger.debug(f"Model initialized")

        chunk_times = []

        for chunk_num, chunk in enumerate(pre_chunked_text[:kwargs['chunks']]):
            start_time = time.time()
            with open(os.devnull, 'w') as f_null, contextlib.redirect_stderr(f_null), contextlib.redirect_stdout(f_null):
                _ = model(chunk)  # Run the model inference
            total_time = (time.time() - start_time) * 1000
            chunk_times.append(total_time)

            # Check against best_chunk_time for pruning
            if best_chunk_time and total_time > best_chunk_time:
                queue.put(RuntimeError("Chunk time exceeded best_chunk_time"))

                return
            
            # Report each chunk time back to the main process
            queue.put((chunk_num, total_time))  # Send the chunk number and its time to the queue
            
        # Send the final result (average chunk time) to the parent process
        queue.put(("done", sum(chunk_times) / len(chunk_times)))

    except Exception as e:
        if 'CUDA out of memory' in str(e) or 'OOM' in str(e):
            queue.put(RuntimeError("OOM"))  # Special case for handling memory issues
        else:
            queue.put(e)  # Send other exceptions to the parent process


def prepare_llama_args(kwargs):
    llama_args = {
        'model_path': kwargs.get('model'),
        'n_ctx': kwargs.get('context_size'),
        'n_gpu_layers': kwargs.get('n_gpu_layers'),
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
        'seed': kwargs.get('seed'),
        'n_threads': kwargs.get('threads')
    }

    # Conditionally add n_batch and n_ubatch if they exist in kwargs
    if 'n_batch' in kwargs:
        llama_args['n_batch'] = kwargs['n_batch']
    if 'n_ubatch' in kwargs:
        llama_args['n_ubatch'] = kwargs['n_ubatch']

    # Remove any None values from the dictionary
    llama_args = {k: v for k, v in llama_args.items() if v is not None}

    return llama_args


def tokenize(model, kwargs):
    """Initializes the model and tokenizes the text."""
    context_size = kwargs['context_size']
    target_num_tokens = kwargs['chunks'] * (context_size - 1)

    input_text = generate_random_text(target_num_tokens, model).encode("utf-8")
    tokenized_text = model.tokenize(input_text)

    return tokenized_text


def get_model_config(model_path):
    """
    Extract model configuration (hidden size, layers) from the model's config file or gguf metadata if available.
    """
    # Check if config.json exists
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    
    hidden_size = None
    num_layers = None
    model = None

    try:
        with open(os.devnull, 'w') as f_null, contextlib.redirect_stderr(f_null), contextlib.redirect_stdout(f_null):
            model = llama_cpp.Llama(model_path)
            metadata = model.metadata
            
            hidden_size = int(metadata['llama.embedding_length'])
            num_layers = int(metadata['llama.block_count'])
    except KeyError as e:
        logger.error(f"Key missing in gguf metadata: {e}")
        raise KeyError(f"Required key missing in gguf metadata: {e}")
    except Exception as e:
        logger.error(f"Failed to load metadata from gguf model: {e}")
        raise RuntimeError("Failed to retrieve model configuration from config.json or gguf metadata.")
    
    # Final check to ensure both values are set
    if hidden_size is None or num_layers is None:
        raise ValueError("Model configuration is incomplete: hidden_size or num_layers is missing.")

    return hidden_size, num_layers, model


def create_trial(study: optuna.Study, batch_exponent_range, ubatch_exponent_range, default_n_batch=None, default_n_ubatch=None):
    if default_n_batch and default_n_ubatch:
        # Set default batch sizes if provided
        n_batch = default_n_batch
        n_ubatch = default_n_ubatch
        params = {
            'n_batch': n_batch,
            'n_ubatch': n_ubatch
        }
        study.enqueue_trial(params=params, user_attrs=params)
        
        # Log for the default trial
        logger.info(f"Created Trial (default): n_batch={n_batch}, n_ubatch={n_ubatch}")
        return study.ask()  # Return a new trial after enqueuing the default

    else:
        trial = study.ask()

        # Suggest exponents within the range and convert them to actual batch sizes
        n_batch_exponent = trial.suggest_int('n_batch_exponent', batch_exponent_range.min, batch_exponent_range.max)
        n_ubatch_exponent = trial.suggest_int('n_ubatch_exponent', ubatch_exponent_range.min, ubatch_exponent_range.max)
        # note: this would be better if we didnt lose magnitude working with int, ie with a hypothetical suggest_ordinal

        n_batch = 2 ** n_batch_exponent
        n_ubatch = 2 ** n_ubatch_exponent

        # Ensure divisibility of batch by ubatch
        while n_batch % n_ubatch != 0:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial = study.ask()
            n_batch_exponent = trial.suggest_int('n_batch_exponent', batch_exponent_range.min, batch_exponent_range.max)
            n_ubatch_exponent = trial.suggest_int('n_ubatch_exponent', ubatch_exponent_range.min, ubatch_exponent_range.max)
            n_batch = 2 ** n_batch_exponent
            n_ubatch = 2 ** n_ubatch_exponent

        # Log the trial created with suggested parameters
        logger.info(f"Created Trial {trial.number}: n_batch={n_batch}, n_ubatch={n_ubatch}")
        trial.set_user_attr('n_batch', n_batch)
        trial.set_user_attr('n_ubatch', n_ubatch)
    
    return trial


def chunk_text(tokenized_text, context_size):
    """Chunks the tokenized input text."""
    return [tokenized_text[i:i + (context_size - 1)] for i in range(0, len(tokenized_text), context_size - 1)]


def execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range):
    logger.debug(
        f"Executing study over batch exponent range: {batch_exponent_range}\n"
        f"and ubatch exponent range: {ubatch_exponent_range}"
    )

    completed_trials = 0
    max_attempts = n_trials * 10  # Prevent infinite loops
    attempts = 0
    best_chunk_times = []  # Track best chunk times for significance testing
    best_batch, best_ubatch = None, None

    while completed_trials < n_trials and attempts < max_attempts:
        attempts += 1

        if completed_trials == 0 and \
           batch_exponent_range.min <= DEFAULT_BATCH_EXPONENT <= batch_exponent_range.max and \
           ubatch_exponent_range.min <= DEFAULT_UBATCH_EXPONENT <= ubatch_exponent_range.max:
            # Set default values based on exponents
            n_batch = 2 ** DEFAULT_BATCH_EXPONENT
            n_ubatch = 2 ** DEFAULT_UBATCH_EXPONENT
            trial = create_trial(
                study, 
                batch_exponent_range, 
                ubatch_exponent_range, 
                default_n_batch=n_batch, 
                default_n_ubatch=n_ubatch
            )
        else:
            trial = create_trial(study, batch_exponent_range, ubatch_exponent_range)
            n_batch = trial.user_attrs.get('n_batch')
            n_ubatch = trial.user_attrs.get('n_ubatch')

        logger.debug(f"Executor running Trial {trial.number}: n_batch={n_batch}, n_ubatch={n_ubatch}")

        try:
            # Pass best average chunk time to the objective_wrapper with a margin (if best times exist)
            avg_best_time = sum(best_chunk_times) / len(best_chunk_times) * 2.5 if best_chunk_times else None
            chunk_times = objective_wrapper(trial, pre_chunked_text, kwargs, avg_best_time)
            
            # Calculate the average time of this trial
            trial_avg_time = sum(chunk_times) / len(chunk_times) if chunk_times else float('inf')
            logger.info(f"Trial {trial.number} completed with average time: {trial_avg_time:.2f} ms")
            study.tell(trial, trial_avg_time)
            
            # Update best_chunk_times using statistical significance check
            best_chunk_times, best_batch, best_ubatch = update_best_chunk_time_with_probability(
                chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
            )

            completed_trials += 1
        except optuna.TrialPruned:
            logger.warning(f"Trial {trial.number} was pruned")
            study.tell(trial, np.inf)
            completed_trials += 1
        except Exception as e:
            if 'CUDA out of memory' in str(e) or 'OOM' in str(e):
                logger.warning(f"Trial {trial.number} pruned due to OOM error: {e}")
                study.tell(trial, np.inf)
            else:
                logger.warning(f"Trial {trial.number} failed with exception: {e}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            completed_trials += 1

    if attempts >= max_attempts:
        logger.warning(
            f"Reached maximum number of attempts ({max_attempts}) while trying to complete {n_trials} unique trials."
        )


def report_results(study):
    """Reports the results of the study with detailed status information."""
    logger.info("\nOptimization Results:")
    logger.info(f"Best average processing time per chunk: {study.best_value:.2f} ms")

    best_n_batch_exponent = study.best_params['n_batch_exponent']
    best_n_ubatch_exponent = study.best_params['n_ubatch_exponent']

    best_n_batch = 2 ** best_n_batch_exponent
    best_n_ubatch = 2 ** best_n_ubatch_exponent

    logger.info(f"Best parameters: n_batch={best_n_batch} (2^{best_n_batch_exponent}), n_ubatch={best_n_ubatch} (2^{best_n_ubatch_exponent})")

    # Track unique trials within margin of error by their parameters, excluding the best parameters
    unique_trials = {}
    if near_best_trials:
        logger.info("\n---- Trials within Margin of Error ----")
        for trial in near_best_trials:
            trial_n_batch = trial['params']['n_batch']
            trial_n_ubatch = trial['params']['n_ubatch']
            trial_key = (trial_n_batch, trial_n_ubatch)
            
            # Skip entries with the same parameters as the best
            if trial_key == (best_n_batch, best_n_ubatch):
                continue
            
            # Add only unique trials to the dictionary
            if trial_key not in unique_trials:
                unique_trials[trial_key] = trial
                logger.info(f"Chunk Time: {trial['chunk_time']} ms | Params: {trial['params']} | Within margin (p={trial['p_value']})")
    else:
        logger.info("No trials were within the margin of error.")

    # Detailed report for all trials
    for trial in study.trials:
        status = trial.user_attrs.get('status', 'unknown')
        params = trial.params or {}
        n_batch = 2 ** params.get('n_batch_exponent', DEFAULT_BATCH_EXPONENT)
        n_ubatch = 2 ** params.get('n_ubatch_exponent', DEFAULT_UBATCH_EXPONENT)

        if status == 'completed':
            chunks_completed = trial.user_attrs.get('chunks_completed', 'unknown')
            logger.info(f"Trial {trial.number}: Average Time={trial.value:.2f} ms, \nCompleted {chunks_completed} chunks, Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")
        
        elif status == 'pruned_optuna':
            chunks_completed = trial.user_attrs.get('chunks_completed', 'unknown')
            logger.debug(f"Trial {trial.number}: Pruned by Optuna after {chunks_completed} chunks, Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")
        
        elif status == 'pruned_time':
            message = trial.user_attrs.get('message', '')
            logger.debug(f"Trial {trial.number}: Pruned (time threshold) - {message}, Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")
        
        elif status == 'pruned_oom':
            logger.debug(f"Trial {trial.number}: Pruned (OOM error), Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")
        
        elif status == 'failed':
            error = trial.user_attrs.get('error', 'Unknown error')
            message = trial.user_attrs.get('message', '')
            error_info = message if message else error
            logger.debug(f"Trial {trial.number}: Failed - {error_info}, Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")
        
        else:
            logger.debug(f"Trial {trial.number}: Status unknown, Params={{'n_batch': {n_batch}, 'n_ubatch': {n_ubatch}}}")


def initialize_batch_and_model_config(kwargs):
    """Initialize model config and estimate batch sizes."""
    model_size_gb = get_model_size_gb(kwargs['model'])
    hidden_size, num_layers, model = get_model_config(kwargs['model'])
    precision_bits = estimate_model_precision(model=model)
    available_memory_gb = get_available_memory_gb()

    # Estimate the maximum batch size
    max_batch_size = estimate_max_batch_size(
        model_size_gb, 
        hidden_size, 
        num_layers, 
        precision_bits, 
        kwargs['context_size'], 
        available_memory_gb
    )

    # Define exponent range for batch sizes, starting from 2^4 (16) up to max_batch_size
    max_batch_size = min(max_batch_size, kwargs['context_size'])
    batch_exponent_range = ExponentRange(4, int(max_batch_size).bit_length() - 1)

    # Ubatch exponents should include 2, 4, 8 (1, 2, 3) as well as the range for batch sizes
    max_ubatch_size = min(batch_exponent_range.max, math.floor(math.log2(model.n_ctx))) \
        if kwargs['conform_to_imatrix'] else batch_exponent_range.max
    ubatch_exponent_range = ExponentRange(1, max_ubatch_size)

    return batch_exponent_range, ubatch_exponent_range


def main(**kwargs):
    study = setup_study()

    batch_exponent_range, ubatch_exponent_range = initialize_batch_and_model_config(kwargs)

    if logger.isEnabledFor(logging.DEBUG):
        batch_sizes = [2 ** exp for exp in range(batch_exponent_range.min, batch_exponent_range.max + 1)]
        ubatch_sizes = [2 ** exp for exp in range(ubatch_exponent_range.min, ubatch_exponent_range.max + 1)]
        
        logger.debug(f"Batch size range (2^{batch_exponent_range.min} to 2^{batch_exponent_range.max}): {batch_sizes}")
        logger.debug(f"Ubatch size range (2^{ubatch_exponent_range.min} to 2^{ubatch_exponent_range.max}): {ubatch_sizes}")

    if kwargs['max_trials'] is not None: 
        n_trials = kwargs['max_trials'] 
    else:
        n_trials = estimate_number_of_trials(batch_exponent_range, ubatch_exponent_range)
        logger.info(f"Estimated number of trials automatically: {n_trials}")
    
    if kwargs['chunks'] is None:
        max_batch_size = 2 ** batch_exponent_range.max
        kwargs['chunks'] = 5 if kwargs['conform_to_imatrix'] else \
            max(5, math.ceil(max_batch_size / kwargs['context_size']))
        logger.info(f"Auto-estimated chunks: {kwargs['chunks']} for batch size {max_batch_size} and context size {kwargs['context_size']}")

    # Initialize model and tokenize text
    args = prepare_llama_args(kwargs)
    with open(os.devnull, 'w') as f_null, contextlib.redirect_stderr(f_null), contextlib.redirect_stdout(f_null):
        model = llama_cpp.Llama(**args)
        try:
            tokenized_text = tokenize(model, kwargs)
        finally:
            model.close()

    pre_chunked_text = chunk_text(tokenized_text, kwargs['context_size'])

    execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

    report_results(study)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize batch sizes using Optuna.")

    # Model path and context size
    parser.add_argument('--model', type=str, required=True, help='Path to the GGUF model file.')
    parser.add_argument('--context-size', type=int, required=True, help="The model's context size.")

    # GPU layers
    parser.add_argument('--n-gpu-layers', type=int, default=50, help='Number of layers to store in VRAM.')

    # Model-specific flags
    parser.add_argument('--temp', type=float, default=0, help='Temperature (default: 0.0)')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling (default: 0, 0 = disabled)')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p sampling (default: 1.0, 1.0 = disabled)')
    parser.add_argument('--min-p', type=float, default=0.0, help='Min-p sampling (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (Default: 0).')
    parser.add_argument('--repeat-last-n', type=int, default=64, help='Last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)')
    parser.add_argument('--repeat-penalty', type=float, default=1.0, help='Penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)')
    parser.add_argument('--presence-penalty', type=float, default=0.0, help='Repeat alpha presence penalty (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--frequency-penalty', type=float, default=0.0, help='Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--dynatemp-range', type=float, default=0.0, help='Dynamic temperature range (default: 0.0, 0.0 = disabled)')
    parser.add_argument('--dynatemp-exp', type=float, default=1.0, help='Dynamic temperature exponent (default: 1.0)')
    parser.add_argument('--mirostat', type=int, default=0, help='Use Mirostat sampling. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)')
    parser.add_argument('--mirostat-lr', type=float, default=0.1, help='Mirostat learning rate, parameter eta (default: 0.1)')
    parser.add_argument('--mirostat-ent', type=float, default=5.0, help='Mirostat target entropy, parameter tau (default: 5.0)')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() - 1), help='Number of threads to use for parallel processing (default: system threads - 1)')
    parser.add_argument('--max-trials', type=int, default=None, help='Number of trials to run (default: selected automatically)')
    parser.add_argument('--chunks', type=int, default=None, help='Number of chunks to process per trial (default: selected automatically)')
    parser.add_argument('--conform-to-imatrix', action='store_true', help='If true, the maximum batch size will be limited to the context_size of the model')
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
    logging.info(f"best_bub starting (version {__version__})")

    main(**args_dict)
