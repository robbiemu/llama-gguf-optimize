from astropy.stats import kuiper
from dataclasses import dataclass, field
import h5py
import json
import logging
import math
import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import beta, expon, norm
from tdigest import TDigest
from typing import List, Optional, Tuple

from version import __version__
from gguf_optimize_logging import setup_logging

logger = logging.getLogger(__name__)

EARLY_STOPPING = "stop"
EARLY_STOPPING_DEFAULTS = {
    'confidence': 0.95,
    'margin_of_error': 0.1,
    'min_chunks': None,
    'theta_E': 0.2,
    'theta_P': 0.1,
    'window_size': None,
    'momentum': 0.3,
    'learning_rate': 0.5,
    'min_prior_weight': 0.2,
    'decay_rate': None
}

class ChunkStats(BaseModel):
    ChunkNumber: int
    Average: float
    StdDev: float
    Median: float
    Minimum: float
    Maximum: float
    KLD_99: float
    KLD_95: float
    KLD_90: float
    KLD_10: float
    KLD_05: float
    KLD_01: float

class KLFileStructure(BaseModel):
    chunks: list[ChunkStats] = Field(description="Statistics for each chunk")
    overall: Optional[ChunkStats] = Field(description="Overall statistics across all chunks")


@dataclass
class EarlyStoppingStats:
    prior_distribution: np.ndarray = None
    confidence: float = 0.95
    alpha: int = 1
    beta: int = 1
    window_size: int = 3
    theta_E: float = 0.2
    theta_P: float = 0.1
    dynamic_thresholds_enabled: bool = False
    historic_theta_E: float = None
    historic_theta_P: float = None
    total_theta_E_updates: int = 0  # Total updates to theta_E
    theta_E_increase_count: int = 0  # Count of increases to theta_E
    base_decrease_rate_theta_E: float = 0.1  # Base decrease rate
    max_decrease_rate_theta_E: float = 0.3  # Maximum decrease rate
    min_decrease_rate_theta_E: float = 0.01  # Minimum decrease rate
    effective_chunk_size: int = None
    sample_size: int = 0 # number of samples seen
    min_samples: int = 45056
    stopped_early: bool = False
    stopping_chunk: Optional[int] = None
    effect_sizes: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    
    # New attributes for EMA
    ema_relative_change: float = 0.0
    ema_p_value_std_dev: float = 0.0
    ema_decay: float = 0.005  # Smoothing factor for EMA

    def add_effect_size(self, effect_size: float):
        if len(self.effect_sizes) >= self.window_size:
            self.effect_sizes.pop(0)
        self.effect_sizes.append(effect_size)
    
    def add_p_value(self, p_value: float):
        if len(self.p_values) >= self.window_size:
            self.p_values.pop(0)
        self.p_values.append(p_value)
    
    def _update_ema(self, relative_change: float, p_value_std_dev: float):
        """
        Update the exponential moving averages for relative changes and p-value std dev.
        """
        if self.sample_size == 0:
            # Initialize EMA with the first data point
            self.ema_relative_change = relative_change
            self.ema_p_value_std_dev = p_value_std_dev
        else:
            # Update EMA using the smoothing factor
            self.ema_relative_change = (self.ema_decay * relative_change +
                                        (1 - self.ema_decay) * self.ema_relative_change)
            self.ema_p_value_std_dev = (self.ema_decay * p_value_std_dev +
                                        (1 - self.ema_decay) * self.ema_p_value_std_dev)
    
    def update_beta_parameters(self):
        if len(self.effect_sizes) < self.window_size or len(self.p_values) < self.window_size:
            return
    
        relative_changes = self._calculate_relative_effect_size_changes()
        p_value_std_dev = self._calculate_p_value_std_dev()
    
        if self.dynamic_thresholds_enabled:
            theta_E, theta_P = self._calculate_dynamic_thresholds()
            logger.debug(f"Updated theta_E {theta_E} and theta_P {theta_P}")
        else:
            logger.debug("Not updating non-dynamic theta")
            theta_E, theta_P = self.theta_E, self.theta_P
    
        # Update EMA with current relative change and p_value_std_dev
        current_relative_change = np.median(relative_changes)
        current_p_value_std_dev = p_value_std_dev
        self._update_ema(current_relative_change, current_p_value_std_dev)
    
        logger.debug(f"Updated EMA_relative_change: {self.ema_relative_change}, EMA_p_value_std_dev: {self.ema_p_value_std_dev}") # THESE ARE THETA_E and THETA_P
    
        # Use EMA values for threshold comparison
        if self.ema_relative_change < theta_E and self.ema_p_value_std_dev < theta_P:
            self.alpha += 1
            logger.debug("Condition met: Incremented alpha.")
        else:
            self.beta += 1
            msg = f"unmet for relative change {self.ema_relative_change} (θ_Ε: {theta_E})" if self.ema_relative_change >= theta_E else ""
            msg += f" unmet for p-value std dev {self.ema_p_value_std_dev} (θ_P: {theta_P})" if self.ema_p_value_std_dev >= theta_P else ""
            self.ema_p_value_std_dev < theta_P
            logger.debug(f"Condition not met: Incremented beta. Reasons: {msg}")
    
        logger.debug(f"Updated Beta parameters: alpha={self.alpha}, beta={self.beta}")

        return theta_E, theta_P, relative_changes, p_value_std_dev
    
    def _calculate_dynamic_thresholds(self):
        if len(self.effect_sizes) < self.window_size or len(self.p_values) < self.window_size:
            return self.theta_E, self.theta_P

        # Calculate relative changes in effect sizes
        relative_changes = self._calculate_relative_effect_size_changes()

        # Compute mean relative change and p-value std dev
        mean_relative_change = np.mean(relative_changes)
        p_value_std_dev = self._calculate_p_value_std_dev()

        # Update EMAs
        self._update_ema(mean_relative_change, p_value_std_dev)

        # Initialize dynamic_theta_E if not already set
        if self.historic_theta_E is None:
            self.historic_theta_E = self.ema_relative_change

        # Track updates
        self.total_theta_E_updates += 1

        # Update dynamic_theta_E
        if self.ema_relative_change > self.historic_theta_E:
            # Increase quickly
            self.historic_theta_E = self.ema_relative_change
            self.theta_E_increase_count += 1  # Count the increase
            logger.debug(f"dynamic_theta_E increased to {self.historic_theta_E}")
        else:

            # Calculate dynamic decrease rate
            increase_ratio = self.theta_E_increase_count / self.total_theta_E_updates
            dynamic_decrease_rate = (1 - increase_ratio) * self.max_decrease_rate_theta_E + \
                                    increase_ratio * self.min_decrease_rate_theta_E
            theta_E = max(self.historic_theta_E * (1 - dynamic_decrease_rate) +
                          self.ema_relative_change * dynamic_decrease_rate,
                          (1 - self.confidence)**(1/math.e))
            if theta_E < self.historic_theta_E:
                self.historic_theta_E = theta_E
                logger.debug(f"dynamic_theta_E decreased slowly to {self.historic_theta_E} with rate {dynamic_decrease_rate}")

        # Similar logic for dynamic_theta_P
        if self.historic_theta_P is None:
            self.historic_theta_P = self.ema_p_value_std_dev

        if self.ema_p_value_std_dev > self.historic_theta_P:
            # Increase quickly
            self.historic_theta_P = self.ema_p_value_std_dev
            logger.debug(f"dynamic_theta_P increased to {self.historic_theta_P}")
        else:
            decrease_rate = self.base_decrease_rate_theta_E  # For simplicity, keep static here
            theta_P = max(self.historic_theta_P * (1 - decrease_rate) +
                          self.ema_p_value_std_dev * decrease_rate, 1 - self.confidence)
            if theta_P < self.historic_theta_P:
                self.historic_theta_P = theta_P
                logger.debug(f"dynamic_theta_P decreased slowly to {self.historic_theta_P}")

        return self._clamp_theta_E(self.historic_theta_E), self._clamp_theta_P(self.historic_theta_P)
    
    def _clamp_theta_E(self, theta_E):
        # return max(theta_E, (1 - self.confidence)**(1/math.e))
        return max(theta_E, 0.2) # cohen's small 

    def _clamp_theta_P(self, theta_P):
        return max(theta_P, 1 - self.confidence)

    def _calculate_relative_effect_size_changes(self) -> List[float]:
        changes = []
        for i in range(1, len(self.effect_sizes)):
            change = abs((self.effect_sizes[i] - self.effect_sizes[i-1]) / self.effect_sizes[i-1])
            changes.append(change)

        return changes

    def _calculate_p_value_std_dev(self) -> float:
        if not self.p_values:
            return 0.0
        
        mean_p = np.mean(self.p_values)
        variance = sum((p - mean_p) ** 2 for p in self.p_values) / (len(self.p_values) - 1)

        return np.sqrt(variance)


@dataclass
class BayesianPriorState:
    initial_alpha: float
    decay_factor: float
    update_count: int
    current_alpha: float
    momentum: float
    previous_dists: list
    average_size: float
    target_alpha: float
    min_weight: float
    window_size: int


class BayesianPriorUpdate:
    """Handle Bayesian updating of the prior distribution with smoothed exponential decay."""
    def __init__(self, initial_prior: np.ndarray,
                 initial_alpha: float = 1.0,    # Starting learning rate
                 min_weight: float = 0.1,       # Minimum weight for old distribution
                 decay_factor: float = 0.3,     # Slower decay factor
                 target_alpha: float = 0.01,    # Lower target to allow more updates
                 momentum: float = 0.7,         # Momentum factor for smoothing
                 window_size: int = 3):         # Window for moving average
        self.current_prior = initial_prior
        self.initial_alpha = initial_alpha
        self.alpha = initial_alpha
        self.min_weight = min_weight
        self.decay_factor = decay_factor
        self.target_alpha = target_alpha
        self.update_count = 0
        self.momentum = momentum
        self.previous_dists = []
        self.window_size = window_size
        self.average_size = 0

        # Initialize moving averages for stability checking
        self.moving_avg_kl = []
        
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions, with smoothing."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.clip(p + epsilon, 0, 1)
        q = np.clip(q + epsilon, 0, 1)
        
        # Normalize to ensure valid probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))

    def check_convergence(self, new_dist: np.ndarray) -> Tuple[bool, float]:
        """Check if the distribution has converged using KL divergence."""
        if len(self.previous_dists) < self.window_size:
            return False, 1.0
        
        # Calculate KL divergence with moving average of previous distributions
        recent_average = np.mean(self.previous_dists[-self.window_size:], axis=0)
        kl_div = self.calculate_kl_divergence(new_dist, recent_average)
        
        self.moving_avg_kl.append(kl_div)
        if len(self.moving_avg_kl) > self.window_size:
            self.moving_avg_kl.pop(0)
        
        # Check if KL divergence is stable
        if len(self.moving_avg_kl) >= self.window_size:
            avg_kl = np.mean(self.moving_avg_kl)
            std_kl = np.std(self.moving_avg_kl)
            
            # Consider converged if variation is small relative to mean
            is_converged = std_kl < (0.1 * avg_kl) and avg_kl < 0.01
            return is_converged, kl_div
            
        return False, kl_div

    def update_learning_rate(self, kl_div: float):
        """Update learning rate using smooth exponential decay."""
        current_alpha = self.alpha
        
        # More aggressive decay when KL divergence is small
        adaptive_decay = self.decay_factor * (1 + kl_div)
        
        # Calculate new alpha with momentum
        distance_to_target = self.alpha - self.target_alpha
        decay_amount = distance_to_target * adaptive_decay
        new_alpha = max(self.target_alpha, self.alpha - decay_amount)
        
        # Apply momentum to smooth the learning rate changes
        self.alpha = (self.momentum * self.alpha + 
                     (1 - self.momentum) * new_alpha)
        
        logger.debug(f"Learning rate updated from {current_alpha} to {self.alpha} "
                    f"(KL div: {kl_div:.4f})")

    def update_prior(self, new_data: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Update prior using weighted combination of old prior and new data.
        Returns (updated distribution, convergence flag)
        """
        # Update average_size using weighted moving average
        if self.update_count == 0:
            self.average_size = len(new_data)  # Initialize for the first chunk
        else:
            self.average_size = (self.average_size * self.update_count + len(new_data)) / (self.update_count + 1)

        # Check convergence before update
        is_converged, kl_div = self.check_convergence(new_data)
        
        # Update learning rate based on KL divergence
        self.update_learning_rate(kl_div)

        # Calculate adaptive weight with smooth transitions
        normalized_size = len(new_data) / max(self.average_size, 1)  # Avoid division by zero
        weight = max(self.min_weight, 1.0 / (1.0 + self.alpha * normalized_size))
        
        # Combine distributions with momentum
        updated_dist = (weight * self.current_prior +
                       (1 - weight) * new_data)
        
        # Store for convergence checking
        self.previous_dists.append(updated_dist.copy())
        if len(self.previous_dists) > self.window_size:
            self.previous_dists.pop(0)
            
        self.current_prior = updated_dist
        self.update_count += 1

        return updated_dist, is_converged
    
    def get_state(self) -> BayesianPriorState:
        """Return the current state for saving."""
        return BayesianPriorState(
            initial_alpha=self.initial_alpha,
            decay_factor=self.decay_factor,
            update_count=self.update_count,
            current_alpha=self.alpha,
            momentum=self.momentum,
            average_size=self.average_size,
            target_alpha=self.target_alpha,
            min_weight=self.min_weight,
            window_size=self.window_size,
            previous_dists=self.previous_dists
        )

    def set_state(self, state: BayesianPriorState):
        """Set the state from a loaded state."""
        self.initial_alpha = state.initial_alpha
        self.decay_factor = state.decay_factor
        self.update_count = state.update_count
        self.alpha = state.current_alpha
        self.momentum = state.momentum
        self.average_size = state.average_size
        self.target_alpha = state.target_alpha
        self.min_weight = state.min_weight
        self.window_size = state.window_size
        self.previous_dists = state.previous_dists


def numpy_encoder(obj):
    # Handle NumPy scalar types
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    
    # Handle NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert array to a list for JSON serialization

    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def compose_overall_stats(
    overall_sum: float, overall_sumsq: float, overall_min: float,
    overall_max: float, total_values: int, digest: TDigest
) -> dict:
    """
    Compose the overall statistics dictionary from cumulative values.

    Args:
        overall_sum (float): Cumulative sum of all values.
        overall_sumsq (float): Cumulative sum of squares of all values.
        overall_min (float): Minimum value seen so far.
        overall_max (float): Maximum value seen so far.
        total_values (int): Total number of values processed so far.
        digest (TDigest): TDigest object for percentile calculations.

    Returns:
        dict: A dictionary containing the overall summary statistics.
    """
    if total_values == 0:
        raise ValueError("Cannot compose overall stats with total_values = 0.")

    # Compute the overall mean and standard deviation
    overall_mean = overall_sum / total_values
    overall_variance = overall_sumsq / total_values - overall_mean ** 2
    overall_stddev = max(0, overall_variance) ** 0.5

    # Compose the overall statistics dictionary
    overall_stats = {
        "Average": overall_mean,
        "StdDev": overall_stddev,
        "Minimum": overall_min,
        "Maximum": overall_max,
        "KLD_99": digest.percentile(99),
        "KLD_95": digest.percentile(95),
        "KLD_90": digest.percentile(90),
        "Median": digest.percentile(50),
        "KLD_10": digest.percentile(10),
        "KLD_05": digest.percentile(5),
        "KLD_01": digest.percentile(1),
        "total_values": total_values
    }

    return overall_stats


def check_output_file_conditions(output_path, from_chunk, to_chunk, clobber):
    """Check output file for existing data and determine if conditions allow processing."""
    existing_chunks, overall_stats, digest = set(), None, None

    if output_path and h5py.is_hdf5(output_path):
        with h5py.File(output_path, 'r') as f_out:
            existing_chunks = {int(chunk.split('_')[1]) for chunk in f_out.keys() if chunk.startswith("chunk_")}
            has_digest = 'digest' in f_out.attrs

            # If clobber is enabled, clear the file and exit the function early
            if clobber:
                logger.info(f"Clobber mode enabled; clearing all existing data in {output_path}.")
                f_out.close()
                with h5py.File(output_path, 'w') as f_out:
                    pass  # Clears the file by reopening in write mode
                return set(), None, None  # Return empty, as file is now cleared

            # Perform additional checks on existing data
            if existing_chunks and not {'overall_sum', 'overall_sumsq', 'overall_min', 'overall_max', 'total_values'}.issubset(f_out.attrs):
                raise ValueError(f"Output file contains partial chunk data without an 'overall' property. Use --clobber to start fresh or specify a valid range with --from and --to. (from {from_chunk}, to {to_chunk})")
            else: # Load and validate overall stats and digest if they exist
                if has_digest and existing_chunks:
                    digest_dict = json.loads(f_out.attrs['digest'])
                    digest = TDigest()
                    digest.update_from_dict(digest_dict)
                elif has_digest and not existing_chunks:
                    raise ValueError("Digest centroids present without processed chunks.")
                else:
                    logger.info("No digest centroids found in the file.")
                
                overall_sum = f_out.attrs['overall_sum']
                overall_sumsq = f_out.attrs['overall_sumsq']
                overall_min = f_out.attrs['overall_min']
                overall_max = f_out.attrs['overall_max']
                total_values = f_out.attrs['total_values']

                # Compose the overall summary using the helper function
                overall_stats = compose_overall_stats(overall_sum, overall_sumsq, overall_min, overall_max, total_values, digest)

            # Check for contiguous chunks if `--from` is set
            if from_chunk is not None:
                required_chunks = set(range(0, from_chunk))
                if not required_chunks.issubset(existing_chunks) and from_chunk != 0:
                    raise ValueError(f"Non-contiguous range detected. All chunks up to {from_chunk - 1} must be present to resume from {from_chunk}.")

    return existing_chunks, overall_stats, digest

def kl_divergence_log_probs(p_logits, q_logits):
    """
    Calculate KL divergence using log-softmax probabilities with numerical stability improvements.
    
    Args:
        p_logits: Logits from the baseline model.
        q_logits: Logits from the target model.
    Returns:
        numpy.ndarray: KL divergence values
    """
    # Apply temperature scaling to avoid overflow/underflow
    temperature = 1.0
    p_logits = p_logits / temperature
    q_logits = q_logits / temperature
    
    # Subtract max for numerical stability before computing softmax
    p_max = np.max(p_logits, axis=-1, keepdims=True)
    q_max = np.max(q_logits, axis=-1, keepdims=True)
    p_logits = p_logits - p_max
    q_logits = q_logits - q_max
    
    # Compute softmax probabilities directly
    p_probs = np.exp(p_logits) / np.sum(np.exp(p_logits), axis=-1, keepdims=True)
    q_probs = np.exp(q_logits) / np.sum(np.exp(q_logits), axis=-1, keepdims=True)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_probs = np.clip(p_probs, epsilon, 1.0)
    q_probs = np.clip(q_probs, epsilon, 1.0)
    
    # Compute KL divergence
    kl = p_probs * (np.log(p_probs) - np.log(q_probs))
    kl_sum = np.sum(kl, axis=-1)
    
    # Validate output
    if np.any(np.isnan(kl_sum)) or np.any(np.isinf(kl_sum)):
        logger.warning("Found NaN or Inf in KL divergence calculation")
        kl_sum = np.nan_to_num(kl_sum, nan=0.0, posinf=0.0, neginf=0.0)
    
    return kl_sum

def kl_divergence_log_probs(p_logits, q_logits):
    """
    Calculate KL divergence using log-softmax probabilities with numerical stability improvements.

    Args:
        p_logits: Logits from the baseline model.
        q_logits: Logits from the target model.
    Returns:
        numpy.ndarray: KL divergence values
    """
    # Apply temperature scaling to avoid overflow/underflow
    temperature = 1.0
    p_logits = p_logits / temperature
    q_logits = q_logits / temperature

    # Subtract max for numerical stability before computing softmax
    p_max = np.max(p_logits, axis=-1, keepdims=True)
    q_max = np.max(q_logits, axis=-1, keepdims=True)
    p_logits = p_logits - p_max
    q_logits = q_logits - q_max

    # Compute softmax probabilities directly
    p_probs = np.exp(p_logits) / np.sum(np.exp(p_logits), axis=-1, keepdims=True)
    q_probs = np.exp(q_logits) / np.sum(np.exp(q_logits), axis=-1, keepdims=True)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_probs = np.clip(p_probs, epsilon, 1.0)
    q_probs = np.clip(q_probs, epsilon, 1.0)

    # Compute KL divergence
    kl = p_probs * (np.log(p_probs) - np.log(q_probs))
    kl_sum = np.sum(kl, axis=-1)

    # Validate output
    if np.any(np.isnan(kl_sum)) or np.any(np.isinf(kl_sum)):
        logger.warning("Found NaN or Inf in KL divergence calculation")
        kl_sum = np.nan_to_num(kl_sum, nan=0.0, posinf=0.0, neginf=0.0)

    return kl_sum


def process_chunk_part(p_logits_part, q_logits_part, chunk_idx, part_idx):
    """
    Process a single part of a chunk with improved validation and error handling.
    """
    logger.debug(f"Processing chunk {chunk_idx}, part {part_idx}")

    # Validate shapes match
    if p_logits_part.shape != q_logits_part.shape:
        raise ValueError(f"Shape mismatch in chunk {chunk_idx}, part {part_idx}: "
                         f"p={p_logits_part.shape}, q={q_logits_part.shape}")

    # Check for finite values and handle them
    p_finite_mask = np.isfinite(p_logits_part)
    q_finite_mask = np.isfinite(q_logits_part)
    if not np.all(p_finite_mask):
        logger.warning(f"Replacing {np.sum(~p_finite_mask)} non-finite values in p_logits")
        p_logits_part = np.nan_to_num(p_logits_part, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(q_finite_mask):
        logger.warning(f"Replacing {np.sum(~q_finite_mask)} non-finite values in q_logits")
        q_logits_part = np.nan_to_num(q_logits_part, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate KL divergence
    kl_values = kl_divergence_log_probs(p_logits_part, q_logits_part)

    # Additional validation
    if np.all(kl_values == 0):
        # Check if inputs are identical
        if np.allclose(p_logits_part, q_logits_part):
            logger.warning(f"Input logits are identical in chunk {chunk_idx}, part {part_idx}")
        else:
            logger.warning(f"Zero KL divergence despite different inputs in chunk {chunk_idx}, part {part_idx}")

        # Print sample of logits for debugging
        sample_idx = 0
        logger.debug(f"Sample p_logits: {p_logits_part[sample_idx, :10]}")
        logger.debug(f"Sample q_logits: {q_logits_part[sample_idx, :10]}")

    return kl_values


def initialize_early_stopping(
        early_stopping: bool, min_samples: int, window_size: int,
        theta_E: float = 0.2, theta_P: float = 0.1, confidence: float = 0.95,
        dynamic_thresholds_enabled: bool = False
) -> Optional[EarlyStoppingStats]:
    """Initialize early stopping statistics if enabled."""
    if not early_stopping:
        return None
    
    return EarlyStoppingStats(
        prior_distribution=None,  # Will be initialized with first chunk
        confidence=confidence,
        alpha=1,                # Initial value for Beta distribution
        beta=1,                 # Initial value for Beta distribution
        window_size=window_size,
        theta_E=theta_E,
        theta_P=theta_P,
        dynamic_thresholds_enabled=dynamic_thresholds_enabled,
        sample_size=0,
        min_samples=min_samples,
        stopped_early=False,
        stopping_chunk=None,
        effect_sizes=[],
        p_values=[]
    )


def initialize_overall_stats(overall_stats: Optional[dict]) -> Tuple[float, float, float, float, int]:
    if overall_stats:
        overall_sum = overall_stats["Average"] * overall_stats["total_values"]
        overall_sumsq = (overall_stats["StdDev"] ** 2 + overall_stats["Average"] ** 2) * overall_stats["total_values"]
        overall_min = overall_stats["Minimum"]
        overall_max = overall_stats["Maximum"]
        total_values = overall_stats["total_values"]
    else:
        overall_sum, overall_sumsq = 0.0, 0.0
        overall_min, overall_max = float('inf'), float('-inf')
        total_values = 0

    return overall_sum, overall_sumsq, overall_min, overall_max, total_values


def initialize_prior(
        output_path: str, prior_learning_rate: float, min_prior_weight: float,
        decay_rate: float, momentum: float, initial_data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, BayesianPriorUpdate]:
    """
    Initialize prior distribution with initial_data and save to the output file.
    """
    prior_distribution = initial_data
    bayesian_updater = BayesianPriorUpdate(
        initial_prior=prior_distribution,
        initial_alpha=prior_learning_rate,  # Starting learning rate
        min_weight=min_prior_weight,
        decay_factor=decay_rate,
        momentum=momentum,
        window_size=window_size  
    )

    with h5py.File(output_path, 'a') as f_out:
        f_out.attrs['prior_distribution'] = json.dumps(prior_distribution.tolist(), default=numpy_encoder)
    logger.info("Initialized prior distribution with initial data.")

    return prior_distribution, bayesian_updater


def determine_chunk_range(from_chunk: Optional[int], to_chunk: Optional[int], total_chunks: int
                        ) -> Tuple[int, int]:
    start_chunk = from_chunk if from_chunk is not None else 0
    end_chunk = to_chunk if to_chunk is not None else total_chunks - 1

    return start_chunk, end_chunk


def process_single_chunk(
    chunk_idx: int, f_baseline, f_target, logits_baseline, chunk_index_baseline,
    chunk_index_target, parts: int, dtype: np.dtype
) -> Tuple[List[np.ndarray], dict]:
    chunk_stats = {
        "ChunkNumber": chunk_idx,
        "Average": 0.0,
        "StdDev": 0.0,
        "Median": 0.0,
        "Minimum": float('inf'),
        "Maximum": float('-inf'),
        "KLD_99": 0.0,
        "KLD_95": 0.0,
        "KLD_90": 0.0,
        "KLD_10": 0.0,
        "KLD_05": 0.0,
        "KLD_01": 0.0,
    }
    kl_values_list = []
    chunk_sum = 0.0
    chunk_sumsq = 0.0
    chunk_min = float('inf')
    chunk_max = float('-inf')
    chunk_total_values = 0
    chunk_digest = TDigest()

    physical_index_baseline = chunk_index_baseline[chunk_idx]
    physical_index_target = chunk_index_target[chunk_idx]

    if physical_index_baseline < 0 or physical_index_target < 0:
        logger.warning(f"Logical chunk {chunk_idx} is not present in one of the files; skipping.")
        return kl_values_list, chunk_stats

    physical_index = physical_index_baseline
    logits_baseline_chunk = logits_baseline[physical_index_baseline]
    num_samples = logits_baseline_chunk.shape[0]

    if num_samples == 0:
        logger.warning(f"Chunk {chunk_idx} has zero samples, skipping.")
        return kl_values_list, chunk_stats

    for part_idx in range(parts):
        start_idx = (num_samples * part_idx) // parts
        end_idx = (num_samples * (part_idx + 1)) // parts

        if start_idx >= end_idx:
            continue

        try:
            p_logits_part = logits_baseline_chunk[start_idx:end_idx].astype(dtype)
            q_logits_part = f_target['logits'][physical_index_target][start_idx:end_idx].astype(dtype)
            kl_values_part = process_chunk_part(p_logits_part, q_logits_part, physical_index, part_idx)

            if np.any(np.isnan(kl_values_part)) or np.any(np.isinf(kl_values_part)):
                logger.warning(f"Found NaN or Inf in KL divergence for chunk {chunk_idx}, part {part_idx}")
                continue

            kl_values_list.append(kl_values_part)

            # Update per-chunk accumulators
            chunk_sum += np.sum(kl_values_part)
            chunk_sumsq += np.sum(kl_values_part ** 2)
            chunk_min = min(chunk_min, kl_values_part.min().item())
            chunk_max = max(chunk_max, kl_values_part.max().item())
            chunk_total_values += len(kl_values_part)

            # Update per-chunk digest
            chunk_digest.batch_update(kl_values_part)

            del p_logits_part, q_logits_part, kl_values_part
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}, part {part_idx}: {e}")
            raise e

    if chunk_total_values == 0:
        logger.warning(f"No valid data in chunk {chunk_idx} after processing all parts.")
        return kl_values_list, chunk_stats

    chunk_mean = chunk_sum / chunk_total_values
    chunk_variance = chunk_sumsq / chunk_total_values - chunk_mean ** 2
    chunk_stddev = np.sqrt(max(0, chunk_variance))

    chunk_stats.update({
        "Average": chunk_mean,
        "StdDev": chunk_stddev,
        "Median": chunk_digest.percentile(50),
        "Minimum": chunk_min,
        "Maximum": chunk_max,
        "KLD_99": chunk_digest.percentile(99),
        "KLD_95": chunk_digest.percentile(95),
        "KLD_90": chunk_digest.percentile(90),
        "KLD_10": chunk_digest.percentile(10),
        "KLD_05": chunk_digest.percentile(5),
        "KLD_01": chunk_digest.percentile(1),
    })

    return kl_values_list, chunk_stats


def update_statistics(
    f_out: h5py.File, kl_values_chunk: np.ndarray, chunk_stats: dict, digest: TDigest,
    overall_sum: float, overall_sumsq: float, overall_min: float,
    overall_max: float, total_values: int
):
    overall_sum += chunk_stats["Average"] * kl_values_chunk.size
    overall_sumsq += (chunk_stats["StdDev"] ** 2 + chunk_stats["Average"] ** 2) * kl_values_chunk.size
    overall_min = min(overall_min, chunk_stats["Minimum"])
    overall_max = max(overall_max, chunk_stats["Maximum"])
    total_values += kl_values_chunk.size

    # Save cumulative statistics for overall
    f_out.attrs['overall_sum'] = overall_sum
    f_out.attrs['overall_sumsq'] = overall_sumsq
    f_out.attrs['overall_min'] = overall_min
    f_out.attrs['overall_max'] = overall_max
    f_out.attrs['total_values'] = total_values

    digest.batch_update(kl_values_chunk)

    return overall_sum, overall_sumsq, overall_min, overall_max, total_values


def save_chunk_stats(f_out: h5py.File, chunk_idx: int, chunk_stats: dict):
    group = f_out.create_group(f'chunk_{chunk_idx}')
    group.attrs.update(chunk_stats)

    logger.info(f"\n===== KL-divergence statistics for Chunk {chunk_idx} =====")
    for key, value in chunk_stats.items():
        if key == "ChunkNumber":
            continue
        logger.info(f"{key:8}: {value:.6f}")


def find_largest_divisor(x: int, y: int) -> int:
    """
    Find the largest divisor of x that is <= y.
    """
    if y > x:
        y = x

    # Start from sample_size and decrement to find the largest divisor
    for i in range(y, 0, -1):
        if x % i == 0:
            logging.debug(f"Found largest divisor: {i} for chunk_size: {x} and sample_size: {y}")
            return i

    return 1  # Fallback to 1 if no divisor is found


def find_nearest_divisor(x: int, y: int) -> int:
    """
    Find the divisor of x that is closest to y.
    """
    # Generate all divisors of x
    divisors = [i for i in range(1, x + 1) if x % i == 0]
    
    # Find the divisor closest to y
    nearest_divisor = min(divisors, key=lambda d: abs(d - y))
    logging.debug(f"Found nearest divisor: {nearest_divisor} for chunk_size: {x} and target: {y}")
    
    return nearest_divisor


def calculate_sample_size(variance, confidence_level=0.95, margin_of_error=0.01, ceiling=np.inf, ideal_kuiper_sample_size=250):
    logger.debug(f"Calculating sample size with variance={variance}, confidence_level={confidence_level}, margin_of_error={margin_of_error}, ceiling={ceiling}")
    
    # Determine z-score based on confidence level
    z = norm.ppf((1 + confidence_level) / 2)
    logger.debug(f"Z-score for confidence level {confidence_level}: {z}")
    
    # Calculate the required sample size before adjustments
    required_size = ideal_kuiper_sample_size * (z * np.sqrt(variance) / margin_of_error) ** 2
    logger.debug(f"Required size (before adjustments): {required_size}")
    
    # Ensure sample size is at least 1
    calculated_size = max(int(required_size), 1)
    logger.debug(f"Calculated size (after applying max constraint): {calculated_size}")
    
    # Adjust for ceiling if finite
    if math.isfinite(ceiling):
        logger.debug(f"Adjusting calculated size {calculated_size} with ceiling {ceiling}")
        calculated_size = find_largest_divisor(ceiling, calculated_size)
        logger.debug(f"Calculated size (after ceiling adjustment): {calculated_size}")
    
    return calculated_size


def adjust_decay_rate(current_decay_rate, p_value_difference, max_decay=.5, min_decay=0.01):
    # Increase decay rate when p-value is far from confidence level
    new_decay_rate = current_decay_rate * (1 + p_value_difference)
    # Ensure decay rate stays within bounds
    new_decay_rate = max(min_decay, min(new_decay_rate, max_decay))
    return new_decay_rate


def handle_early_stopping(
    kl_values_chunk: np.ndarray,
    prior_distribution: Optional[np.ndarray],
    bayesian_updater: Optional[BayesianPriorUpdate],
    early_stopping_stats: EarlyStoppingStats,
    confidence_level, margin_of_error,
    initial_prior_learning_rate: float,
    initial_min_prior_weight: float,
    decay_rate: float,
    momentum: float,
    chunk_idx, f_out: h5py.File,
    window_size: int,
    effective_chunk_size: int,
    log_effect_sizes=False,
) -> bool:
    if early_stopping_stats is None:
        return False  # Early stopping not enabled

    logger.debug(f"Samples seen: {early_stopping_stats.sample_size}")

    num_values = len(kl_values_chunk)
    if effective_chunk_size is None:
        effective_chunk_size = early_stopping_stats.effective_chunk_size 
        if effective_chunk_size is None or effective_chunk_size == 0: 
            effective_chunk_size = calculate_sample_size(
                np.var(kl_values_chunk),
                confidence_level=confidence_level, 
                margin_of_error=margin_of_error, 
                ceiling=num_values
            )
            early_stopping_stats.effective_chunk_size = effective_chunk_size

    num_segments = (num_values + effective_chunk_size - 1) // effective_chunk_size  # Ceiling division

    # Initialize prior if necessary
    if prior_distribution is None:
        # Use the first segment to initialize the prior
        initial_segment = kl_values_chunk[:effective_chunk_size]
        prior_distribution = initial_segment.copy()
        bayesian_updater = BayesianPriorUpdate(
            initial_prior=prior_distribution,
            initial_alpha=initial_prior_learning_rate,
            min_weight=initial_min_prior_weight,
            decay_factor=decay_rate,
            momentum=momentum,
            window_size=window_size
        )
        early_stopping_stats.prior_distribution = prior_distribution
        early_stopping_stats.window_size = window_size
        early_stopping_stats.alpha = 1
        early_stopping_stats.beta = 1
        early_stopping_stats.effect_sizes = []
        early_stopping_stats.p_values = []
        logger.info("Initialized prior distribution with the first segment.")
        start_segment = 1  # Since we used the first segment
    else:
        start_segment = 0

    logger.debug(f"segmentation {start_segment}/{num_segments} ({num_values} + {effective_chunk_size} - 1/ {effective_chunk_size})")

    for segment_idx in range(start_segment, num_segments):
        start_idx = segment_idx * effective_chunk_size
        end_idx = min((segment_idx + 1) * effective_chunk_size, num_values)
        kl_segment = kl_values_chunk[start_idx:end_idx]

        if len(kl_segment) == 0:
            logger.warning("kl_segement is empty")
            continue

        # Update prior
        prior_distribution, _ = bayesian_updater.update_prior(kl_segment)

        # Perform simple Kuiper test
        loc, scale = expon.fit(prior_distribution, floc=0)
        statistic, p_value = kuiper(kl_segment, cdf=expon.cdf, args=(loc, scale))

        logger.info(f"Kuiper statistic={statistic:.6f}, p-value={p_value:.6f}")

        # Update early stopping stats
        early_stopping_stats.add_effect_size(statistic)
        early_stopping_stats.add_p_value(p_value)

        # Adjust decay rate based on p-value difference
        p_value_difference = abs(p_value - confidence_level)
        bayesian_updater.decay_factor = adjust_decay_rate(
            bayesian_updater.decay_factor, p_value_difference
        )
        logger.debug(f"Adjusted decay rate: {bayesian_updater.decay_factor}")

        # Update beta parameters
        if len(early_stopping_stats.effect_sizes) >= early_stopping_stats.window_size:
            early_stopping_stats.update_beta_parameters()
        else:
            logger.debug("Window not full yet; using traditional p-value comparison.")
            # No need to adjust beta parameters yet

        # Calculate stopping probability
        prob_stop = beta.sf(confidence_level, early_stopping_stats.alpha, early_stopping_stats.beta)
        logger.info(f"Chunk {chunk_idx}: Beta parameters updated (alpha={early_stopping_stats.alpha}, beta={early_stopping_stats.beta}), "
                    f"stopping probability={prob_stop:.6f}")

        if log_effect_sizes:
            logger.info(f"Segment {segment_idx}: Effect sizes: {early_stopping_stats.effect_sizes}")

        # Check for early stopping
        early_stopping_stats.sample_size += len(kl_segment)
        if (early_stopping_stats.sample_size >= early_stopping_stats.min_samples and
                prob_stop >= confidence_level):
            early_stopping_stats.stopped_early = True
            early_stopping_stats.stopping_chunk = chunk_idx
            save_early_stopping_info(f_out, prior_distribution, early_stopping_stats, bayesian_updater)
            logger.info(f"Early stopping at chunk {early_stopping_stats.stopping_chunk}, segment {segment_idx}")
            return True

    # Save updated prior and stats after each segment
    save_prior_and_stats(f_out, prior_distribution, early_stopping_stats, bayesian_updater)

    return False


def save_common_state(f_out: h5py.File, prior_distribution: np.ndarray, 
                     early_stopping_stats: EarlyStoppingStats, bayesian_updater: BayesianPriorUpdate):
    f_out.attrs['prior_distribution'] = json.dumps(prior_distribution.tolist(), default=numpy_encoder)
    
    # Save early stopping stats
    if early_stopping_stats is not None:
        f_out.attrs['early_stopping_stats'] = json.dumps(early_stopping_stats.__dict__, default=numpy_encoder)
    
    # Save BayesianPriorUpdate state
    prior_state = bayesian_updater.get_state()
    f_out.attrs['bayesian_prior_state'] = json.dumps(prior_state.__dict__, default=numpy_encoder)


def save_prior_and_stats(f_out: h5py.File, prior_distribution: np.ndarray, 
                         early_stopping_stats: EarlyStoppingStats, bayesian_updater: BayesianPriorUpdate):
    save_common_state(f_out, prior_distribution, early_stopping_stats, bayesian_updater)

    logger.info("Saved prior distribution, early stopping stats, and Bayesian prior state to the output file.")


def save_early_stopping_info(f_out: h5py.File, prior_distribution: np.ndarray, 
                             early_stopping_stats: EarlyStoppingStats, 
                             bayesian_updater: BayesianPriorUpdate):
    early_stopping_stats.stopped_early = True
    save_common_state(f_out, prior_distribution, early_stopping_stats, bayesian_updater)
    f_out.attrs['early_stopping_stats'] = json.dumps(early_stopping_stats.__dict__, default=numpy_encoder)
    
    logger.info("Saved early stopping stats and Bayesian prior state to the output file.")


def finalize_processing(
    f_out: h5py.File, digest: TDigest, compute_overall: bool,
    overall_sum: float, overall_sumsq: float, overall_min: float,
    overall_max: float, total_values: int
):
    digest_dict = digest.to_dict()
    f_out.attrs['digest'] = json.dumps(digest_dict, default=numpy_encoder)

    logger.info(f"TOTAL CHUNKS processed.")

    if compute_overall:
        overall_mean = overall_sum / total_values
        overall_variance = overall_sumsq / total_values - overall_mean ** 2
        overall_stddev = np.sqrt(max(0, overall_variance))
        overall_stats = {
            "Average": overall_mean,
            "StdDev": overall_stddev,
            "Minimum": overall_min,
            "Maximum": overall_max,
            "KLD_99": digest.percentile(99),
            "KLD_95": digest.percentile(95),
            "KLD_90": digest.percentile(90),
            "Median": digest.percentile(50),
            "KLD_10": digest.percentile(10),
            "KLD_05": digest.percentile(5),
            "KLD_01": digest.percentile(1),
            "total_values": total_values
        }
        logger.info("\n===== Overall KL-divergence statistics =====")
        for key, value in overall_stats.items():
            if key != "total_values":
                logger.info(f"{key:8}: {value:.6f}")

        f_out.attrs['overall'] = json.dumps(overall_stats, default=numpy_encoder)


def load_state_from_file(f_out: h5py.File, prior_learning_rate: float, 
                         min_prior_weight: float, window_size: int) -> Tuple[Optional[np.ndarray], Optional[BayesianPriorUpdate], Optional[EarlyStoppingStats]]:
    """Load early stopping and Bayesian prior state from HDF5 file."""
    prior_distribution = None
    bayesian_updater = None
    early_stopping_stats = None

    if 'prior_distribution' in f_out.attrs:
        prior_distribution = np.array(json.loads(f_out.attrs['prior_distribution']))
        
        # Load BayesianPriorUpdate state
        if 'bayesian_prior_state' in f_out.attrs:
            state_dict = json.loads(f_out.attrs['bayesian_prior_state'])
            prior_state = BayesianPriorState(**state_dict)
            
            bayesian_updater = BayesianPriorUpdate(
                initial_prior=prior_distribution,
            )
            bayesian_updater.set_state(prior_state)
        else:
            bayesian_updater = BayesianPriorUpdate(
                initial_prior=prior_distribution,
                initial_alpha=prior_learning_rate,
                min_weight=min_prior_weight,
                window_size=window_size  
            )
        
        # Load EarlyStoppingStats if available
        if 'early_stopping_stats' in f_out.attrs:
            stats_dict = json.loads(f_out.attrs['early_stopping_stats'])
            # Ensure all necessary fields are present
            if 'alpha' not in stats_dict:
                stats_dict['alpha'] = 1
            if 'beta' not in stats_dict:
                stats_dict['beta'] = 1
            if 'window_size' not in stats_dict:
                stats_dict['window_size'] = window_size
            if 'theta_E' not in stats_dict:
                stats_dict['theta_E'] = 0.2
            if 'theta_P' not in stats_dict:
                stats_dict['theta_P'] = 0.1
            if 'effect_sizes' not in stats_dict:
                stats_dict['effect_sizes'] = []
            if 'p_values' not in stats_dict:
                stats_dict['p_values'] = []

            early_stopping_stats = EarlyStoppingStats(**stats_dict)
        
        logger.info("Loaded prior distribution, early stopping stats, and Bayesian prior state from file.")
    
    return prior_distribution, bayesian_updater, early_stopping_stats


def save_state_to_file(f_out: h5py.File, prior_distribution: np.ndarray, 
                      early_stopping_stats: Optional[EarlyStoppingStats], 
                      bayesian_updater: BayesianPriorUpdate, final: bool = False):
    """Save early stopping and Bayesian prior state to HDF5 file."""
    if prior_distribution is not None:
        f_out.attrs['prior_distribution'] = json.dumps(prior_distribution.tolist(), default=numpy_encoder)
        
        if early_stopping_stats is not None:
            if final:
                early_stopping_stats.stopped_early = True
            
            f_out.attrs['early_stopping_stats'] = json.dumps(early_stopping_stats.__dict__, 
                                                           default=numpy_encoder)
        
        if bayesian_updater is not None:
            prior_state = bayesian_updater.get_state()
            # Ensure all state variables are included
            f_out.attrs['bayesian_prior_state'] = json.dumps(prior_state.__dict__, default=numpy_encoder)
        
        logger.info("Saved prior distribution, early stopping stats, and Bayesian prior state to file.")


def process_chunks(
    baseline_path: str,
    target_path: str,
    output_path: Optional[str] = None,
    from_chunk: Optional[int] = None,
    to_chunk: Optional[int] = None,
    clobber: bool = False,
    precision: int = 64,
    parts: int = 1,
    early_stopping: bool = False,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.1,
    compute_overall: bool = False,
    min_samples: int = 45056,
    prior_learning_rate: float = 0.1,
    min_prior_weight: float = 0.2,
    decay_rate: Optional[float] = None,
    effective_chunk_size: Optional[int] = None,
    momentum: Optional[float] = None,
    window_size: int = 3,  
    theta_E: float = 0.2,  
    theta_P: float = 0.1,  
    dynamic_thresholds_enabled: bool = False,
    log_effect_sizes: bool = False,
):
    # Load existing file conditions and initialize overall stats
    existing_chunks, overall_stats, digest = check_output_file_conditions(
        output_path, from_chunk, to_chunk, clobber
    )
    overall_sum, overall_sumsq, overall_min, overall_max, total_values = initialize_overall_stats(overall_stats)
    digest = digest or TDigest()
    
    with h5py.File(baseline_path, 'r') as f_baseline, \
         h5py.File(target_path, 'r') as f_target, \
         h5py.File(output_path, 'a') as f_out:
        
        # Load or initialize early stopping and prior state
        prior_distribution, bayesian_updater, early_stopping_stats = load_state_from_file(
            f_out, prior_learning_rate, min_prior_weight, window_size
        )
        
        # Initialize early stopping if necessary
        if early_stopping and early_stopping_stats is None:
            early_stopping_stats = initialize_early_stopping(
                early_stopping=early_stopping,
                confidence=confidence_level,
                min_samples=min_samples,
                window_size=window_size,
                theta_E=theta_E,
                theta_P=theta_P,
                dynamic_thresholds_enabled=dynamic_thresholds_enabled
            )
        
        logits_baseline = f_baseline['logits']
        total_chunks = logits_baseline.shape[0]
        chunk_index_baseline = f_baseline['chunk_index'][:]
        chunk_index_target = f_target['chunk_index'][:]
        
        start_chunk, end_chunk = determine_chunk_range(from_chunk, to_chunk, total_chunks)
        logger.info(f"Processing chunks {start_chunk} to {end_chunk}...")
        
        dtype = np.float32 if precision == 32 else np.float64

        early_stopping_requested = False        
        for chunk_idx in range(start_chunk, end_chunk + 1):
            if chunk_idx in existing_chunks:
                logger.info(f"Skipping already processed chunk {chunk_idx}")
                continue
            
            kl_values_list, chunk_stats = process_single_chunk(
                chunk_idx, f_baseline, f_target, logits_baseline, chunk_index_baseline,
                chunk_index_target, parts, dtype
            )
            
            if not kl_values_list:
                logger.warning("Error, empty kl_values_list")
                continue
            logger.debug(f"kl_values_list size {[part.shape for part in kl_values_list]}")
            
            kl_values_chunk = np.concatenate(kl_values_list)
            
            overall_sum, overall_sumsq, overall_min, overall_max, total_values = update_statistics(
                f_out, kl_values_chunk, chunk_stats, digest, overall_sum, 
                overall_sumsq, overall_min, overall_max, total_values
            )
            
            save_chunk_stats(f_out, chunk_idx, chunk_stats)
            
            # Early stopping check
            if early_stopping:
                if decay_rate is None:
                    decay_rate = 1 / math.log(total_chunks)
                
                early_stopping_requested = handle_early_stopping(
                    kl_values_chunk=kl_values_chunk,
                    prior_distribution=prior_distribution,
                    bayesian_updater=bayesian_updater,
                    early_stopping_stats=early_stopping_stats,
                    confidence_level=confidence_level,
                    margin_of_error=margin_of_error,
                    initial_prior_learning_rate=prior_learning_rate,
                    initial_min_prior_weight=min_prior_weight,
                    decay_rate=decay_rate,
                    momentum=momentum,
                    chunk_idx=chunk_idx,
                    f_out=f_out,
                    window_size=window_size, 
                    effective_chunk_size=effective_chunk_size,  
                    log_effect_sizes=log_effect_sizes,
                )
                # Log the confidence level
                logger.info(f"confidence_level: {confidence_level}")

                
                if early_stopping_requested:
                    compute_overall = True
                    save_state_to_file(f_out, prior_distribution, early_stopping_stats, bayesian_updater, final=True)
                    break
                
                # Save intermediate state periodically
                save_state_to_file(f_out, prior_distribution, early_stopping_stats, bayesian_updater)

        # Final cleanup and state saving
        # TODO -- why doesn't this work if the stopping condition isnt met and we get to the final chunk? total_values is 0
        finalize_processing(f_out, digest, compute_overall, overall_sum, 
                          overall_sumsq, overall_min, overall_max, total_values)
    
        if early_stopping_requested:
            return EARLY_STOPPING


def calculate_min_chunks_and_window_size(confidence):
    """
    Calculate min_chunks and window_size based on the confidence level
    using the normal distribution from SciPy.

    """
    # Calculate the Z-score for the given confidence level
    z_score = norm.ppf((1 + confidence) / 2)  # Two-tailed confidence

    # Use the Z-score to determine min_chunks and window_size
    min_chunks = math.ceil(1 / (1 - confidence))  # Approximation for min_chunks
    window_size = math.ceil(z_score**2) + 1 # Window size is based on Z-score significance
    logger.debug(f"calculated {min_chunks} min chunks and {window_size} window size from z-score: {z_score} and confidence: {confidence}")

    return min_chunks, window_size



def calculate_min_tokens(min_chunks: int, sample_file: str) -> int:
    """
    MAIN helper function
    Calculate the minimum number of tokens required based on min_chunks and context_size from sample file.
    """
    # Determine context_size from the target file
    with h5py.File(sample_file, 'r') as f_sample:
        context_size = f_sample['logits'].shape[1]
        logger.debug(f"calculated {min_chunks * context_size} min_samples from min_chunks {min_chunks} shape of each chunk: {f_sample['logits'].shape} context size: {context_size}")

    return min_chunks * context_size


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Calculate KL-divergence between two logits HDF5 files. Supports early-stopping for initial run to avoid unnecessary testing.")
    
    parser.add_argument('baseline_file', type=str, help="Path to the baseline logits HDF5 file.")
    parser.add_argument('target_file', type=str, help="Path to the target logits HDF5 file.")
    parser.add_argument('--output-file', type=str, help="Optional path to save KL-divergence statistics.")
    parser.add_argument('--from-chunk', type=int, help="Starting chunk index for processing (inclusive).")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (inclusive).")
    parser.add_argument('--clobber', action='store_true', help="Allow overwriting of existing output file data.")
    parser.add_argument('--precision', type=int, choices=[32, 64], default=64,
                        help="Precision of the calculations on the logits for kl-divergence (default: 64). Note: currently llama.cpp only supports fp32 for processing the output weights.")
    parser.add_argument('--parts', type=int, default=1, help="Number of parts to split each chunk into for processing.")
    parser.add_argument('--compute-overall', action='store_true',
                        help="Compute overall statistics even if early stopping occurs.")
    
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
    early_stopping_group.add_argument('--effective_chunk_size', type=int,
                                      help="Sample size divisor to use for the Kuiper test. It should divide the context-size parameter x of the shape of the logits (x,y) in a chunk. (can be automatically generated)")

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

    # Verbosity arguments
    early_stopping_group.add_argument('--log-effect-sizes', action='store_true',
                                      help="Enable logging of effect sizes for each chunk.")
    parser.add_argument('--verbosity', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help="Set the output verbosity level (default: INFO).")

    parser.epilog = """
Statistical tools for early stopping include the Kuiper test, which performs a bootstrapped analysis to derive meaningful p-values and assess the statistical significance of differences between the model's predictions and the baseline. A beta distribution model is employed to estimate the probability of stopping based on the stability of effect sizes and p-values, helping to determine whether further processing is necessary. Additionally, traditional p-value comparison is used when the chunk window is not full, relying on a likelihood argument to evaluate the need for early stopping. By analyzing the stability of effect sizes and p-values across chunks in a reference run, these tools can guide the selection of an appropriate dataset size for additional runs.
"""

    args = parser.parse_args()
    setattr(args, "dynamic_thresholds", False)

    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))
    logger.info(f"compare_logits (version {__version__})")

    min_tokens = None

    if args.early_stopping:
        setattr(args, "dynamic_thresholds", args.theta_E is None and args.theta_P is None)

        # Set defaults only for arguments that are None
        for arg, default in EARLY_STOPPING_DEFAULTS.items():
            if getattr(args, arg) is None:
                setattr(args, arg, default)

        min_chunks, window_size = args.min_chunks, args.window_size
        if min_chunks is None or window_size is None:
            calculated_min_chunks, calculated_window_size = calculate_min_chunks_and_window_size(args.confidence)
            args.min_chunks = min_chunks or calculated_min_chunks
            args.window_size = window_size or calculated_window_size

        min_tokens = calculate_min_tokens(args.min_chunks, args.baseline_file)

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

    # Call process_chunks with parsed arguments
    process_chunks(
        baseline_path=args.baseline_file,
        target_path=args.target_file,
        output_path=args.output_file,
        from_chunk=args.from_chunk,
        to_chunk=args.to_chunk,
        clobber=args.clobber,
        precision=args.precision,
        parts=args.parts,
        early_stopping=args.early_stopping,
        confidence_level=args.confidence,
        margin_of_error=args.margin_of_error,
        compute_overall=args.compute_overall,
        min_samples=min_tokens,
        momentum=args.momentum,
        prior_learning_rate=args.learning_rate,
        min_prior_weight=args.min_prior_weight,
        decay_rate=args.decay_rate,
        effective_chunk_size=args.effective_chunk_size,
        window_size=args.window_size,
        theta_E=args.theta_E,
        theta_P=args.theta_P,
        dynamic_thresholds_enabled=args.dynamic_thresholds,
        log_effect_sizes=args.log_effect_sizes,
    )
