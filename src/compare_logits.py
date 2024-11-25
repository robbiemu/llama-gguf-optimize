import h5py
import json
import logging
import numpy as np
from pydantic import BaseModel, Field
from scipy.special import logsumexp
from tdigest import TDigest
from typing import Optional

from version import __version__
from gguf_optimize_logging import setup_logging

logger = logging.getLogger(__name__)

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

def numpy_encoder(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def calculate_statistics(kl_values, chunk_index):
    """Calculate statistics for a list of KL-divergence values."""
    stats = {
        "ChunkNumber": chunk_index,
        "Average": kl_values.mean(),
        "StdDev": kl_values.std(),
        "Median": np.median(kl_values),
        "Minimum": kl_values.min(),
        "Maximum": kl_values.max(),
        "KLD_99": np.percentile(kl_values, 99),
        "KLD_95": np.percentile(kl_values, 95),
        "KLD_90": np.percentile(kl_values, 90),
        "KLD_10": np.percentile(kl_values, 10),
        "KLD_05": np.percentile(kl_values, 5),
        "KLD_01": np.percentile(kl_values, 1),
    }
    return stats

def check_output_file_conditions(output_path, from_chunk, to_chunk, clobber):
    """Check output file for existing data and determine if conditions allow processing."""
    existing_chunks, overall_stats, digest = set(), None, None

    if output_path and h5py.is_hdf5(output_path):
        with h5py.File(output_path, 'r') as f_out:
            existing_chunks = {int(chunk.split('_')[1]) for chunk in f_out.keys() if chunk.startswith("chunk_")}
            has_overall = 'overall' in f_out.attrs
            has_digest = 'digest' in f_out.attrs

            # If clobber is enabled, clear the file and exit the function early
            if clobber:
                logger.info(f"Clobber mode enabled; clearing all existing data in {output_path}.")
                f_out.close()
                with h5py.File(output_path, 'w') as f_out:
                    pass  # Clears the file by reopening in write mode
                return set(), None, None  # Return empty, as file is now cleared

            # Load and validate overall stats if they exist
            if has_overall:
                overall_stats = json.loads(f_out.attrs['overall'])
            else:
                logger.debug("No overall stats found in the file.")

            # Load and validate the digest if it exists
            if has_digest and existing_chunks:
                digest_dict = json.loads(f_out.attrs['digest'])
                digest = TDigest()
                digest.update_from_dict(digest_dict)
            elif has_digest and not existing_chunks:
                raise ValueError("Digest centroids present without processed chunks.")
            else:
                logger.info("No digest centroids found in the file.")

            # Perform additional checks on existing data
            if has_overall and not existing_chunks:
                raise ValueError(f"Output file contains only 'overall' property without any chunk data. Use --clobber to start fresh. (from {from_chunk}, to {to_chunk})")
            if existing_chunks and not {'overall_sum', 'overall_sumsq', 'overall_min', 'overall_max', 'total_values'}.issubset(f_out.attrs):
                raise ValueError(f"Output file contains partial chunk data without an 'overall' property. Use --clobber to start fresh or specify a valid range with --from and --to. (from {from_chunk}, to {to_chunk})")
            if existing_chunks and has_overall:
                raise ValueError(f"Output file already contains completed data (chunks and 'overall' property). Use --clobber to start fresh. (from {from_chunk}, to {to_chunk})")

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

# def kl_divergence_log_probs(p_logits, q_logits):
#     """
#     Calculate KL divergence using log-softmax probabilities in a numerically stable way.
    
#     Args:
#         p_logits: Logits from the baseline model.
#         q_logits: Logits from the target model.
#     """
#     # Compute log-softmax probabilities for p and q
#     p_log_probs = p_logits - logsumexp(p_logits, axis=-1, keepdims=True)
#     q_log_probs = q_logits - logsumexp(q_logits, axis=-1, keepdims=True)
    
#     # Compute KL divergence in a numerically stable way:
#     # KL = sum(p * (log(p) - log(q)))
#     kl = np.exp(p_log_probs) * (p_log_probs - q_log_probs)
    
#     return np.sum(kl, axis=-1)


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
        logger.debug(f"Sample p_logits: {p_logits_part[sample_idx,:10]}")
        logger.debug(f"Sample q_logits: {q_logits_part[sample_idx,:10]}")
    
    return kl_values


def process_chunks(
    baseline_path,
    target_path,
    output_path: Optional[str] = None,
    from_chunk: Optional[int] = None,
    to_chunk: Optional[int] = None,
    clobber: bool = False,
    precision: int = 64,
    parts: int = 1
):
    # Load existing file conditions
    existing_chunks, overall_stats, digest = check_output_file_conditions(output_path, from_chunk, to_chunk, clobber)

    # Initialize TDigest if not loaded from file
    if digest is None:
        digest = TDigest()
    # Initialize overall stats if not loaded from file
    if overall_stats:
        overall_sum = overall_stats["Average"] * overall_stats["total_values"]
        overall_sumsq = (overall_stats["StdDev"] ** 2 + overall_stats["Average"] ** 2) * overall_stats["total_values"]
        overall_min = overall_stats["Minimum"]
        overall_max = overall_stats["Maximum"]
        total_values = overall_stats["total_values"]
    else:
        overall_sum, overall_sumsq, overall_min, overall_max, total_values = 0.0, 0.0, float('inf'), float('-inf'), 0

    with h5py.File(baseline_path, 'r') as f_baseline, h5py.File(target_path, 'r') as f_target, h5py.File(output_path, 'a') as f_out:
        logits_baseline = f_baseline['logits']
        total_chunks = logits_baseline.shape[0]

        # Load chunk_index datasets
        chunk_index_baseline = f_baseline['chunk_index'][:]
        chunk_index_target = f_target['chunk_index'][:]

        # Determine the range of chunks to process
        start_chunk = from_chunk if from_chunk is not None else 0
        end_chunk = to_chunk if to_chunk is not None else total_chunks - 1
        logger.info(f"Processing chunks {start_chunk} to {end_chunk}...")

        dtype = np.float32 if precision == 32 else np.float64

        for chunk_idx in range(start_chunk, end_chunk + 1):
            if chunk_idx in existing_chunks:
                logger.info(f"Skipping already processed chunk {chunk_idx}")
                continue

            chunk_shape = logits_baseline[chunk_idx].shape
            num_samples = chunk_shape[0]

            if num_samples == 0:
                logger.warning(f"Chunk {chunk_idx} has zero samples, skipping.")
                continue

            # Initialize per-chunk accumulators
            chunk_sum = 0.0
            chunk_sumsq = 0.0
            chunk_min = float('inf')
            chunk_max = float('-inf')
            chunk_total_values = 0

            # Initialize per-chunk digest
            chunk_digest = TDigest()

            for part_idx in range(parts):
                # Get the correct physical index for this logical chunk using chunk_index
                physical_index_baseline = chunk_index_baseline[chunk_idx]
                physical_index_target = chunk_index_target[chunk_idx]
                assert (physical_index_baseline == physical_index_target), "baseline and target out of step"

                if physical_index_baseline < 0 or physical_index_target < 0:
                    logger.warning(f"Logical chunk {chunk_idx} is not present in one of the files; skipping.")
                    continue

                physical_index = physical_index_baseline

                # Validate the indices to ensure the chunk is present
                if physical_index_baseline < 0 or physical_index_target < 0:
                    logger.warning(f"Chunk {chunk_idx} is not present in one of the files; skipping.")
                    continue

                # Calculate start and end indices for partitioning this chunk
                start_idx = (num_samples * part_idx) // parts
                end_idx = (num_samples * (part_idx + 1)) // parts

                if start_idx >= end_idx:
                    continue

                # Access the logits at the physical index, not the logical index
                p_logits_part = logits_baseline[physical_index_baseline][start_idx:end_idx].astype(dtype)
                q_logits_part = f_target['logits'][physical_index_target][start_idx:end_idx].astype(dtype)

                kl_values_part = process_chunk_part(p_logits_part, q_logits_part, physical_index, part_idx)

                # Check for invalid values in output
                if np.any(np.isnan(kl_values_part)) or np.any(np.isinf(kl_values_part)):
                    logger.warning(f"Found NaN or Inf in KL divergence for chunk {chunk_idx}, part {part_idx}")
                    continue

                # Update per-chunk accumulators
                chunk_sum += np.sum(kl_values_part)
                chunk_sumsq += np.sum(kl_values_part ** 2)
                chunk_min = min(chunk_min, kl_values_part.min())
                chunk_max = max(chunk_max, kl_values_part.max())
                chunk_total_values += len(kl_values_part)

                # Update per-chunk digest
                chunk_digest.batch_update(kl_values_part)

                # Update overall digest
                digest.batch_update(kl_values_part)

                # Clean up
                del p_logits_part, q_logits_part, kl_values_part

            # After processing all parts, compute chunk statistics
            if chunk_total_values == 0:
                logger.warning(f"No valid data in chunk {chunk_idx} after processing all parts.")
                continue

            chunk_mean = chunk_sum / chunk_total_values
            chunk_variance = chunk_sumsq / chunk_total_values - chunk_mean ** 2
            chunk_stddev = np.sqrt(max(0, chunk_variance))

            # Compute percentiles using per-chunk digest
            chunk_stats = {
                "ChunkNumber": chunk_idx,
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
            }

            # Save chunk stats
            f_out.create_group(f'chunk_{chunk_idx}').attrs.update(chunk_stats)

            print(f"\n===== KL-divergence statistics for Chunk {chunk_idx} =====")
            for key, value in chunk_stats.items():
                if key == "ChunkNumber":
                    continue
                else:
                    print(f"{key:8}: {value:.6f}")

            # Update cumulative overall stats
            overall_sum += chunk_sum
            overall_sumsq += chunk_sumsq
            overall_min = min(overall_min, chunk_min)
            overall_max = max(overall_max, chunk_max)
            total_values += chunk_total_values

            # Save cumulative statistics for overall
            f_out.attrs['overall_sum'] = overall_sum
            f_out.attrs['overall_sumsq'] = overall_sumsq
            f_out.attrs['overall_min'] = overall_min
            f_out.attrs['overall_max'] = overall_max
            f_out.attrs['total_values'] = total_values

        # Serialize TDigest using to_dict() and save in HDF5 as JSON-compatible data
        digest_dict = digest.to_dict()
        f_out.attrs['digest'] = json.dumps(digest_dict, default=numpy_encoder)  # Convert dict to string for storage

        print(f"TOTAL CHUNKS {total_chunks} (processed from {start_chunk}, to {end_chunk})")

        # Final overall statistics based on cumulative data
        if end_chunk == total_chunks - 1:  # Only finalize if we're at the end
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
                "KLD_10": digest.percentile(10),
                "KLD_05": digest.percentile(5),
                "KLD_01": digest.percentile(1),
                "total_values": total_values
            }
            print("\n===== Overall KL-divergence statistics =====")
            for key, value in overall_stats.items():
                if key != "total_values":
                    print(f"{key:8}: {value:.6f}")

            f_out.attrs['overall'] = json.dumps(overall_stats, default=numpy_encoder)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Calculate KL-divergence between two logits HDF5 files.")
    parser.add_argument('baseline_file', type=str, help="Path to the baseline logits HDF5 file.")
    parser.add_argument('target_file', type=str, help="Path to the target logits HDF5 file.")
    parser.add_argument('--output_file', type=str, help="Optional path to save KL-divergence statistics.")
    parser.add_argument('--from-chunk', type=int, help="Starting chunk index for processing (inclusive).")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (inclusive).")
    parser.add_argument('--clobber', action='store_true', help="Allow overwriting of existing output file data.")
    parser.add_argument('--verbosity', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the output verbosity level (default: INFO).")
    parser.add_argument('--precision', type=int, choices=[32,64], default=64, help='Precision of the calculations on the logits for kl-divergence (default: 64) note: currently llama.cpp only supports fp32 for processing the output weights.')
    parser.add_argument('--parts', type=int, default=1, help="Number of parts to split each chunk into for processing.")

    args = parser.parse_args()

    setup_logging(getattr(logging, args.verbosity.upper(), logging.INFO))
    logging.info(f"compare_logits starting (version {__version__})")

    process_chunks(
        baseline_path=args.baseline_file,
        target_path=args.target_file,
        output_path=args.output_file,
        from_chunk=args.from_chunk,
        to_chunk=args.to_chunk,
        clobber=args.clobber,
        precision=args.precision,
        parts=args.parts,
    )
