import h5py
import json
import logging
import numpy as np
from pydantic import BaseModel, Field
from scipy.special import rel_entr
from tdigest import TDigest
from typing import Optional


LOG_LEVEL = logging.INFO

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


def configure_logger(external_logger=None):
    """Configures a logger for generate_logits. Uses an external logger if provided."""
    global logger  # Make logger accessible throughout the module
    if external_logger:
        logger = external_logger
    else:
        logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        logger = logging.getLogger(__name__)


def kl_divergence(p, q):
    """Calculate the KL-divergence between two probability distributions."""
    return np.sum(rel_entr(p, q), axis=-1)


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
                logger.info("No overall stats found in the file.")

            # Load and validate the digest if it exists
            if has_digest:
                digest_dict = json.loads(f_out.attrs['digest'])
                digest = TDigest()
                digest.update_from_dict(digest_dict)
            else:
                logger.info("No digest centroids found in the file.")

            # Perform additional checks on existing data
            if has_overall and not existing_chunks:
                raise ValueError("Output file contains only 'overall' property without any chunk data. Use --clobber to start fresh.")
            if existing_chunks and not has_overall:
                raise ValueError("Output file contains partial chunk data without an 'overall' property. Use --clobber to start fresh or specify a valid range with --from and --to.")
            if existing_chunks and has_overall:
                raise ValueError("Output file already contains completed data (chunks and 'overall' property). Use --clobber to start fresh.")

            # Check for contiguous chunks if `--from` is set
            if from_chunk is not None:
                required_chunks = set(range(0, from_chunk))
                if not required_chunks.issubset(existing_chunks) and from_chunk != 0:
                    raise ValueError(f"Non-contiguous range detected. All chunks up to {from_chunk - 1} must be present to resume from {from_chunk}.")
            
    return existing_chunks, overall_stats, digest


def process_chunks(baseline_path, target_path, output_path: Optional[str] = None, from_chunk: Optional[int] = None, to_chunk: Optional[int] = None, clobber: bool = False):
    # Load existing file conditions
    existing_chunks, overall_stats, digest = check_output_file_conditions(output_path, from_chunk, to_chunk, clobber)

    # Initialize TDigest if not loaded from file
    if digest is None:
        digest = TDigest()
    # Initialize overall stats if not loaded from file
    if overall_stats:
        overall_sum = overall_stats["Average"] * overall_stats["total_values"]
        overall_sumsq = (overall_stats["StdDev"] ** 2) * overall_stats["total_values"]
        overall_min = overall_stats["Minimum"]
        overall_max = overall_stats["Maximum"]
        total_values = overall_stats["total_values"]
    else:
        overall_sum, overall_sumsq, overall_min, overall_max, total_values = 0.0, 0.0, float('inf'), float('-inf'), 0

    with h5py.File(baseline_path, 'r') as f_baseline, h5py.File(target_path, 'r') as f_target, h5py.File(output_path, 'a') as f_out:
        logits_baseline = f_baseline['logits']
        total_chunks = logits_baseline.shape[0]

        # Determine the range of chunks to process
        start_chunk = from_chunk if from_chunk is not None else 0
        end_chunk = to_chunk if to_chunk is not None else total_chunks
        logger.info(f"Processing chunks {start_chunk} to {end_chunk}...")

        for chunk_idx in range(start_chunk, end_chunk):
            if chunk_idx in existing_chunks:
                logger.info(f"Skipping already processed chunk {chunk_idx}")
                continue
            
            # Calculate kl_values for the chunk
            p_logits = logits_baseline[chunk_idx]
            q_logits = f_target['logits'][chunk_idx]
            p = np.exp(p_logits) / np.sum(np.exp(p_logits), axis=-1, keepdims=True)
            q = np.exp(q_logits) / np.sum(np.exp(q_logits), axis=-1, keepdims=True)
            kl_values = kl_divergence(p, q)
            
            # Calculate and save chunk statistics
            chunk_stats = calculate_statistics(kl_values, chunk_idx)
            digest.batch_update(kl_values)
            f_out.create_group(f'chunk_{chunk_idx}').attrs.update(chunk_stats)
            
            print(f"\n===== KL-divergence statistics for Chunk {chunk_idx} =====")
            for key, value in chunk_stats.items():
                print(f"{key:8}: {value:.6f}")
            
            # Update cumulative overall stats
            chunk_mean = chunk_stats['Average']
            overall_sum += chunk_mean * len(kl_values)
            overall_sumsq += chunk_stats['StdDev'] ** 2 * len(kl_values)
            overall_min = min(overall_min, chunk_stats['Minimum'])
            overall_max = max(overall_max, chunk_stats['Maximum'])
            total_values += len(kl_values)

            del p_logits, q_logits, p, q, kl_values

        # Save cumulative statistics for overall
        f_out.attrs['overall_sum'] = overall_sum
        f_out.attrs['overall_sumsq'] = overall_sumsq
        f_out.attrs['overall_min'] = overall_min
        f_out.attrs['overall_max'] = overall_max
        f_out.attrs['total_values'] = total_values

        # Serialize TDigest using to_dict() and save in HDF5 as JSON-compatible data
        digest_dict = digest.to_dict()
        f_out.attrs['digest'] = str(digest_dict)  # Convert dict to string for storage

        # Final overall statistics based on cumulative data
        if end_chunk == total_chunks:  # Only finalize if we're at the end
            overall_stats = {
                "Average": overall_sum / total_values,
                "StdDev": np.sqrt(np.maximum(0, overall_sumsq / total_values - (overall_sum / total_values) ** 2)),
                "Minimum": overall_min,
                "Maximum": overall_max,
                "KLD_99": digest.percentile(99),
                "KLD_95": digest.percentile(95),
                "KLD_90": digest.percentile(90),
                "KLD_10": digest.percentile(10),
                "KLD_05": digest.percentile(5),
                "KLD_01": digest.percentile(1)
            }
            print("\n===== Overall KL-divergence statistics =====")
            for key, value in overall_stats.items():
                print(f"{key:8}: {value:.6f}")
                f_out.attrs['overall'] = json.dumps(overall_stats)


if __name__ == '__main__':
    import argparse

    configure_logger()
    
    parser = argparse.ArgumentParser(description="Calculate KL-divergence between two logits HDF5 files.")
    parser.add_argument('baseline_file', type=str, help="Path to the baseline logits HDF5 file.")
    parser.add_argument('target_file', type=str, help="Path to the target logits HDF5 file.")
    parser.add_argument('--output_file', type=str, help="Optional path to save KL-divergence statistics.")
    parser.add_argument('--from-chunk', type=int, help="Starting chunk index for processing (inclusive).")
    parser.add_argument('--to-chunk', type=int, help="Ending chunk index for processing (exclusive).")
    parser.add_argument('--clobber', action='store_true', help="Allow overwriting of existing output file data.")
    parser.add_argument('--verbosity', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the output verbosity level (default: INFO).")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.verbosity))
    
    process_chunks(
        baseline_path=args.baseline_file,
        target_path=args.target_file,
        output_path=args.output_file,
        from_chunk=args.from_chunk,
        to_chunk=args.to_chunk,
        clobber=args.clobber
    )
