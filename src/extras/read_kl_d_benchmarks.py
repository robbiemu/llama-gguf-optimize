import h5py
import json


def read_kl_divergence(h5_file, from_chunk=None, to_chunk=None, overall=False):
    """
    Reads and prints KL-divergence statistics for chunks in the specified range.
    If no range is specified, processes all available chunks.
    """
    with h5py.File(h5_file, "r") as f:
        # Get all chunk keys
        all_chunks = sorted([key for key in f.keys() if key.startswith("chunk_")])
        chunk_indices = [int(chunk.split("_")[1]) for chunk in all_chunks]
        
        # Determine range of chunks to process
        from_chunk = from_chunk if from_chunk is not None else min(chunk_indices)
        to_chunk = to_chunk if to_chunk is not None else max(chunk_indices)
        
        for chunk in range(from_chunk, to_chunk + 1):
            chunk_key = f"chunk_{chunk}"
            if chunk_key in f:
                chunk_stats = dict(f[chunk_key].attrs)
                print(f"\n===== KL-divergence statistics for Chunk {chunk} =====")
                print_statistics(chunk_stats)
            else:
                print(f"\n[WARNING] Chunk {chunk} not found in the file.")

        if overall and "overall" in f.attrs:
            overall_stats = json.loads(f.attrs["overall"])
            print("\n===== Overall KL-divergence statistics =====")
            print_statistics(overall_stats)


def print_statistics(chunk_stats):
    """
    Prints KL-divergence statistics from chunk attributes.
    """
    for key, value in chunk_stats.items():
        if key == "ChunkNumber":
            continue
        print(f"{key:8}: {value:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract KL-divergence statistics from HDF5 files.")
    parser.add_argument("h5_file", help="Path to the HDF5 file containing KL-divergence data.")
    parser.add_argument("--from-chunk", type=int, help="Starting chunk number (optional).")
    parser.add_argument("--to-chunk", type=int, help="Ending chunk number (optional).")
    parser.add_argument("--overall", type=bool, help="include overall stats (defaults to true if --to-chunk is not given)")

    args = parser.parse_args()

    if args.to_chunk is None and args.overall is None:
        args.overall = True

    read_kl_divergence(**args.__dict__)
