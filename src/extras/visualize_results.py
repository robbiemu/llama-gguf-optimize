import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def process_hdf5_file(file_path, max_chunks=None):
    """
    Process the HDF5 file to extract KL-divergence statistics for chunks.

    Args:
        file_path (str): Path to the HDF5 file.
        max_chunks (int, optional): Maximum number of chunks to process. 
                                    If None, process all available chunks.

    Returns:
        list[dict]: List of dictionaries containing chunk statistics.
    """
    chunk_stats = []
    with h5py.File(file_path, 'r') as f:
        # Determine the total number of chunks if max_chunks is None
        if max_chunks is None:
            chunk_keys = [key for key in f.keys() if key.startswith('chunk_')]
            max_chunks = len(chunk_keys)

        for i in range(max_chunks):
            chunk_key = f'chunk_{i}'
            if chunk_key in f:
                stats = f[chunk_key].attrs
                chunk_stats.append({
                    "ChunkNumber": stats["ChunkNumber"],
                    "Average": stats["Average"],
                    "StdDev": stats["StdDev"],
                    "Median": stats["Median"],
                    "Minimum": stats["Minimum"],
                    "Maximum": stats["Maximum"],
                    "Percentiles": {
                        "P99": stats["KLD_99"],
                        "P95": stats["KLD_95"],
                        "P90": stats["KLD_90"],
                        "P10": stats["KLD_10"],
                        "P05": stats["KLD_05"],
                        "P01": stats["KLD_01"]
                    }
                })

    return chunk_stats


def approximate_distribution(stats, num_samples=1000):
    """
    Approximate a distribution based on statistics.

    Args:
        stats (dict): Dictionary of statistics including percentiles, mean, and std dev.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Synthetic data approximating the distribution.
    """
    percentiles = stats["Percentiles"]
    minimum = stats["Minimum"]
    maximum = stats["Maximum"]

    samples = np.linspace(0, 1, num_samples)
    approximate_values = np.piecewise(
        samples,
        [
            samples <= 0.01,
            (samples > 0.01) & (samples <= 0.05),
            (samples > 0.05) & (samples <= 0.1),
            (samples > 0.1) & (samples <= 0.9),
            (samples > 0.9) & (samples <= 0.95),
            (samples > 0.95) & (samples <= 0.99),
            samples > 0.99,
        ],
        [
            lambda s: np.random.uniform(minimum, percentiles["P01"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P01"], percentiles["P05"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P05"], percentiles["P10"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P10"], percentiles["P90"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P90"], percentiles["P95"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P95"], percentiles["P99"], size=s.shape),
            lambda s: np.random.uniform(percentiles["P99"], maximum, size=s.shape),
        ]
    )

    return np.sort(approximate_values)


def ultra_compress_density(density):
    """Triple logarithmic compression with scaling."""
    return np.log2(1 + np.log2(1 + np.log2(1 + density)))


def plot_3d_kl_divergence(chunk_stats, num_samples=1000, bins=50, debug_chunk=None):
    """
    Plot a 3D manifold of KL-divergence statistics using KDE with ultra-strong compression.
    """
    X, Y, Z, C = [], [], [], []

    for stat in chunk_stats:
        chunk_number = stat["ChunkNumber"]
        distribution = approximate_distribution(stat, num_samples=num_samples)

        left_edge = max(0, min(distribution))
        right_edge = max(distribution)
        p10 = stat["Percentiles"]["P10"]
        p90 = stat["Percentiles"]["P90"]

        x_grid = np.concatenate([
            np.linspace(left_edge, p10, bins // 2),
            np.linspace(p10, p90, bins // 3),
            np.linspace(p90, right_edge, bins // 6)
        ])
        x_grid = np.unique(x_grid)

        bw = min(stat["StdDev"] / 4, (stat["Maximum"] - stat["Minimum"]) / 50)
        kde = gaussian_kde(distribution, bw_method=bw)
        density = kde(x_grid)

        compressed_density = ultra_compress_density(density)

        Z.extend(x_grid)
        Y.extend(compressed_density)
        X.extend([chunk_number] * len(x_grid))
        C.extend(x_grid)  # Color mapped to KL-Divergence values

        if debug_chunk is not None and chunk_number == debug_chunk:
            plt.figure(figsize=(10, 6))
            plt.hist(distribution, bins=50, density=True, alpha=0.3, label="Raw Histogram")
            plt.plot(x_grid, density, 'r-', label="KDE", alpha=0.7)
            plt.plot(x_grid, compressed_density, 'g-', label="Ultra-Compressed KDE", alpha=0.7)
            plt.title(f"Distribution for Chunk {chunk_number}")
            plt.xlabel("KL-Divergence")
            plt.ylabel("Density / Compressed Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

    X, Y, Z, C = map(np.array, [X, Y, Z, C])
    target_points = 5000
    if len(X) > target_points:
        indices = np.linspace(0, len(X) - 1, target_points, dtype=int)
        X, Y, Z, C = X[indices], Y[indices], Z[indices], C[indices]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(Z, X, Y, cmap="viridis", edgecolor="none", facecolors=plt.cm.viridis(C / max(C)))

    ax.view_init(elev=45, azim=-60)
    ax.set_xlabel("KL-Divergence Value")
    ax.set_ylabel("Chunk Number")
    ax.set_zlabel("Density")
    ax.set_title("3D KL-Divergence Manifold (Color by KL-D)")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot 3D KL-divergence manifolds from HDF5 file.")
    parser.add_argument("filename", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--from-chunk", type=int, default=0, help="Starting chunk index (inclusive). Default is 0.")
    parser.add_argument("--to-chunk", type=int, default=477, help="Ending chunk index (exclusive). Default is 477.")
    parser.add_argument("--debug-chunk", type=int, help="Chunk to debug with a histogram.")
    parser.add_argument("--num-samples", type=int, default=1000, help="number of samples to generate to emulate the expected distribution. (defualt: 1000)")
    parser.add_argument("--bins", type=int, default=50, help="number of grid points to use in kernel density estimation. (defualt: 50)")
    args = parser.parse_args()

    chunk_stats = process_hdf5_file(args.filename, max_chunks=args.to_chunk)
    filtered_stats = [stat for stat in chunk_stats if args.from_chunk <= stat["ChunkNumber"] < args.to_chunk]

    plot_3d_kl_divergence(
        filtered_stats, num_samples=args.num_samples, bins=args.bins, 
        debug_chunk=args.debug_chunk
    )
