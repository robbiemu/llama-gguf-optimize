import h5py
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import json


def calculate_composite_metric(stats):
    """
    Calculate the composite metric based on the provided formula.
    """
    median = stats.get('Median', 0.0)
    kld_99 = stats.get('KLD_99', 0.0)
    kld_95 = stats.get('KLD_95', 0.0)
    kld_90 = stats.get('KLD_90', 0.0)

    score = (
        (median ** (1 / 3)) *
        ((kld_99 * 1 + kld_95 * 4 + kld_90 * 5) ** (2 / 3))
    )

    return score


def read_kl_divergence(h5_files, from_chunk=None, to_chunk=None, overall=False):
    """
    Reads and processes KL-divergence statistics for multiple files.

    Args:
        h5_files (list): List of HDF5 file paths.
        from_chunk (int, optional): Starting chunk number.
        to_chunk (int, optional): Ending chunk number.
        overall (bool): Whether to process overall stats.

    Returns:
        tuple: (chunk_data, overall_data)
    """
    all_chunk_data = []
    all_overall_data = []

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            # Get all chunk keys
            all_chunks = sorted([key for key in f.keys() if key.startswith("chunk_")])
            chunk_indices = [int(chunk.split("_")[1]) for chunk in all_chunks]

            if not chunk_indices:
                print(f"No chunks found in {h5_file}. Skipping.")
                continue

            # Determine range of chunks to process
            from_chunk_actual = from_chunk if from_chunk is not None else min(chunk_indices)
            to_chunk_actual = to_chunk if to_chunk is not None else max(chunk_indices)

            for chunk in range(from_chunk_actual, to_chunk_actual + 1):
                chunk_key = f"chunk_{chunk}"
                if chunk_key in f:
                    chunk_stats = dict(f[chunk_key].attrs)
                    composite_metric = calculate_composite_metric(chunk_stats)
                    all_chunk_data.append({
                        "File": h5_file,
                        "Composite Metric": composite_metric,
                        "Chunk": chunk
                    })

            # Process overall stats if present
            if overall and "overall" in f.attrs:
                try:
                    overall_stats = json.loads(f.attrs["overall"])  # Deserialize JSON string
                    # Verify required percentile keys
                    required_keys = {"KLD_99", "KLD_95", "KLD_90", "KLD_10", "KLD_05", "KLD_01", "Median", "Average", "StdDev", "Minimum", "Maximum"}
                    if not required_keys.issubset(overall_stats.keys()):
                        missing = required_keys - set(overall_stats.keys())
                        print(f"Skipping {h5_file} due to missing overall stats keys: {missing}")
                        continue

                    composite_metric = calculate_composite_metric(overall_stats)
                    overall_stats.update({
                        "File": h5_file,
                        "Composite Metric": composite_metric
                    })
                    all_overall_data.append(overall_stats)
                except json.JSONDecodeError:
                    print(f"Skipping {h5_file} due to invalid JSON in 'overall' attribute.")
                except Exception as e:
                    print(f"Skipping {h5_file} due to error processing 'overall' attribute: {e}")

    return all_chunk_data, all_overall_data


def approximate_distribution_from_overall(stats, num_samples=10000):
    """
    Approximate a distribution based on overall statistics.

    Args:
        stats (dict): Overall statistics dictionary.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Synthetic data approximating the overall distribution.
    """
    percentiles = {
        "P99": stats["KLD_99"],
        "P95": stats["KLD_95"],
        "P90": stats["KLD_90"],
        "P10": stats["KLD_10"],
        "P05": stats["KLD_05"],
        "P01": stats["KLD_01"]
    }
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


def allocate_bins(bins, cum_prob_diffs):
    # Initial allocation
    bins_per_interval = np.floor(bins * cum_prob_diffs).astype(int)
    total_bins_allocated = np.sum(bins_per_interval)
    bins_remaining = bins - total_bins_allocated

    # Distribute remaining bins
    fractional_parts = (bins * cum_prob_diffs) - bins_per_interval
    indices = np.argsort(-fractional_parts)  # Indices with largest fractional parts
    for i in range(bins_remaining):
        bins_per_interval[indices[i]] += 1

    return bins_per_interval


def plot_3d_overall_metrics_all(overall_data, num_samples=10000, bins=500):
    """
    Generate a single 3D plot for all overall KL-divergence statistics.

    Args:
        overall_data (list): List of overall statistics dictionaries.
        num_samples (int): Number of samples to generate for each distribution.
        bins (int): Number of grid points for the KDE.
    """
    X_list, Y_list, Z_list, C_list = [], [], [], []
    file_labels = []
    composite_metrics = []

    # Assign unique X values to each file
    file_indices = {stats['File']: idx for idx, stats in enumerate(overall_data)}

    for stats in overall_data:
        file_index = file_indices[stats['File']]
        try:
            # Generate the synthetic distribution
            distribution = approximate_distribution_from_overall(stats, num_samples=num_samples)
        except ValueError as e:
            print(f"Skipping 3D plot for {stats['File']} due to missing data: {e}")
            continue

        # Collect cumulative probabilities and percentiles
        cumulative_probs = [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0]
        percentiles = [
            stats["Minimum"],
            stats["KLD_01"],
            stats["KLD_05"],
            stats["KLD_10"],
            stats["Median"],
            stats["KLD_90"],
            stats["KLD_95"],
            stats["KLD_99"],
            stats["Maximum"]
        ]

        # Compute cumulative probability differences
        cum_prob_diffs = np.diff(cumulative_probs)

        # Allocate bins per interval
        bins_per_interval = allocate_bins(bins, cum_prob_diffs)

        # Create grid for KDE
        x_grid = []
        for i in range(len(percentiles) - 1):
            start = percentiles[i]
            end = percentiles[i + 1]
            num_points = bins_per_interval[i] + 1  # +1 to include end point
            if num_points >= 2:
                grid = np.linspace(start, end, num_points)
                x_grid.extend(grid[:-1])  # Exclude last point to avoid duplicates
        x_grid.append(percentiles[-1])  # Include the maximum value at the end

        # Convert to NumPy array and clip to the actual maximum
        x_grid = np.array(x_grid)
        actual_max = stats["Maximum"]  # Clip to actual maximum
        x_grid = x_grid[x_grid <= actual_max]

        # Calculate density using KDE
        bw = min(stats["StdDev"] / 4, (stats["Maximum"] - stats["Minimum"]) / 50)
        kde = gaussian_kde(distribution, bw_method=bw)
        density = kde(x_grid)

        # Normalize the density to ensure the total area = 1
        area = np.trapezoid(density, x_grid)
        density /= area

        # Apply ultra-compressed density transformation
        compressed_density = np.log2(1 + np.log2(1 + np.log2(1 + density)))

        # Prepare data for 3D plotting
        X = np.full_like(x_grid, file_index)
        Y = compressed_density
        Z = x_grid
        C = x_grid  # Color mapping to KL-divergence values

        # Append data within valid range
        X_list.extend(X)
        Y_list.extend(Y)
        Z_list.extend(Z)
        C_list.extend(C)

        # Collect labels and composite metrics for annotations
        file_labels.append(stats['File'])
        composite_metrics.append(stats['Composite Metric'])

    # Convert lists to arrays
    X = np.array(X_list)
    Y = np.array(Y_list)
    Z = np.array(Z_list)
    C = np.array(C_list)

    # Check if there is data to plot
    if len(X) == 0:
        print("No data available for 3D plotting.")
        return

    # Create the 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Triangulate the data
    tesselation = Triangulation(X, Z)
    
    # Mask triangles based on your criteria
    # Here, we assume you want to mask triangles where both other vertices have higher Z-values than the current file's maximum Z-value
    mask = np.zeros(tesselation.triangles.shape[0], dtype=bool)
    
    # Compute the maximum Z value per file index
    max_z_per_file = {file_index: np.max(Z[X == file_index]) for file_index in np.unique(X)}

    print(f"Total triangles: {len(tesselation.triangles)}")

    for i, triangle in enumerate(tesselation.triangles):
        z_vals = Z[triangle]
        file_vals = X[triangle]

        # Check the triangle for each vertex's file
        for x_vertex_index in range(3):  # Iterate over all 3 vertices
            x_file = file_vals[x_vertex_index]
            x_max_z = max_z_per_file[x_file]

            # Check if the other two vertices have Z > x_max_z for this file
            other_vertices = [v for v in range(3) if v != x_vertex_index]
            if all(z_vals[v] > x_max_z for v in other_vertices):
                mask[i] = True
                print(f"Triangle {i}: Masked because other vertices exceed max_z={x_max_z} for file {x_file}.")
                break  # Stop checking this triangle further once it's masked

    # Apply the mask to the triangles
    tesselation.set_mask(mask)

    # Extract valid triangles for plotting
    valid_triangles = tesselation.get_masked_triangles()

    # Plot only the valid triangles
    ax.plot_trisurf(
        Z, X, Y, cmap="viridis", edgecolor="none", antialiased=True, alpha=0.8,
        triangles=valid_triangles
    )

    # Set labels
    ax.set_xlabel("KL-Divergence Value (Z)")
    ax.set_ylabel("File Index (X)")
    ax.set_zlabel("Density (Y)")
    ax.set_title("3D KL-Divergence Manifold Across Files", fontsize=16)

    # Create a legend manually using annotations
    for idx, (file, metric) in enumerate(zip(file_labels, composite_metrics)):
        ax.text(
            np.mean(Z[X == idx]),  # Average Z for positioning
            idx,
            np.max(Y[X == idx]),
            f"{file}\nMetric: {metric:.4f}",
            fontsize=10,
            ha='center',
            va='bottom'
        )

    plt.tight_layout()


def plot_results(chunk_data, overall_data):
    """
    Plot the composite metrics for chunks and overall statistics.

    Args:
        chunk_data (list): List of dictionaries with chunk-level data.
        overall_data (list): List of dictionaries with overall-level data.
    """
    # Convert chunk data to DataFrame
    chunk_df = pd.DataFrame(chunk_data)
    chunk_df['File'] = pd.Categorical(chunk_df['File'], categories=args.h5_files, ordered=True)

    # Plot per-chunk composite metrics
    if not chunk_df.empty:
        chunk_pivot = chunk_df.pivot_table(index="Chunk", columns="File", values="Composite Metric")
        ax = chunk_pivot.plot(kind='bar', figsize=(16, 8))

        ax.set_title("Composite Metrics by Chunk", fontsize=16)
        ax.set_xlabel("Chunk", fontsize=14)
        ax.set_ylabel("Composite Metric", fontsize=14)
        ax.legend(title="Files", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
        plt.tight_layout()

    # Plot 3D visualization for overall metrics
    if overall_data:
        plot_3d_overall_metrics_all(overall_data)
    else:
        print("No overall data available for 3D plotting.")


def main(**args):
    """
    Main function to process HDF5 files and plot composite metrics.
    """
    # Configure pandas to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Determine if overall stats should be processed by default
    if args['to_chunk'] is None and not args['overall']:
        args['overall'] = True

    chunk_data, overall_data = read_kl_divergence(
        args['h5_files'], args['from_chunk'], args['to_chunk'], args['overall']
    )

    if chunk_data:
        print("\n===== Chunk-Level Composite Metrics =====")
        chunk_df = pd.DataFrame(chunk_data)
        print(chunk_df)
    else:
        print("\nNo chunk-level data available.")
        chunk_df = pd.DataFrame()

    if overall_data:
        print("\n===== Overall Composite Metrics =====")
        overall_df = pd.DataFrame(overall_data)
        print(overall_df[['File', 'Composite Metric']])
    else:
        print("\nNo overall data available.")
        overall_df = pd.DataFrame()

    plot_results(chunk_data, overall_data)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare and plot composite metrics from multiple HDF5 files.")
    parser.add_argument("h5_files", nargs="+", help="Paths to the HDF5 files containing KL-divergence data.")
    parser.add_argument("--from-chunk", type=int, help="Starting chunk number (optional).")
    parser.add_argument("--to-chunk", type=int, help="Ending chunk number (optional).")
    parser.add_argument("--overall", action='store_true', help="Include overall stats (defaults to true if --to-chunk is not given)")

    args = parser.parse_args()

    main(**vars(args))
