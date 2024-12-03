import h5py
import numpy as np
import json
import sys

def compute_and_save_overall(filename):
    with h5py.File(filename, "a") as f:
        if not all(key in f.attrs for key in ["overall_sum", "overall_sumsq", "overall_min", "overall_max", "total_values"]):
            raise ValueError("Required attributes ('overall_sum', 'overall_sumsq', 'overall_min', 'overall_max', 'total_values') are missing in the file.")

        overall_sum = f.attrs["overall_sum"]
        overall_sumsq = f.attrs["overall_sumsq"]
        overall_min = f.attrs["overall_min"]
        overall_max = f.attrs["overall_max"]
        total_values = f.attrs["total_values"]

        overall_mean = overall_sum / total_values
        variance = (overall_sumsq / total_values) - (overall_mean ** 2)
        stddev = np.sqrt(max(0, variance))  # Ensure non-negative variance

        overall_stats = {
            "Average": overall_mean,
            "StdDev": stddev,
            "Minimum": overall_min,
            "Maximum": overall_max,
        }

        # Add percentiles using the TDigest, if available
        digest = f.attrs.get("digest")
        if digest:
            from tdigest import TDigest
            tdigest = TDigest()
            tdigest.update_from_dict(json.loads(digest))
            overall_stats.update({
                "KLD_99": tdigest.percentile(99),
                "KLD_95": tdigest.percentile(95),
                "KLD_90": tdigest.percentile(90),
                "Median": tdigest.percentile(50),  # Median is the 50th percentile
                "KLD_10": tdigest.percentile(10),
                "KLD_05": tdigest.percentile(5),
                "KLD_01": tdigest.percentile(1),
            })

        # Save to "overall" attribute
        f.attrs["overall"] = json.dumps(overall_stats)

        print(f"Successfully computed and saved 'overall' attribute to {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_overall.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    compute_and_save_overall(filename)

