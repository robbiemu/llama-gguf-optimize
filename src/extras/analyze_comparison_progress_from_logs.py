import argparse
import json
import math
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import re
from scipy.stats import beta
import sys
import time
from functools import lru_cache
import os

@lru_cache(maxsize=1024)
def calculate_required_alpha(alpha, beta_value, confidence_level):
    """Pre-calculate required alpha value efficiently using binary search with memoization"""
    if beta.sf(confidence_level, alpha, beta_value) >= confidence_level:
        return alpha
    
    # Expand search range significantly
    left = alpha
    right = alpha + 10000  # Increased from 1000
    
    # Ensure we haven't hit the upper bound
    while beta.sf(confidence_level, right, beta_value) < confidence_level:
        right *= 2
    
    while left < right:
        mid = (left + right) // 2
        if beta.sf(confidence_level, mid, beta_value) >= confidence_level:
            right = mid
        else:
            left = mid + 1
            
    return left

class LogParser:
    def __init__(self):
        # Compile patterns once during initialization
        self.patterns = {
            "kuiper": re.compile(r"Kuiper statistic=([\d.]+), p-value=([\d.]+)"),
            "stopping": re.compile(r"Chunk (\d+): Beta parameters updated.*stopping probability=([\d.]+)"),
            "confidence": re.compile(r"confidence_level: ([\d.]+)"),
            "beta_params": re.compile(r"Updated Beta parameters: alpha=(\d+), beta=(\d+)"),
            "divisor": re.compile(r"Found largest divisor: (\d+) for chunk_size: (\d+)"),
            "ema": re.compile(r"Updated EMA_relative_change: ([\d.]+), EMA_p_value_std_dev: ([\d.]+)"),
            "theta": re.compile(r"Updated theta_E ([\d.]+) and theta_P ([\d.]+)")
        }
        self.last_modified = None
        self.last_size = None
        self.cached_results = None
        self.cached_file_position = 0
        self.largest_divisor = None

    def should_reparse(self, filename):
        """Check if file has been modified since last parse"""
        try:
            current_stat = os.stat(filename)
            current_modified = current_stat.st_mtime
            current_size = current_stat.st_size

            if (self.last_modified is None or 
                current_modified > self.last_modified or 
                current_size != self.last_size):
                self.last_modified = current_modified
                self.last_size = current_size
                return True
            return False
        except OSError:
            return True

    def parse_log(self, log_file):
        """Parse log file with incremental updates and caching"""
        if not self.should_reparse(log_file):
            return self.cached_results

        if self.cached_results is None:
            self.cached_results = {
                "chunks": [],
                "kuiper_statistics": [],
                "kuiper_p_values": [],
                "stopping_probabilities": [],
                "confidence_level": None,
                "chunk_size": None,
                "projected_chunks": [],
                "theta_E": [],
                "theta_P": [],
                "ema_relative_change": [],
                "ema_p_value_std_dev": [],
                "alpha_values": [],
                "beta_values": []
            }

        try:
            with open(log_file, 'r') as file:
                # Seek to last known position for incremental parsing
                file.seek(self.cached_file_position)
                new_lines = file.readlines()
                self.cached_file_position = file.tell()

                if not new_lines:
                    return self.cached_results

                # Parse only new lines
                for line in new_lines:
                    self._parse_line(line)

                # Update projected chunks if necessary
                self._update_projected_chunks()

                return self.cached_results

        except Exception as e:
            print(f"Error parsing log file: {e}")
            return self.cached_results

    def _parse_line(self, line):
        """Parse a single line and update cached results"""
        results = self.cached_results

        # Parse EMA and theta values
        ema_match = self.patterns["ema"].search(line)
        if ema_match:
            results["ema_relative_change"].append(float(ema_match.group(1)))
            results["ema_p_value_std_dev"].append(float(ema_match.group(2)))
            
        theta_match = self.patterns["theta"].search(line)
        if theta_match:
            results["theta_E"].append(float(theta_match.group(1)))
            results["theta_P"].append(float(theta_match.group(2)))
            
        if results["confidence_level"] is None:
            confidence_match = self.patterns["confidence"].search(line)
            if confidence_match:
                results["confidence_level"] = float(confidence_match.group(1))
            
        kuiper_match = self.patterns["kuiper"].search(line)
        if kuiper_match:
            results["kuiper_statistics"].append(float(kuiper_match.group(1)))
            results["kuiper_p_values"].append(float(kuiper_match.group(2)))
            
        stopping_match = self.patterns["stopping"].search(line)
        if stopping_match:
            results["chunks"].append(int(stopping_match.group(1)))
            results["stopping_probabilities"].append(float(stopping_match.group(2)))
            
        beta_params_match = self.patterns["beta_params"].search(line)
        if beta_params_match:
            results["alpha_values"].append(int(beta_params_match.group(1)))
            results["beta_values"].append(int(beta_params_match.group(2)))
            
        divisor_match = self.patterns["divisor"].search(line)
        if divisor_match and results["chunk_size"] is None:
            self.largest_divisor = int(divisor_match.group(1))
            chunk_size = int(divisor_match.group(2))
            results["chunk_size"] = chunk_size

    def _update_projected_chunks(self):
        """Update projected chunks calculation with cached values"""
        results = self.cached_results
        if not results["confidence_level"] or not self.largest_divisor:
            return

        current_projections = len(results["projected_chunks"])
        need_projections = len(results["alpha_values"])

        if current_projections < need_projections:
            for i in range(current_projections, need_projections):
                alpha = results["alpha_values"][i]
                beta_value = results["beta_values"][i]
                
                required_alpha = calculate_required_alpha(
                    alpha, 
                    beta_value, 
                    results["confidence_level"]
                )
                
                # Calculate increment per chunk correctly using largest_divisor
                projected_chunk = math.ceil(
                    (required_alpha - alpha) / (results["chunk_size"] / self.largest_divisor)
                ) if results["chunk_size"] and self.largest_divisor else None
                
                results["projected_chunks"].append(projected_chunk)


def update_kuiper_plot(ax, kuiper_statistics, kuiper_p_values, confidence_level=0.95,
                       ema_relative_change=None, ema_p_value_std_dev=None,
                       theta_E=None, theta_P=None):
    ax.clear()
    x = range(1, max(len(kuiper_statistics), len(kuiper_p_values)) + 1)
    
    # Calculate p-value threshold
    if confidence_level:
        p_value_threshold = 1 - confidence_level
    else:
        p_value_threshold = 0.05
    
    ax.set_ylim(0, 1)
    ax.spines['top'].set_linewidth(0)  # Remove the top spine to give the appearance of escaping

    # Plot Kuiper statistic and p-values
    ax.plot(x[:len(kuiper_statistics)], kuiper_statistics, marker='o', markersize=4, label="Kuiper Statistic", color='b')
    ax.plot(x[:len(kuiper_p_values)], kuiper_p_values, marker='x', markersize=4, label="Kuiper P-Value", color='g')
    ax.axhline(p_value_threshold, color='g', linestyle='--', label=f"P-Value Threshold ({p_value_threshold:.4f})")

    # Plot EMA values (solid lines)
    if ema_relative_change:
        ax.plot(range(1, len(ema_relative_change) + 1), ema_relative_change, label="EMA Relative Change", color='orange', linestyle='-')
    if ema_p_value_std_dev:
        ax.plot(range(1, len(ema_p_value_std_dev) + 1), ema_p_value_std_dev, label="EMA P-Value Std Dev", color='red', linestyle='-')

    # Plot Theta values (dashed lines)
    if theta_E:
        ax.plot(range(1, len(theta_E) + 1), theta_E, label="Theta E", color='orange', linestyle='--')
    if theta_P:
        ax.plot(range(1, len(theta_P) + 1), theta_P, label="Theta P", color='red', linestyle='--')

    ax.set_title("Kuiper Statistics, EMA, and Theta Values")
    ax.set_xlabel("Chunk Number")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)


def update_stopping_prob_plot(ax, ax2, chunks, stopping_probabilities, projected_chunks):
    ax.clear()
    ax2.clear()
    
    # Plot stopping probabilities and threshold
    _line1, = ax.plot(chunks, stopping_probabilities, marker='o', label="Stopping Probability")
    _line2 = ax.axhline(0.95, color='r', linestyle='--', label="Threshold (0.95)")

    # Plot projected chunks if available
    if any(projected_chunks):
        valid_chunks = [chunk for chunk, proj in zip(chunks, projected_chunks) if proj is not None]
        valid_projections = [proj for proj in projected_chunks if proj is not None]
        line3, = ax2.plot(valid_chunks, valid_projections, linestyle=':', color='purple', label="Projected Chunks")

    # Set titles and labels
    ax.set_title("Stopping Probabilities and Projected Chunks")
    ax.set_xlabel("Chunk Number")
    ax.set_ylabel("Stopping Probability")
    ax2.set_ylabel("Projected Chunks", color='purple')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    if any(projected_chunks):
        proj_handles, proj_labels = ax2.get_legend_handles_labels()
        handles += proj_handles
        labels += proj_labels
    ax.legend(handles, labels, loc="upper left")

    # Enable grid
    ax.grid(True)


def create_plots(data):
    fig1, ax1 = plt.subplots()

    update_kuiper_plot(ax1, 
                       data["kuiper_statistics"], 
                       data["kuiper_p_values"], 
                       data["confidence_level"],
                       data["ema_relative_change"],
                       data["ema_p_value_std_dev"],
                       data["theta_E"],
                       data["theta_P"])
    fig1.canvas.manager.set_window_title("Kuiper Statistics and Additional Metrics")
    
    fig2, ax2 = plt.subplots()
    ax2_left = ax2.twinx()

    update_stopping_prob_plot(ax2, ax2_left, data["chunks"], data["stopping_probabilities"], data["projected_chunks"])
    fig2.canvas.manager.set_window_title("Stopping Probabilities")
    
    return fig1, ax1, fig2, ax2, ax2_left


def main():
    parser = argparse.ArgumentParser(description="Chart Kuiper test and beta distribution progress from log file.")
    parser.add_argument('logfile', type=str, help="Path to the log file to analyze.")
    parser.add_argument('--read-tee', type=int, nargs='?', const=60, 
                       help="Continuously poll the log file and update the graph every N seconds (default: 60).")
    parser.add_argument('--export-raw', type=str, 
                       help="Path to export the raw extracted values as a JSON file.")
    args = parser.parse_args()

    plt.style.use('fast')
    log_parser = LogParser()

    if args.read_tee:
        print(f"Reading log file {args.logfile} every {args.read_tee} seconds...")
        print("Press 'q' or close the windows to quit")
        
        fig1, ax1, fig2, ax2, ax2_left = create_plots(log_parser.parse_log(args.logfile))
        
        def on_close(event):
            plt.close('all')
            sys.exit(0)
            
        fig1.canvas.mpl_connect('close_event', on_close)
        fig2.canvas.mpl_connect('close_event', on_close)
        
        try:
            while plt.get_fignums():
                data = log_parser.parse_log(args.logfile)
                if data:
                    print(f"Updating graphs at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
                    update_kuiper_plot(ax1,
                                    data["kuiper_statistics"], 
                                    data["kuiper_p_values"], 
                                    data["confidence_level"],
                                    data["ema_relative_change"],
                                    data["ema_p_value_std_dev"],
                                    data["theta_E"],
                                    data["theta_P"])

                    update_stopping_prob_plot(ax2, ax2_left, data["chunks"], 
                                           data["stopping_probabilities"], 
                                           data["projected_chunks"])

                    fig1.canvas.draw()
                    fig2.canvas.draw()

                    if args.export_raw:
                        with open(args.export_raw, 'w') as outfile:
                            json.dump(data, outfile, indent=4)
                
                plt.pause(args.read_tee)
                
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
        except Exception as e:
            raise e
        finally:
            plt.close('all')
            sys.exit(0)
    
    else:
        data = log_parser.parse_log(args.logfile)
        if args.export_raw:
            with open(args.export_raw, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        
        print("Press 'q' or close the windows to quit")
        fig1, ax1, fig2, ax2, ax2_left = create_plots(data)
        
        def on_key(event):
            if event.key == 'q':
                plt.close('all')
                sys.exit(0)
                
        fig1.canvas.mpl_connect('key_press_event', on_key)
        fig2.canvas.mpl_connect('key_press_event', on_key)
        
        for fig in [fig1, fig2]:
            canvas = fig.canvas
            toolbar = NavigationToolbar2QT(canvas, canvas.parent())
            toolbar.update()
        
        plt.show(block=True)

if __name__ == '__main__':
    main()
