from contextlib import redirect_stdout
import h5py
import json
import logging
import os
import numpy as np
from scipy.special import rel_entr
import tempfile
import unittest


from compare_logits import (
    kl_divergence,
    calculate_statistics,
    check_output_file_conditions,
    process_chunks,
)


class TestKLScript(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

        self.temp_files = [
            tempfile.NamedTemporaryFile(delete=False).name for _ in range(3)
        ]
        self.temp_output, self.temp_baseline, self.temp_target = self.temp_files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_check_output_file_conditions_new_file(self):
        """
        Validates that check_output_file_conditions correctly identifies a new (non-existent) output file
        as having no existing chunk data, overall statistics, or digest centroids.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            # Run the check while the file is guaranteed to exist
            existing_chunks, overall_stats, digest = check_output_file_conditions(output_path, None, None, False)
            self.assertEqual(existing_chunks, set())
            self.assertIsNone(overall_stats)
            self.assertIsNone(digest)
        finally:
            os.remove(output_path)

    def test_check_output_file_conditions_existing_empty_file(self):
        """Test check_output_file_conditions with an existing empty output file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            # Create an empty HDF5 file
            with h5py.File(output_path, 'w') as f_out:
                pass

            existing_chunks, overall_stats, digest = check_output_file_conditions(output_path, None, None, False)
            self.assertEqual(existing_chunks, set())
            self.assertIsNone(overall_stats)
            self.assertIsNone(digest)
        finally:
            os.remove(output_path)

    def test_check_output_file_conditions_existing_with_chunks(self):
        """Test check_output_file_conditions raises an error for file with both chunk data and overall stats."""
        with h5py.File(self.temp_output, 'w') as f_out:
            # Create chunk data
            chunk_group = f_out.create_group("chunk_0")
            chunk_group.attrs["ChunkNumber"] = 0

            # Add overall stats to simulate a complete file
            overall_stats = {
                "Average": 0.5, "StdDev": 0.1, "Minimum": 0.0, "Maximum": 1.0,
                "KLD_99": 0.99, "KLD_95": 0.95, "KLD_90": 0.9,
                "KLD_10": 0.1, "KLD_05": 0.05, "KLD_01": 0.01
            }
            f_out.attrs["overall"] = json.dumps(overall_stats)  # Serialize overall stats as JSON

        # Expecting ValueError due to both chunk data and overall property
        with self.assertRaises(ValueError):
            check_output_file_conditions(self.temp_output, None, None, False)

    def test_kl_divergence(self):
        """Test kl_divergence function with known distributions."""
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        expected_kl = np.sum(rel_entr(p, q))
        computed_kl = kl_divergence(p, q)
        self.assertAlmostEqual(computed_kl, expected_kl, places=6)

    def test_calculate_statistics(self):
        """Test calculate_statistics function with synthetic KL-divergence values."""
        kl_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        chunk_index = 0
        stats = calculate_statistics(kl_values, chunk_index)
        expected_stats = {
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
        for key in expected_stats:
            self.assertAlmostEqual(stats[key], expected_stats[key], places=6)

    def test_process_chunks(self):
        """Test process_chunks function with synthetic data."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_baseline, \
             tempfile.NamedTemporaryFile(delete=False) as temp_target, \
             tempfile.NamedTemporaryFile(delete=False) as temp_output:

            baseline_path = temp_baseline.name
            target_path = temp_target.name
            output_path = temp_output.name

            # Create synthetic logits data
            num_chunks = 3
            chunk_size = 5
            num_classes = 4

            # Generate random logits
            with h5py.File(baseline_path, 'w') as f_baseline:
                logits_baseline = np.random.rand(num_chunks, chunk_size, num_classes)
                f_baseline.create_dataset('logits', data=logits_baseline)

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)

            # Run the process_chunks function
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    from_chunk=None,
                    to_chunk=None,
                    clobber=False
                )

            # Verify that the output file has the expected data
            with h5py.File(output_path, 'r') as f_out:
                for chunk_idx in range(num_chunks):
                    self.assertIn(f'chunk_{chunk_idx}', f_out)
                    chunk_group = f_out[f'chunk_{chunk_idx}']
                    attrs = chunk_group.attrs
                    # Check if all expected attributes are present
                    for attr in ["Average", "StdDev", "Median", "Minimum", "Maximum",
                                 "KLD_99", "KLD_95", "KLD_90", "KLD_10", "KLD_05", "KLD_01"]:
                        self.assertIn(attr, attrs)
                # Check overall statistics
                self.assertIn('overall', f_out.attrs)

            # Clean up temporary files
            os.remove(baseline_path)
            os.remove(target_path)
            os.remove(output_path)

    def test_process_chunks_with_from_to(self):
        """Test process_chunks with specific from_chunk and to_chunk parameters."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_baseline, \
             tempfile.NamedTemporaryFile(delete=False) as temp_target, \
             tempfile.NamedTemporaryFile(delete=False) as temp_output:

            baseline_path = temp_baseline.name
            target_path = temp_target.name
            output_path = temp_output.name

            num_chunks = 5
            chunk_size = 10
            num_classes = 3

            with h5py.File(baseline_path, 'w') as f_baseline:
                logits_baseline = np.random.rand(num_chunks, chunk_size, num_classes)
                f_baseline.create_dataset('logits', data=logits_baseline)

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)

            # Process chunks from index 1 to 3
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    from_chunk=1,
                    to_chunk=3,
                    clobber=False
                )

            # Verify that only chunks 1 and 2 are processed
            with h5py.File(output_path, 'r') as f_out:
                self.assertNotIn('chunk_0', f_out)
                self.assertIn('chunk_1', f_out)
                self.assertIn('chunk_2', f_out)
                self.assertNotIn('chunk_3', f_out)

            # Clean up temporary files
            os.remove(baseline_path)
            os.remove(target_path)
            os.remove(output_path)

    def test_process_chunks_with_clobber(self):
        """Test process_chunks with clobber option."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_baseline, \
             tempfile.NamedTemporaryFile(delete=False) as temp_target, \
             tempfile.NamedTemporaryFile(delete=False) as temp_output:

            baseline_path = temp_baseline.name
            target_path = temp_target.name
            output_path = temp_output.name

            num_chunks = 2
            chunk_size = 5
            num_classes = 3

            # Initial run without clobber
            with h5py.File(baseline_path, 'w') as f_baseline:
                logits_baseline = np.random.rand(num_chunks, chunk_size, num_classes)
                f_baseline.create_dataset('logits', data=logits_baseline)

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)

            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    clobber=False
                )

                # Second run with clobber=True
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    clobber=True
                )

            # Verify that the output file has been overwritten
            with h5py.File(output_path, 'r') as f_out:
                for chunk_idx in range(num_chunks):
                    self.assertIn(f'chunk_{chunk_idx}', f_out)

            # Clean up temporary files
            os.remove(baseline_path)
            os.remove(target_path)
            os.remove(output_path)

    def test_resume_behavior_only_overall_property(self):
        """Test handling of file with only 'overall' property but no chunks."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.attrs["overall"] = json.dumps({"Average": 0.5})
            with self.assertRaises(ValueError):
                check_output_file_conditions(output_path, None, None, False)
        finally:
            os.remove(output_path)

    def test_resume_behavior_overall_and_chunks_present(self):
        """Test handling of file with both 'overall' property and chunk data."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_group("chunk_0")
                f_out.attrs["overall"] = json.dumps({"Average": 0.5})
            with self.assertRaises(ValueError):
                check_output_file_conditions(output_path, None, None, False)
        finally:
            os.remove(output_path)

    def test_resume_behavior_chunks_present_no_overall(self):
        """Test handling of file with chunks but no 'overall' property."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_group("chunk_0")
            with self.assertRaises(ValueError):
                check_output_file_conditions(output_path, None, None, False)
        finally:
            os.remove(output_path)

    def test_resume_behavior_clobber_option(self):
        """Test that clobber option removes all existing data, including 'overall'."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_group("chunk_0")
                f_out.attrs["overall"] = json.dumps({"Average": 0.5})
            # Ensure clobber clears the file
            existing_chunks, overall_stats, digest = check_output_file_conditions(output_path, None, None, True)
            self.assertEqual(existing_chunks, set())
            self.assertIsNone(overall_stats)
            self.assertIsNone(digest)
        finally:
            os.remove(output_path)

    def test_resume_behavior_non_contiguous_chunks(self):
        """Test for non-contiguous chunks with 'from_chunk' specified."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_group("chunk_0")
                f_out.create_group("chunk_2")  # Missing chunk_1 to simulate non-contiguity
            with self.assertRaises(ValueError):
                check_output_file_conditions(output_path, from_chunk=2, to_chunk=None, clobber=False)
        finally:
            os.remove(output_path)

    def test_resume_behavior_with_to_chunk_specified(self):
        """Test that 'to_chunk' does not overwrite existing chunks without clobber."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name
        try:
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_group("chunk_0")
                f_out.create_group("chunk_1")
            with self.assertRaises(ValueError):
                check_output_file_conditions(output_path, from_chunk=0, to_chunk=2, clobber=False)
        finally:
            os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
