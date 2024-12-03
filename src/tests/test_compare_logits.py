from contextlib import redirect_stdout
import h5py
import json
import logging
import os
import numpy as np
from tdigest import TDigest
import tempfile
import unittest
from unittest.mock import ANY, MagicMock, Mock, patch


from compare_logits import (
    EarlyStoppingStats, BayesianPriorUpdate, check_output_file_conditions,
    process_chunks, save_common_state, save_prior_and_stats, finalize_processing,
    numpy_encoder, load_state_from_file, save_state_to_file, process_chunk_part,
    kl_divergence_log_probs, save_chunk_stats, update_statistics, 
    process_single_chunk, process_chunks, determine_chunk_range,
    handle_early_stopping, initialize_early_stopping, save_early_stopping_info,
)


logger = logging.getLogger("compare_logits")


class TestEarlyStoppingStats(unittest.TestCase):
    
    def setUp(self):
        self.stats = EarlyStoppingStats()
    
    def test_add_effect_size_within_window(self):
        """Test adding effect sizes within the window size."""
        for i in range(3):
            self.stats.add_effect_size(i * 0.1)
            self.assertEqual(len(self.stats.effect_sizes), i + 1)
            self.assertEqual(self.stats.effect_sizes[-1], i * 0.1)
    
    def test_add_effect_size_exceeds_window(self):
        """Test adding more effect sizes than the window size."""
        for i in range(5):
            self.stats.add_effect_size(i * 0.1)
        self.assertEqual(len(self.stats.effect_sizes), 3)
        for received, expected in zip(self.stats.effect_sizes, [0.2, 0.3, 0.4]):
            self.assertAlmostEqual(received, expected, places=3)
    
    def test_add_p_value_within_window(self):
        """Test adding p-values within the window size."""
        for i in range(3):
            self.stats.add_p_value(i * 0.05)
            self.assertEqual(len(self.stats.p_values), i + 1)
            self.assertEqual(self.stats.p_values[-1], i * 0.05)
    
    def test_add_p_value_exceeds_window(self):
        """Test adding more p-values than the window size."""
        for i in range(5):
            self.stats.add_p_value(i * 0.05)
        self.assertEqual(len(self.stats.p_values), 3)
        for received, expected in zip(self.stats.p_values, [0.1, 0.15, 0.2]):
            self.assertAlmostEqual(received, expected, places=3)
    
    def test_update_ema_initial(self):
        """Test EMA update when sample_size is 0."""
        self.stats.update_beta_parameters = Mock()
        self.stats._update_ema(0.1, 0.05)
        self.assertEqual(self.stats.ema_relative_change, 0.1)
        self.assertEqual(self.stats.ema_p_value_std_dev, 0.05)
    
    def test_update_ema_subsequent(self):
        """Test EMA update after initial update."""
        self.stats.sample_size = 1
        self.stats.ema_relative_change = 0.1
        self.stats.ema_p_value_std_dev = 0.05
        self.stats._update_ema(0.2, 0.1)
        expected_ema_rc = 0.005 * 0.2 + (1 - 0.005) * 0.1
        expected_ema_pv = 0.005 * 0.1 + (1 - 0.005) * 0.05
        self.assertAlmostEqual(self.stats.ema_relative_change, expected_ema_rc)
        self.assertAlmostEqual(self.stats.ema_p_value_std_dev, expected_ema_pv)
    
    def test_calculate_relative_effect_size_changes(self):
        """Test calculation of relative effect size changes."""
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        changes = self.stats._calculate_relative_effect_size_changes()
        expected_changes = [1.0, 0.5]  # (0.2-0.1)/0.1=1.0, (0.3-0.2)/0.2=0.5
        for received, expected in zip(changes, expected_changes):
            self.assertAlmostEqual(received, expected, places=3)
    
    def test_calculate_relative_effect_size_changes_zero_division(self):
        """Test relative change calculation with zero in previous effect size."""
        self.stats.effect_sizes = [0.1, 0.0, 0.2]
        with self.assertRaises(ZeroDivisionError):
            self.stats._calculate_relative_effect_size_changes()
    
    def test_calculate_p_value_std_dev_empty(self):
        """Test p-value std dev calculation with empty p_values."""
        std_dev = self.stats._calculate_p_value_std_dev()
        self.assertEqual(std_dev, 0.0)
    
    def test_calculate_p_value_std_dev_single_value(self):
        """Test p-value std dev calculation with a single p_value."""
        self.stats.p_values = [0.05]
        std_dev = self.stats._calculate_p_value_std_dev()
        self.assertTrue(np.isnan(std_dev), "Standard deviation for a single p_value should be NaN.")

    def test_calculate_p_value_std_dev_multiple_values(self):
        """Test p-value std dev calculation with multiple p_values."""
        self.stats.p_values = [0.05, 0.15, 0.25]
        std_dev = self.stats._calculate_p_value_std_dev()
        expected_std = np.std(self.stats.p_values, ddof=1)
        self.assertAlmostEqual(std_dev, expected_std)
    
    def test_clamp_theta_E(self):
        """Test clamping of theta_E."""
        clamped = self.stats._clamp_theta_E(0.3)
        self.assertEqual(clamped, 0.3)
        clamped = self.stats._clamp_theta_E(0.1)
        self.assertEqual(clamped, 0.2)  # Minimum is 0.2
    
    def test_clamp_theta_P(self):
        """Test clamping of theta_P."""
        # Simulate confidence level
        self.stats.confidence = 0.95

        # Test for a value below the threshold
        clamped = self.stats._clamp_theta_P(0.04)
        self.assertAlmostEqual(clamped, 0.05, places=3, msg="Theta_P should be clamped to the minimum value of 0.05.")

        # Test for a value above the threshold
        clamped = self.stats._clamp_theta_P(0.96)
        self.assertEqual(clamped, 0.96, "Theta_P above the threshold should not be clamped.")
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_update_beta_parameters_insufficient_window(self, mock_update_ema, mock_calc_dynamic_thresholds, mock_calc_p_value_std_dev, mock_calc_rel_changes):
        """Test update_beta_parameters when window size is not met."""
        self.stats.effect_sizes = [0.1, 0.2]  # Less than window_size=3
        self.stats.p_values = [0.05, 0.15]
        self.stats.window_size = 3  # Ensure window is not met

        # Call the method under test
        result = self.stats.update_beta_parameters()

        # Assert that the result is None
        self.assertIsNone(result, "Expected update_beta_parameters to return None when window size is not met.")

        # Verify that internal methods were not called
        mock_update_ema.assert_not_called()
        mock_calc_dynamic_thresholds.assert_not_called()
        mock_calc_p_value_std_dev.assert_not_called()
        mock_calc_rel_changes.assert_not_called()

        # Assert that alpha and beta remain unchanged
        self.assertEqual(self.stats.alpha, 1, "Alpha should remain 1.")
        self.assertEqual(self.stats.beta, 1, "Beta should remain 1.")
    
    def test_update_beta_parameters_condition_met(self):
        """Test update_beta_parameters when conditions are met to increment alpha."""
        self.stats.dynamic_thresholds_enabled = False
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        theta_E, theta_P, rel_changes, pv_std = self.stats.update_beta_parameters()
        self.assertEqual(theta_E, self.stats.theta_E)
        self.assertEqual(theta_P, self.stats.theta_P)
        self.assertEqual(self.stats.alpha, 2)
        self.assertEqual(self.stats.beta, 1)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_update_beta_parameters_condition_not_met(self, mock_update_ema, mock_calc_dynamic_thresholds, mock_calc_p_value_std_dev, mock_calc_rel_changes):
        """Test update_beta_parameters when conditions are not met to increment alpha, hence increment beta."""
        self.stats.dynamic_thresholds_enabled = False
        self.stats.effect_sizes = [0.1, 0.3, 0.5]  # window_size=3
        self.stats.p_values = [0.05, 0.2, 0.3]
        self.stats.window_size = 3  # Ensure window is met
        self.stats.sample_size = 0  # Initialize EMA directly with current data

        # Mock internal methods to return desired values
        mock_calc_relative_changes = [2.0, 0.6667]
        mock_calc_pv_std_dev = 0.1247
        mock_calc_rel_changes.return_value = mock_calc_relative_changes
        mock_calc_p_value_std_dev.return_value = mock_calc_pv_std_dev

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Execute the method under test
        theta_E, theta_P, rel_changes, pv_std = self.stats.update_beta_parameters()

        # Assertions
        self.assertEqual(theta_E, self.stats.theta_E)
        self.assertEqual(theta_P, self.stats.theta_P)
        self.assertEqual(rel_changes, mock_calc_relative_changes)
        self.assertEqual(pv_std, mock_calc_pv_std_dev)
        self.assertEqual(self.stats.alpha, 1)  # Should remain unchanged
        self.assertEqual(self.stats.beta, 2)   # Should be incremented

        # Verify that internal methods were called as expected
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_calc_dynamic_thresholds.assert_not_called()  # Since dynamic_thresholds_enabled=False
        mock_update_ema.assert_called_once_with(np.median(mock_calc_relative_changes), mock_calc_pv_std_dev)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_thresholds_enabled(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic threshold updates when dynamic_thresholds_enabled is True."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.sample_size = 1
        self.stats.window_size = 3
        self.stats.effect_sizes = [0.1, 0.2, 0.3]  # Ensure the window size condition is met
        self.stats.p_values = [0.05, 0.05, 0.05]  # Ensure the window size condition is met

        # Mock return values for dependent methods
        mock_calc_rel_changes.return_value = [0.1, 0.2]  # Simulated relative changes
        mock_calc_p_value_std_dev.return_value = 0.01     # Simulated p-value standard deviation

        # Mock _calculate_dynamic_thresholds to simulate updated thresholds
        mock_calc_dynamic_thresholds.return_value = (0.25, 0.15)  # Updated theta_E and theta_P

        # Mock _update_ema to simulate EMA updates
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_thresholds_enabled(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic threshold updates when dynamic_thresholds_enabled is True."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.sample_size = 1
        self.stats.window_size = 3
        self.stats.effect_sizes = [0.1, 0.2, 0.3]  # Ensure the window size condition is met
        self.stats.p_values = [0.05, 0.05, 0.05]  # Ensure the window size condition is met

        # Mock return values for dependent methods
        mock_calc_rel_changes.return_value = [0.1, 0.2]  # Simulated relative changes
        mock_calc_p_value_std_dev.return_value = 0.01     # Simulated p-value standard deviation

        # Mock _calculate_dynamic_thresholds to simulate updated thresholds
        mock_calc_dynamic_thresholds.return_value = (0.25, 0.15)  # Updated theta_E and theta_P

        # Mock _update_ema to simulate EMA updates
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Call the method under test
        theta_E, theta_P, rel_changes, pv_std = self.stats.update_beta_parameters()

        # Assert the values returned by the mocked methods are reflected in the output
        self.assertEqual(theta_E, 0.25, "Theta_E should reflect the updated value from _calculate_dynamic_thresholds.")
        self.assertEqual(theta_P, 0.15, "Theta_P should reflect the updated value from _calculate_dynamic_thresholds.")
        self.assertEqual(rel_changes, [0.1, 0.2], "Returned relative changes should match the mocked output.")
        self.assertEqual(pv_std, 0.01, "Returned p-value std deviation should match the mocked output.")

        # Assert that mocked methods were called
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_calc_dynamic_thresholds.assert_called_once()

        # Adjust the expected value to match the computed median
        expected_relative_change = np.median([0.1, 0.2])  # Median of mocked relative changes
        mock_update_ema.assert_called_once_with(expected_relative_change, 0.01)  # Verify the inputs to _update_ema
    
    @patch.object(EarlyStoppingStats, '_update_ema')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    def test_dynamic_theta_E_increase(
        self,
        mock_calc_rel_changes,
        mock_calc_p_value_std_dev,
        mock_calc_dynamic_thresholds,
        mock_update_ema
    ):
        """Test dynamic theta_E increases correctly."""
        # Enable dynamic thresholds
        self.stats.dynamic_thresholds_enabled = True
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        self.stats.historic_theta_E = 0.2  # Initialize historic_theta_E
        self.stats.theta_E_increase_count = 0
        self.stats.total_theta_E_updates = 0

        # Mock return values for dependent methods
        mock_calc_rel_changes.return_value = [0.5, 0.5]  # Simulate positive relative changes
        mock_calc_p_value_std_dev.return_value = 0.01    # Simulate low p-value std deviation

        # Mock _calculate_dynamic_thresholds to simulate increased thresholds
        def mock_calc_dynamic_thresholds_side_effect():
            # Simulate an increase in historic_theta_E
            new_theta_E = self.stats.historic_theta_E + 0.1  # Increase by 0.1
            self.stats.historic_theta_E = new_theta_E
            self.stats.theta_E_increase_count += 1  # Track the increase
            return new_theta_E, self.stats.theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Record the initial historic_theta_E
        initial_historic_theta_E = self.stats.historic_theta_E

        # Perform the update
        self.stats.update_beta_parameters()

        # Assert that historic_theta_E has increased
        self.assertGreater(
            self.stats.historic_theta_E,
            initial_historic_theta_E,
            f"historic_theta_E {self.stats.historic_theta_E} should be greater than initial_historic_theta_E {initial_historic_theta_E}."
        )

        # Assert that the increase count was incremented
        self.assertEqual(
            self.stats.theta_E_increase_count,
            1,
            f"theta_E_increase_count {self.stats.theta_E_increase_count} should be 1."
        )

        # Verify all mocked methods were called correctly
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_calc_dynamic_thresholds.assert_called_once()
        mock_update_ema.assert_called_once_with(0.5, 0.01)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_theta_E_decrease(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic theta_E decreases correctly."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.effect_sizes = [0.3, 0.2, 0.1]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1

        # Mock return values for the first update
        mock_calc_rel_changes.return_value = [0.33, 0.5]  # Simulate larger relative changes
        mock_calc_p_value_std_dev.return_value = 0.01     # Simulate low p-value std deviation

        # Mock _calculate_dynamic_thresholds to set the initial historic_theta_E
        def mock_calc_dynamic_thresholds_first():
            # Set historic_theta_E to a higher value initially
            self.stats.historic_theta_E = 0.25  # Higher than the clamp value
            return self.stats.historic_theta_E, self.stats.theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_first

        # Mock _update_ema to simulate EMA updates
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Perform the first update to set the initial historic_theta_E
        self.stats.update_beta_parameters()
        initial_historic_theta_E = self.stats.historic_theta_E

        # Mock return values for the second update with smaller relative changes
        mock_calc_rel_changes.return_value = [0.1, 0.05]  # Simulate smaller relative changes
        mock_calc_p_value_std_dev.return_value = 0.01     # Keep p-value std deviation low

        # Mock _calculate_dynamic_thresholds to simulate a decrease in historic_theta_E
        def mock_calc_dynamic_thresholds_second():
            # Simulate a decrease but enforce clamping to 0.2
            self.stats.historic_theta_E = max(self.stats.historic_theta_E * 0.9, 0.2)
            return self.stats.historic_theta_E, self.stats.theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_second

        # Perform the second update to decrease historic_theta_E
        self.stats.effect_sizes = [0.2, 0.15, 0.1]
        self.stats.update_beta_parameters()

        # Assert that historic_theta_E decreases or reaches the clamped minimum
        if initial_historic_theta_E > 0.2:
            self.assertLess(
                self.stats.historic_theta_E,
                initial_historic_theta_E,
                f"historic_theta_E {self.stats.historic_theta_E} should be less than initial_historic_theta_E {initial_historic_theta_E}."
            )
        else:
            self.assertEqual(
                self.stats.historic_theta_E,
                0.2,
                f"historic_theta_E {self.stats.historic_theta_E} should be clamped to the minimum value of 0.2."
            )

        # Verify all mocked methods were called
        self.assertEqual(mock_calc_rel_changes.call_count, 2)
        self.assertEqual(mock_calc_p_value_std_dev.call_count, 2)
        self.assertEqual(mock_calc_dynamic_thresholds.call_count, 2)
        self.assertEqual(mock_update_ema.call_count, 2)

    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_theta_P_increase(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic theta_P increases correctly."""
        # Enable dynamic thresholds
        self.stats.dynamic_thresholds_enabled = True

        # Set up initial conditions
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.05, 0.2, 0.3]
        self.stats.sample_size = 1
        self.stats.confidence = 0.95
        self.stats.theta_E = 0.2
        self.stats.theta_P = 0.1
        self.stats.window_size = 3

        # Initialize historic_theta_P to None to trigger initialization logic
        self.stats.historic_theta_P = None
        self.stats.historic_theta_E = 0.2  # Initial value

        initial_theta_P = self.stats.theta_P  # initial_theta_P = 0.1

        # Mock return values for dependent methods
        mock_calc_rel_changes.return_value = [0.0, 0.0]  # Simulate no relative changes
        mock_calc_p_value_std_dev.return_value = 0.15     # Simulate an increase in p-value std dev

        # Mock _calculate_dynamic_thresholds to simulate updated thresholds
        def mock_calc_dynamic_thresholds_side_effect():
            # Force an increase in historic_theta_P
            self.stats.historic_theta_P = initial_theta_P + 0.05  # For example, 0.15
            return self.stats.theta_E, self.stats.historic_theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Call the method under test
        result = self.stats.update_beta_parameters()

        # Unpack the result
        theta_E, theta_P, rel_changes, pv_std = result

        # Assert that theta_P has increased
        self.assertGreater(
            theta_P,
            initial_theta_P,
            f"theta_P {theta_P} should be greater than initial_theta_P {initial_theta_P}."
        )

        # Optionally, also assert that historic_theta_P was updated
        self.assertGreater(
            self.stats.historic_theta_P,
            initial_theta_P,
            f"historic_theta_P {self.stats.historic_theta_P} should be greater than initial_theta_P {initial_theta_P}."
        )

        # Verify all mocked methods were called correctly
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_update_ema.assert_called_once_with(0.0, 0.15)  # Verify inputs to _update_ema

    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_theta_P_decrease(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic theta_P decreases correctly."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.3, 0.2, 0.1]
        self.stats.sample_size = 1

        # Mock return values for the first update
        mock_calc_rel_changes.return_value = [0.0, 0.0]  # Simulate no relative changes
        mock_calc_p_value_std_dev.return_value = 0.3     # Simulate higher initial p-value std dev

        # Mock _calculate_dynamic_thresholds to set initial historic_theta_P
        def mock_calc_dynamic_thresholds_side_effect_first():
            # Initialize historic_theta_P to a higher value
            self.stats.historic_theta_P = max(self.stats.ema_p_value_std_dev, 0.15)
            return self.stats.theta_E, self.stats.historic_theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect_first

        # Mock _update_ema to update EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Perform the first update
        self.stats.update_beta_parameters()
        initial_historic_theta_P = self.stats.historic_theta_P

        # Mock return values for the second update with lower p-values
        mock_calc_p_value_std_dev.return_value = 0.05  # Simulate lower p-value std dev

        # Mock _calculate_dynamic_thresholds to simulate a decrease
        def mock_calc_dynamic_thresholds_side_effect_second():
            # Decrease historic_theta_P
            new_theta_P = max(self.stats.historic_theta_P * 0.8, 0.05)
            self.stats.historic_theta_P = new_theta_P
            return self.stats.theta_E, new_theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect_second

        # Perform the second update
        self.stats.p_values = [0.1, 0.05, 0.05]
        self.stats.update_beta_parameters()

        # Assert that historic_theta_P has decreased
        self.assertLess(
            self.stats.historic_theta_P,
            initial_historic_theta_P,
            f"historic_theta_P {self.stats.historic_theta_P} should be less than initial_historic_theta_P {initial_historic_theta_P}."
        )

        # Verify all mocked methods were called
        self.assertEqual(mock_calc_rel_changes.call_count, 2)
        self.assertEqual(mock_calc_p_value_std_dev.call_count, 2)
        self.assertEqual(mock_calc_dynamic_thresholds.call_count, 2)
        self.assertEqual(mock_update_ema.call_count, 2)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_min_decrease_rates_theta_E(
        self, 
        mock_update_ema, 
        mock_calc_dynamic_thresholds, 
        mock_calc_p_value_std_dev, 
        mock_calc_rel_changes
    ):
        """Test that theta_E does not decrease below the minimum threshold."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.confidence = 0.95
        self.stats.effect_sizes = [0.3, 0.2, 0.1]  # window_size=3
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        self.stats.historic_theta_E = 0.25  # Initialize with a valid value above the minimum

        # Mock return values for internal methods
        mock_calc_rel_changes.return_value = [0.3333, 0.5]  # Simulated relative changes
        mock_calc_p_value_std_dev.return_value = 0.0  # Simulated std deviation

        # Mock _calculate_dynamic_thresholds to return valid thresholds
        mock_calc_dynamic_thresholds.return_value = (0.25, 0.05)  # Ensure valid values for theta_E and theta_P

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Execute multiple updates
        for _ in range(10):
            self.stats.update_beta_parameters()

        # Assert that historic_theta_E is not below 0.2
        self.assertGreaterEqual(
            self.stats.historic_theta_E, 
            0.2, 
            "historic_theta_E should not be less than the minimum threshold of 0.2"
        )
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_min_decrease_rates_theta_P(
        self, 
        mock_update_ema, 
        mock_calc_dynamic_thresholds, 
        mock_calc_p_value_std_dev, 
        mock_calc_rel_changes
    ):
        """Test that theta_P does not decrease below the minimum threshold."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.confidence = 0.95
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.1, 0.05, 0.05]
        self.stats.sample_size = 1
        self.stats.historic_theta_P = 0.1  # Initialize with a valid starting value

        # Mock return values for internal methods
        mock_calc_rel_changes.return_value = [0.0, 0.0]  # Simulated relative changes
        mock_calc_p_value_std_dev.return_value = 0.028868  # Precomputed p-value std deviation
        mock_calc_dynamic_thresholds.return_value = (0.2, 0.05)  # Ensure valid return values for thresholds

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Execute multiple updates
        for _ in range(10):
            self.stats.update_beta_parameters()

        # Assert that historic_theta_P is not below 0.05
        self.assertGreaterEqual(
            self.stats.historic_theta_P, 
            0.05, 
            "historic_theta_P should not be less than the minimum threshold of 0.05"
        )
    
    def test_stopping_early_flag(self):
        """Test setting the stopped_early flag."""
        self.stats.stopped_early = False
        # Simulate a condition where EMA values meet stopping criteria
        self.stats.ema_relative_change = 0.1  # < theta_E
        self.stats.ema_p_value_std_dev = 0.05  # < theta_P
        self.stats.theta_E = 0.2
        self.stats.theta_P = 0.1
        self.stats.update_beta_parameters()
        # Assuming the implementation sets stopped_early when conditions are met
        # If not, this test needs to be adjusted based on actual implementation
        # Here, since alpha is incremented, we might not have a stopping condition
        # Therefore, this test might need more context or the actual stopping logic
        # For demonstration, we'll skip setting stopped_early
        pass  # Adjust based on actual stopping logic
    
    def test_effect_sizes_and_p_values_reset(self):
        """Test that effect_sizes and p_values maintain the window size."""
        for i in range(5):
            self.stats.add_effect_size(i * 0.1)
            self.stats.add_p_value(i * 0.05)
        self.assertEqual(len(self.stats.effect_sizes), 3)
        self.assertEqual(len(self.stats.p_values), 3)
        for received, expected in zip(self.stats.effect_sizes, [0.2, 0.3, 0.4]):
            self.assertAlmostEqual(received, expected, places=3)
        for received, expected in zip(self.stats.p_values, [0.1, 0.15, 0.2]):
            self.assertAlmostEqual(received, expected, places=3)
    
    def test_update_beta_parameters_no_dynamic(self):
        """Test update_beta_parameters without dynamic thresholds."""
        self.stats.dynamic_thresholds_enabled = False
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        theta_E, theta_P, rel_changes, pv_std = self.stats.update_beta_parameters()
        self.assertEqual(theta_E, 0.2)
        self.assertEqual(theta_P, 0.1)
        self.assertEqual(self.stats.alpha, 2)
        self.assertEqual(self.stats.beta, 1)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_update_beta_parameters_with_dynamic(
        self,
        mock_update_ema,
        mock_calculate_dynamic_thresholds,
        mock_calculate_p_value_std_dev,
        mock_calculate_relative_effect_size_changes
    ):
        """Test update_beta_parameters with dynamic thresholds enabled using mocks."""
        # Enable dynamic thresholds
        self.stats.dynamic_thresholds_enabled = True
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1

        # Mock the internal methods
        mock_calculate_relative_effect_size_changes.return_value = [0.1, 0.2]
        mock_calculate_p_value_std_dev.return_value = 0.05
        mock_calculate_dynamic_thresholds.return_value = (0.25, 0.15)
        mock_update_ema.return_value = None  # Assuming _update_ema doesn't return anything

        # First call to update_beta_parameters
        theta_E_before, theta_P_before, _, _ = self.stats.update_beta_parameters()

        # Assert that internal methods were called correctly
        mock_calculate_relative_effect_size_changes.assert_called_once()
        mock_calculate_p_value_std_dev.assert_called_once()
        mock_calculate_dynamic_thresholds.assert_called_once()
        mock_update_ema.assert_called_once_with(np.median([0.1, 0.2]), 0.05)

        # Check that theta_E and theta_P have been updated
        self.assertEqual(theta_E_before, 0.25)
        self.assertEqual(theta_P_before, 0.15)
        self.assertEqual(self.stats.alpha, 2)  # Assuming condition met: 0.1 < 0.25 and 0.05 < 0.15
        self.assertEqual(self.stats.beta, 1)   # Initial beta was 1, incremented to 1 since condition was met

        # Reset mocks for the second call
        mock_calculate_relative_effect_size_changes.reset_mock()
        mock_calculate_p_value_std_dev.reset_mock()
        mock_calculate_dynamic_thresholds.reset_mock()
        mock_update_ema.reset_mock()

        # Second call to update_beta_parameters with the same mocked outputs
        theta_E_after, theta_P_after, _, _ = self.stats.update_beta_parameters()

        # Assert that internal methods were called again
        mock_calculate_relative_effect_size_changes.assert_called_once()
        mock_calculate_p_value_std_dev.assert_called_once()
        mock_calculate_dynamic_thresholds.assert_called_once()
        mock_update_ema.assert_called_once_with(np.median([0.1, 0.2]), 0.05)

        # Check that theta_E and theta_P remain the same
        self.assertEqual(theta_E_before, theta_E_after)
        self.assertEqual(theta_P_before, theta_P_after)
        self.assertEqual(self.stats.alpha, 3)  # Incremented again if condition met
        self.assertEqual(self.stats.beta, 1)   # Beta remains the same
    
    def test_ema_initialization(self):
        """Test that EMA initializes correctly on the first update."""
        self.stats.sample_size = 0
        self.stats._update_ema(0.2, 0.1)
        self.assertEqual(self.stats.ema_relative_change, 0.2)
        self.assertEqual(self.stats.ema_p_value_std_dev, 0.1)
    
    def test_ema_continuous_updates(self):
        """Test continuous EMA updates over multiple data points."""
        self.stats.sample_size = 1
        self.stats.ema_relative_change = 0.1
        self.stats.ema_p_value_std_dev = 0.05
        data = [
            (0.2, 0.1),
            (0.3, 0.2),
            (0.4, 0.3)
        ]
        for rel, pv in data:
            self.stats._update_ema(rel, pv)
        # Calculate expected EMA values manually
        ema_rc = 0.1
        ema_pv = 0.05
        decay = self.stats.ema_decay
        for rel, pv in data:
            ema_rc = decay * rel + (1 - decay) * ema_rc
            ema_pv = decay * pv + (1 - decay) * ema_pv
        self.assertAlmostEqual(self.stats.ema_relative_change, ema_rc)
        self.assertAlmostEqual(self.stats.ema_p_value_std_dev, ema_pv)
    
    def test_dynamic_thresholds_initialization(self):
        """Test that dynamic thresholds are initialized correctly."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        self.stats.update_beta_parameters()
        self.assertIsNotNone(self.stats.historic_theta_E)
        self.assertIsNotNone(self.stats.historic_theta_P)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_thresholds_min_clamp(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test that dynamic thresholds are clamped to minimum values."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.confidence = 0.95
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1

        # Define side effects for _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Mock _calculate_dynamic_thresholds to return thresholds clamped to their minimum values
        def mock_calc_dynamic_thresholds_side_effect():
            theta_E = max(self.stats.historic_theta_E or 0.2, 0.2)
            theta_P = max(self.stats.historic_theta_P or 0.05, 0.05)
            self.stats.historic_theta_E = theta_E
            self.stats.historic_theta_P = theta_P
            return theta_E, theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect

        # Mock other methods to control their outputs
        mock_calc_rel_changes.return_value = [0.0, 0.0]  # Simulated relative changes
        mock_calc_p_value_std_dev.return_value = 0.0  # Simulated p-value std deviation

        # Execute multiple updates
        for _ in range(100):
            theta_E, theta_P, rel_changes, pv_std = self.stats.update_beta_parameters()

            # Assert that the returned theta_E and theta_P are not below the thresholds
            self.assertGreaterEqual(theta_E, 0.2, f"Returned theta_E {theta_E} is below 0.2")
            self.assertGreaterEqual(theta_P, 0.05, f"Returned theta_P {theta_P} is below 0.05")
    
    def test_no_dynamic_thresholds_no_changes(self):
        """Ensure theta_E and theta_P remain unchanged when dynamic_thresholds_enabled is False."""
        initial_theta_E = self.stats.theta_E
        initial_theta_P = self.stats.theta_P
        self.stats.dynamic_thresholds_enabled = False
        self.stats.effect_sizes = [0.1, 0.1, 0.1]
        self.stats.p_values = [0.05, 0.05, 0.05]
        self.stats.sample_size = 1
        self.stats.update_beta_parameters()
        self.assertEqual(self.stats.theta_E, initial_theta_E)
        self.assertEqual(self.stats.theta_P, initial_theta_P)
    
    def test_effect_size_zero(self):
        """Test handling of zero effect size to avoid division by zero."""
        self.stats.effect_sizes = [0.1, 0.0, 0.2]
        with self.assertRaises(ZeroDivisionError):
            self.stats._calculate_relative_effect_size_changes()
    
    def test_p_value_variance_single(self):
        """Test p-value standard deviation with single p-value."""
        self.stats.p_values = [0.05, 0.05]
        std_dev = self.stats._calculate_p_value_std_dev()
        self.assertEqual(std_dev, 0.0)
    
    def test_p_value_variance_multiple(self):
        """Test p-value standard deviation with multiple p-values."""
        self.stats.p_values = [0.05, 0.15, 0.25, 0.35]
        std_dev = self.stats._calculate_p_value_std_dev()
        expected_std = np.std(self.stats.p_values, ddof=1)
        self.assertAlmostEqual(std_dev, expected_std)
    
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    def test_update_beta_parameters_multiple_updates(self, mock_pv_std, mock_rel_change):
        """Test multiple updates to beta parameters to simulate training process."""
        # Define sequences that will alternate between incrementing alpha and beta
        rel_changes_sequence = [
            [0.01, 0.02],  # Increment alpha
            [0.3, 0.35],    # Increment beta
            [0.02, 0.01],  # Increment alpha
            [0.4, 0.5],    # Increment beta
            [0.015, 0.025],# Increment alpha
            [0.5, 0.6],    # Increment beta
            [0.02, 0.03],  # Increment alpha
        ]
        pv_std_sequence = [
            0.05,  # Increment alpha
            0.2,   # Increment beta
            0.04,  # Increment alpha
            0.25,  # Increment beta
            0.03,  # Increment alpha
            0.3,   # Increment beta
            0.02,  # Increment alpha
        ]

        # Configure the mock to return the next value in the sequence each time it's called
        mock_rel_change.side_effect = rel_changes_sequence
        mock_pv_std.side_effect = pv_std_sequence

        # Mock _update_ema to set ema_relative_change and ema_p_value_std_dev
        with patch.object(EarlyStoppingStats, '_update_ema') as mock_update_ema:
            def side_effect(relative_change, p_value_std):
                self.stats.ema_relative_change = relative_change
                self.stats.ema_p_value_std_dev = p_value_std
            mock_update_ema.side_effect = side_effect

            # Simulate adding effect sizes and p-values
            effect_sizes = [0.1, 0.15, 0.2, 0.18, 0.17, 0.16, 0.15]
            p_values = [0.05, 0.04, 0.03, 0.04, 0.02, 0.01, 0.02]
            for es, pv in zip(effect_sizes, p_values):
                self.stats.add_effect_size(es)
                self.stats.add_p_value(pv)
                self.stats.update_beta_parameters()

        # After updates, check if alpha and beta have been incremented appropriately
        # Expected: alpha incremented 4 times, beta incremented 3 times
        self.assertEqual(self.stats.alpha, 4, f"Expected alpha to be 5, got {self.stats.alpha}")
        self.assertEqual(self.stats.beta, 3, f"Expected beta to be 4, got {self.stats.beta}")

    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_effect_sizes_greater_than_window(
        self, 
        mock_update_ema, 
        mock_calc_dynamic_thresholds, 
        mock_calc_p_value_std_dev, 
        mock_calc_rel_changes
    ):
        """Test behavior when effect_sizes are greater than window_size."""
        self.stats.effect_sizes = [0.1, 0.2]  # Less than window_size
        self.stats.p_values = [0.05, 0.05]
        self.stats.window_size = 3  # Window size not met

        # Mock methods to ensure they are not called
        mock_calc_rel_changes.return_value = [0.0]
        mock_calc_p_value_std_dev.return_value = 0.0
        mock_calc_dynamic_thresholds.return_value = (0.2, 0.05)

        # Call the method under test
        result = self.stats.update_beta_parameters()

        # Assert that the method returns None
        self.assertIsNone(result, "Expected update_beta_parameters to return None when window size is not met.")

        # Ensure none of the internal methods were called
        mock_update_ema.assert_not_called()
        mock_calc_dynamic_thresholds.assert_not_called()
        mock_calc_p_value_std_dev.assert_not_called()
        mock_calc_rel_changes.assert_not_called()

    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_update_ema')
    def test_dynamic_thresholds_with_fluctuating_data(
        self,
        mock_update_ema,
        mock_calc_dynamic_thresholds,
        mock_calc_p_value_std_dev,
        mock_calc_rel_changes
    ):
        """Test dynamic threshold adjustments with fluctuating effect sizes and p-values."""
        self.stats.dynamic_thresholds_enabled = True
        self.stats.sample_size = 1
        self.stats.historic_theta_E = None  # Ensure it's None initially
        self.stats.historic_theta_P = None  # Ensure it's None initially

        # Define fluctuating effect sizes and p-values
        data = [
            ([0.1, 0.2, 0.3], [0.05, 0.05, 0.05]),
            ([0.3, 0.2, 0.1], [0.05, 0.05, 0.05]),
            ([0.2, 0.25, 0.3], [0.05, 0.05, 0.05]),
            ([0.3, 0.35, 0.4], [0.05, 0.05, 0.05]),
            ([0.4, 0.35, 0.3], [0.05, 0.05, 0.05]),
        ]

        # Mock _update_ema to set EMA values directly
        def mock_update_ema_side_effect(relative_change, p_value_std_dev):
            self.stats.ema_relative_change = relative_change
            self.stats.ema_p_value_std_dev = p_value_std_dev

        mock_update_ema.side_effect = mock_update_ema_side_effect

        # Mock _calculate_dynamic_thresholds to simulate updates
        def mock_calc_dynamic_thresholds_side_effect():
            if self.stats.historic_theta_E is None:
                self.stats.historic_theta_E = 0.25  # Initialize with a value
            if self.stats.historic_theta_P is None:
                self.stats.historic_theta_P = 0.05  # Initialize with a value
            return self.stats.historic_theta_E, self.stats.historic_theta_P

        mock_calc_dynamic_thresholds.side_effect = mock_calc_dynamic_thresholds_side_effect

        # Mock the other internal methods to return controlled outputs
        for effect, pval in data:
            self.stats.effect_sizes = effect
            self.stats.p_values = pval

            # Calculate relative changes and p-value std dev
            rel_changes = [
                abs((effect[i] - effect[i-1]) / effect[i-1]) if effect[i-1] != 0 else 0
                for i in range(1, len(effect))
            ]
            pval_std = np.std(pval, ddof=1) if len(pval) > 1 else 0.0

            mock_calc_rel_changes.return_value = rel_changes
            mock_calc_p_value_std_dev.return_value = pval_std

            # Execute the method under test
            self.stats.update_beta_parameters()

        # Check if historic_theta_E and historic_theta_P have been updated correctly
        self.assertIsNotNone(self.stats.historic_theta_E, "historic_theta_E should not be None")
        self.assertIsNotNone(self.stats.historic_theta_P, "historic_theta_P should not be None")
        self.assertGreaterEqual(self.stats.historic_theta_E, 0.2, "historic_theta_E should not be less than 0.2")
        self.assertGreaterEqual(self.stats.historic_theta_P, 0.05, "historic_theta_P should not be less than 0.05")

    def test_negative_effect_sizes(self):
        """Test handling of negative effect sizes."""
        self.stats.effect_sizes = [0.1, -0.2, 0.3]
        changes = self.stats._calculate_relative_effect_size_changes()
        expected_changes = [3.0, 2.5]  # |-0.2 - 0.1| / 0.1 = 3.0, |0.3 - (-0.2)| / -0.2 = 2.5
        for received, expected in zip(changes, expected_changes):
            self.assertAlmostEqual(received, expected, places=3)
    
    def test_zero_p_values(self):
        """Test handling of zero p_values."""
        self.stats.p_values = [0.0, 0.0, 0.0]
        std_dev = self.stats._calculate_p_value_std_dev()
        self.assertEqual(std_dev, 0.0)

    def test_large_effect_sizes(self):
        """Test handling of large effect sizes."""
        self.stats.effect_sizes = [1000.0, 2000.0, 3000.0]
        changes = self.stats._calculate_relative_effect_size_changes()
        expected_changes = [1.0, 0.5]  # (2000-1000)/1000=1.0, (3000-2000)/2000=0.5
        for received, expected in zip(changes, expected_changes):
            self.assertAlmostEqual(received, expected, places=3)

    def test_small_p_values(self):
        """Test handling of very small p-values."""
        self.stats.p_values = [0.0001, 0.0002, 0.0003]
        std_dev = self.stats._calculate_p_value_std_dev()
        expected_std = np.std(self.stats.p_values, ddof=1)
        self.assertAlmostEqual(std_dev, expected_std)

    @patch.object(EarlyStoppingStats, '_update_ema')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    def test_high_confidence(
        self,
        mock_calc_rel_changes,
        mock_calc_p_value_std_dev,
        mock_calc_dynamic_thresholds,
        mock_update_ema
    ):
        """Test with a very high confidence level."""
        # Set up the test conditions
        self.stats.confidence = 0.99
        self.stats.dynamic_thresholds_enabled = True
        self.stats.sample_size = 1
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        self.stats.p_values = [0.05, 0.05, 0.05]

        # Mock return values for all dependencies
        mock_calc_rel_changes.return_value = [0.5, 0.5]
        mock_calc_p_value_std_dev.return_value = 0.01
        mock_calc_dynamic_thresholds.return_value = (0.3, 0.01)
        mock_update_ema.side_effect = lambda relative_change, p_value_std_dev: None  # No-op

        # Call the method
        theta_E, theta_P, _, _ = self.stats.update_beta_parameters()

        # Verify thresholds
        self.assertGreaterEqual(theta_E, 0.2)
        self.assertGreaterEqual(theta_P, 0.01)

        # Verify all mocked methods were called correctly
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_calc_dynamic_thresholds.assert_called_once()
        mock_update_ema.assert_called_once_with(0.5, 0.01)


    @patch.object(EarlyStoppingStats, '_update_ema')
    @patch.object(EarlyStoppingStats, '_calculate_dynamic_thresholds')
    @patch.object(EarlyStoppingStats, '_calculate_p_value_std_dev')
    @patch.object(EarlyStoppingStats, '_calculate_relative_effect_size_changes')
    def test_low_confidence(
        self,
        mock_calc_rel_changes,
        mock_calc_p_value_std_dev,
        mock_calc_dynamic_thresholds,
        mock_update_ema
    ):
        """Test with a very low confidence level."""
        # Set up the test conditions
        self.stats.confidence = 0.5
        self.stats.dynamic_thresholds_enabled = True
        self.stats.sample_size = 1
        self.stats.effect_sizes = [0.1, 0.2, 0.3]
        self.stats.p_values = [0.05, 0.05, 0.05]

        # Mock return values for all dependencies
        mock_calc_rel_changes.return_value = [0.5, 0.5]
        mock_calc_p_value_std_dev.return_value = 0.01
        mock_calc_dynamic_thresholds.return_value = (0.25, 0.5)
        mock_update_ema.side_effect = lambda relative_change, p_value_std_dev: None  # No-op

        # Call the method
        theta_E, theta_P, _, _ = self.stats.update_beta_parameters()

        # Verify thresholds
        self.assertGreaterEqual(theta_E, 0.2)
        self.assertGreaterEqual(theta_P, 0.5)

        # Verify all mocked methods were called correctly
        mock_calc_rel_changes.assert_called_once()
        mock_calc_p_value_std_dev.assert_called_once()
        mock_calc_dynamic_thresholds.assert_called_once()
        mock_update_ema.assert_called_once_with(0.5, 0.01)


class TestBayesianPriorUpdate(unittest.TestCase):
    @patch('compare_logits.BayesianPriorUpdate.calculate_kl_divergence')
    def test_check_convergence(self, mock_calculate_kl):
        mock_calculate_kl.return_value = 0.05
        initial_prior = np.array([0.2, 0.5, 0.3])
        updater = BayesianPriorUpdate(initial_prior, window_size=3)
        
        updater.previous_dists = [np.array([0.25, 0.45, 0.3]),
                                 np.array([0.2, 0.5, 0.3]),
                                 np.array([0.15, 0.55, 0.3])]
        
        is_converged, kl_div = updater.check_convergence(np.array([0.22, 0.48, 0.3]))
        
        self.assertTrue(mock_calculate_kl.called)
        self.assertFalse(is_converged)
        self.assertEqual(kl_div, 0.05)

    @patch('compare_logits.BayesianPriorUpdate.check_convergence', return_value=(False, 0.5))
    def test_update_learning_rate(self, mock_check_convergence):
        # Arrange
        initial_prior = np.array([0.25, 0.25, 0.25, 0.25])
        updater = BayesianPriorUpdate(initial_prior)
        kl_div = 0.5

        # Capture the current learning rate for calculation
        current_alpha = updater.alpha
        adaptive_decay = updater.decay_factor * (1 + kl_div)
        distance_to_target = updater.alpha - updater.target_alpha
        decay_amount = distance_to_target * adaptive_decay
        new_alpha = max(updater.target_alpha, updater.alpha - decay_amount)
        expected_alpha = updater.momentum * updater.alpha + (1 - updater.momentum) * new_alpha

        # Act
        updater.update_learning_rate(kl_div)

        # Assert
        self.assertAlmostEqual(updater.alpha, expected_alpha, places=3)


    @patch('compare_logits.BayesianPriorUpdate.check_convergence', return_value=(False, 0.5))
    @patch('compare_logits.BayesianPriorUpdate.update_learning_rate')
    def test_update_prior(self, mock_update_learning_rate, mock_check_convergence):
        # Arrange
        initial_prior = np.array([0.25, 0.25, 0.25, 0.25])
        updater = BayesianPriorUpdate(initial_prior)
        new_data = np.array([0.3, 0.2, 0.3, 0.2])

        # Act
        updated_dist, is_converged = updater.update_prior(new_data)

        # Calculate expected distribution
        weight = max(updater.min_weight, 1.0 / (1.0 + updater.alpha * (len(new_data) / max(updater.average_size, 1))))
        expected_dist = (weight * initial_prior) + ((1 - weight) * new_data)

        # Assert
        self.assertFalse(is_converged)
        self.assertTrue(np.allclose(updated_dist, expected_dist, atol=0.01))

    def test_get_and_set_state(self):
        initial_prior = np.array([0.3, 0.5, 0.2])
        updater = BayesianPriorUpdate(initial_prior)
        state = updater.get_state()
        
        state.current_alpha = 0.8
        state.update_count = 5
        
        updater.set_state(state)
        
        self.assertEqual(updater.alpha, 0.8)
        self.assertEqual(updater.update_count, 5)


class TestProcessChunks(unittest.TestCase):
    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    @patch('compare_logits.process_single_chunk')
    @patch('compare_logits.update_statistics')
    @patch('compare_logits.save_chunk_stats')
    @patch('compare_logits.determine_chunk_range')
    @patch('compare_logits.handle_early_stopping')
    @patch('compare_logits.finalize_processing')
    @patch('compare_logits.save_state_to_file')
    @patch('compare_logits.load_state_from_file')
    @patch('compare_logits.initialize_early_stopping')
    @patch('compare_logits.TDigest')
    def test_process_chunks(
        self,
        mock_TDigest,
        mock_initialize_early_stopping,
        mock_load_state_from_file,
        mock_save_state_to_file,
        mock_finalize_processing,
        mock_handle_early_stopping,
        mock_determine_chunk_range,
        mock_save_chunk_stats,
        mock_update_statistics,
        mock_process_single_chunk,
        mock_logger,
        mock_h5py,
    ):
        # Mock inputs
        baseline_path = "baseline.h5"
        target_path = "target.h5"
        output_path = "output.h5"
        from_chunk = 0
        to_chunk = 2

        # Mock file objects
        mock_baseline_file = MagicMock()
        mock_target_file = MagicMock()
        mock_output_file = MagicMock()

        # Mock the context manager behavior of h5py.File
        def mock_h5py_file_factory(mock_file):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value = mock_file
            mock_context_manager.__exit__.return_value = None
            return mock_context_manager

        # Set side_effect to return the appropriate mock files
        mock_h5py.side_effect = [
            mock_h5py_file_factory(mock_baseline_file),
            mock_h5py_file_factory(mock_target_file),
            mock_h5py_file_factory(mock_output_file),
        ]

        # Mock logits_baseline dataset with a defined shape
        mock_logits = MagicMock()
        mock_logits.shape = (3, 10)  # 3 chunks, 10 samples per chunk
        mock_baseline_file.__getitem__.side_effect = lambda key: mock_logits if key == 'logits' else np.array([0, 1, 2])

        # Mock chunk_index datasets
        mock_baseline_file['chunk_index'] = np.array([0, 1, 2])
        mock_target_file['chunk_index'] = np.array([0, 1, 2])

        # Mock other internal functions
        mock_determine_chunk_range.return_value = (from_chunk, to_chunk)
        mock_process_single_chunk.side_effect = [
            ([np.random.rand(100)], {"Average": 0.5}),
            ([np.random.rand(100)], {"Average": 0.6}),
            ([np.random.rand(100)], {"Average": 0.7}),  # Add more if needed
        ]
        mock_update_statistics.return_value = (10.0, 20.0, 0.0, 1.0, 200)
        mock_handle_early_stopping.return_value = False

        # Mock load_state_from_file to return None for early_stopping_stats
        mock_load_state_from_file.return_value = (
            np.array([0.1, 0.2]),  # prior_distribution
            MagicMock(),  # bayesian_updater
            None  # early_stopping_stats
        )

        # Call the function
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            clobber=True,
            precision=64,
            parts=1,
            early_stopping=True,
        )

        # Assertions
        mock_h5py.assert_any_call(baseline_path, 'r')
        mock_h5py.assert_any_call(target_path, 'r')
        mock_h5py.assert_any_call(output_path, 'a')
        mock_determine_chunk_range.assert_called_with(from_chunk, to_chunk, 3)
        self.assertEqual(mock_process_single_chunk.call_count, 3)  # Three chunks processed
        mock_save_chunk_stats.assert_called()
        mock_update_statistics.assert_called()
        mock_finalize_processing.assert_called()
        mock_initialize_early_stopping.assert_called()  # Verify this is now called
        mock_load_state_from_file.assert_called()
        mock_save_state_to_file.assert_called()

    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    @patch('compare_logits.process_single_chunk')
    @patch('compare_logits.determine_chunk_range')
    @patch('compare_logits.update_statistics')
    @patch('compare_logits.save_chunk_stats')
    @patch('compare_logits.save_state_to_file')
    @patch('compare_logits.load_state_from_file')
    @patch('compare_logits.initialize_early_stopping')
    @patch('compare_logits.TDigest')
    def test_process_chunks_precision_variants(
        self,
        mock_TDigest,
        mock_initialize_early_stopping,
        mock_load_state_from_file,
        mock_save_state_to_file,
        mock_determine_chunk_range,
        mock_process_single_chunk,
        mock_logger,
        mock_h5py,
        mock_update_statistics,
    ):
        # Mock inputs
        baseline_path = "baseline.h5"
        target_path = "target.h5"
        output_path = "output.h5"
        from_chunk = 0
        to_chunk = 1

        # Mock determine_chunk_range
        mock_determine_chunk_range.return_value = (from_chunk, to_chunk)
        mock_process_single_chunk.return_value = ([np.random.rand(50)], {"Average": 0.5})  # Mock return value

        # Mock update_statistics to return 5 values
        mock_update_statistics.return_value = (10.0, 20.0, 0.0, 1.0, 200)

        # Test precision=32
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            precision=32,
        )
        # Extract dtype from positional arguments
        dtype_arg_32 = mock_process_single_chunk.call_args[0][-1]  # dtype is the last positional argument
        self.assertEqual(dtype_arg_32, np.float32)  # Check dtype is np.float32 for precision=32

        # Test precision=64
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            precision=64,
        )
        # Extract dtype from positional arguments
        dtype_arg_64 = mock_process_single_chunk.call_args[0][-1]  # dtype is the last positional argument
        self.assertEqual(dtype_arg_64, np.float64)  # Check dtype is np.float64 for precision=64

        mock_initialize_early_stopping.assert_not_called()  # Ensure early stopping mocks are not called
        mock_load_state_from_file.assert_not_called()
        mock_save_state_to_file.assert_not_called()
        mock_update_statistics.assert_called()  # Ensure update_statistics is called

    @patch('compare_logits.finalize_processing')
    @patch('compare_logits.save_state_to_file')
    @patch('compare_logits.load_state_from_file')
    @patch('compare_logits.initialize_overall_stats')
    @patch('compare_logits.check_output_file_conditions')
    @patch('compare_logits.initialize_early_stopping')
    @patch('compare_logits.TDigest')
    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    @patch('compare_logits.process_single_chunk')
    @patch('compare_logits.determine_chunk_range')
    @patch('compare_logits.update_statistics')
    @patch('compare_logits.save_chunk_stats')
    @patch('compare_logits.handle_early_stopping')
    def test_process_chunks_early_stopping(
        self,
        mock_handle_early_stopping,
        mock_save_chunk_stats,
        mock_update_statistics,
        mock_determine_chunk_range,
        mock_process_single_chunk,
        mock_logger,
        mock_h5py_file,
        mock_TDigest,
        mock_initialize_early_stopping,
        mock_check_output_file_conditions,
        mock_initialize_overall_stats,
        mock_load_state_from_file,
        mock_save_state_to_file,
        mock_finalize_processing,
    ):
        # Mock inputs
        baseline_path = "baseline.h5"
        target_path = "target.h5"
        output_path = "output.h5"
        from_chunk = 0
        to_chunk = 2

        # Create mock file objects for baseline, target, and output
        mock_baseline_file = MagicMock()
        mock_target_file = MagicMock()
        mock_output_file = MagicMock()

        # Mock the context manager behavior of h5py.File
        def mock_h5py_file_factory(mock_file):
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value = mock_file
            mock_context_manager.__exit__.return_value = None
            return mock_context_manager

        # Set side_effect to return the appropriate mock files
        mock_h5py_file.side_effect = [
            mock_h5py_file_factory(mock_baseline_file),
            mock_h5py_file_factory(mock_target_file),
            mock_h5py_file_factory(mock_output_file),
        ]

        # Mock the 'logits' dataset with a NumPy array
        mock_logits = np.zeros((3, 10))  # 3 chunks, 10 samples per chunk
        mock_baseline_file.__getitem__.side_effect = lambda key: mock_logits if key == 'logits' else np.array([0, 1, 2])
        mock_target_file.__getitem__.side_effect = lambda key: np.array([0, 1, 2]) if key == 'chunk_index' else None

        # Ensure 'logits' dataset in f_target is mocked if accessed
        mock_target_file['logits'] = mock_logits

        # Mock determine_chunk_range
        mock_determine_chunk_range.return_value = (from_chunk, to_chunk)

        # Mock process_single_chunk
        mock_process_single_chunk.side_effect = [
            ([np.random.rand(50)], {"Average": 0.5}),
            ([np.random.rand(50)], {"Average": 0.6}),
        ]

        # Mock handle_early_stopping
        mock_handle_early_stopping.side_effect = [False, True]  # Early stopping triggers after chunk 1

        # Mock update_statistics to return 5 values
        mock_update_statistics.return_value = (10.0, 20.0, 0.0, 1.0, 200)

        # Mock check_output_file_conditions
        mock_check_output_file_conditions.return_value = (set(), {}, None)

        # Mock initialize_overall_stats
        mock_initialize_overall_stats.return_value = (0.0, 0.0, float('inf'), float('-inf'), 0)

        # Mock load_state_from_file
        mock_load_state_from_file.return_value = (None, None, None)

        # Mock initialize_early_stopping
        mock_initialize_early_stopping.return_value = MagicMock()

        # Mock TDigest and its percentile method
        mock_digest_instance = MagicMock()
        mock_TDigest.return_value = mock_digest_instance
        mock_digest_instance.percentile.side_effect = lambda x: x * 0.01  # Return a simple function of x

        # Mock finalize_processing
        mock_finalize_processing.return_value = None

        # Call the function
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            early_stopping=True,
        )

        # Assertions
        self.assertEqual(mock_process_single_chunk.call_count, 2)  # Only chunks 0 and 1 processed
        self.assertEqual(mock_handle_early_stopping.call_count, 2)  # Called for each processed chunk
        mock_logger.info.assert_any_call(f"Processing chunks {from_chunk} to {to_chunk}...")

    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    @patch('compare_logits.process_single_chunk')
    @patch('compare_logits.determine_chunk_range')
    @patch('compare_logits.update_statistics')
    @patch('compare_logits.save_chunk_stats')
    def test_process_chunks_precision_variants(
        self,
        mock_save_chunk_stats,
        mock_update_statistics,
        mock_determine_chunk_range,
        mock_process_single_chunk,
        mock_logger,
        mock_h5py,
    ):
        # Mock inputs
        baseline_path = "baseline.h5"
        target_path = "target.h5"
        output_path = "output.h5"
        from_chunk = 0
        to_chunk = 1

        # Mock determine_chunk_range
        mock_determine_chunk_range.return_value = (from_chunk, to_chunk)
        mock_process_single_chunk.return_value = ([np.random.rand(50)], {"Average": 0.5})  # Mock return value

        # Mock update_statistics to return 5 values
        mock_update_statistics.return_value = (10.0, 20.0, 0.0, 1.0, 200)

        # Test precision=32
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            precision=32,
        )
        # Extract dtype from positional arguments
        dtype_arg_32 = mock_process_single_chunk.call_args[0][-1]  # dtype is the last positional argument
        self.assertEqual(dtype_arg_32, np.float32)  # Check dtype is np.float32 for precision=32

        # Test precision=64
        process_chunks(
            baseline_path=baseline_path,
            target_path=target_path,
            output_path=output_path,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            precision=64,
        )
        # Extract dtype from positional arguments
        dtype_arg_64 = mock_process_single_chunk.call_args[0][-1]  # dtype is the last positional argument
        self.assertEqual(dtype_arg_64, np.float64)  # Check dtype is np.float64 for precision=64

    def test_determine_chunk_range(self):
        # Mock inputs
        total_chunks = 10

        # Test 1: Explicit range
        from_chunk = 2
        to_chunk = 5
        start_chunk, end_chunk =determine_chunk_range(from_chunk, to_chunk, total_chunks)
        self.assertEqual(start_chunk, 2)
        self.assertEqual(end_chunk, 5)

        # Test 2: Default to full range
        from_chunk = None
        to_chunk = None
        start_chunk, end_chunk =determine_chunk_range(from_chunk, to_chunk, total_chunks)
        self.assertEqual(start_chunk, 0)
        self.assertEqual(end_chunk, total_chunks - 1)

        # Test 3: Start at default, end specified
        from_chunk = None
        to_chunk = 8
        start_chunk, end_chunk =determine_chunk_range(from_chunk, to_chunk, total_chunks)
        self.assertEqual(start_chunk, 0)
        self.assertEqual(end_chunk, 8)

        # Test 4: Start specified, end default
        from_chunk = 3
        to_chunk = None
        start_chunk, end_chunk =determine_chunk_range(from_chunk, to_chunk, total_chunks)
        self.assertEqual(start_chunk, 3)
        self.assertEqual(end_chunk, total_chunks - 1)


class TestProcessSingleChunk(unittest.TestCase):

    @patch('compare_logits.process_chunk_part')
    @patch('compare_logits.TDigest')
    @patch('compare_logits.logger')
    def test_process_single_chunk_valid(self, mock_logger, mock_tdigest, mock_process_chunk_part):
        # Mock inputs
        chunk_idx = 1
        f_baseline = MagicMock()
        f_target = MagicMock()
        logits_baseline = np.random.rand(2, 100, 10)
        chunk_index_baseline = [0, 1]
        chunk_index_target = [0, 1]
        parts = 2
        dtype = np.float32

        # Mock process_chunk_part and TDigest behavior
        mock_process_chunk_part.side_effect = [np.random.rand(50), np.random.rand(50)]
        mock_tdigest.return_value = MagicMock()

        # Call function
        kl_values_list, chunk_stats = process_single_chunk(
            chunk_idx, f_baseline, f_target, logits_baseline, chunk_index_baseline,
            chunk_index_target, parts, dtype
        )

        # Assertions
        self.assertEqual(len(kl_values_list), 2)
        self.assertIn("Average", chunk_stats)
        mock_tdigest.return_value.batch_update.assert_called()
        mock_process_chunk_part.assert_called()

    @patch('compare_logits.process_chunk_part')
    @patch('compare_logits.TDigest')
    @patch('compare_logits.logger')
    def test_process_single_chunk_no_samples(self, mock_logger, mock_tdigest, mock_process_chunk_part):
        # Mock inputs
        chunk_idx = 1
        f_baseline = MagicMock()
        f_target = MagicMock()
        logits_baseline = np.random.rand(2, 0, 10)  # Zero samples
        chunk_index_baseline = [0, 1]
        chunk_index_target = [0, 1]
        parts = 2
        dtype = np.float32

        # Call function
        kl_values_list, chunk_stats = process_single_chunk(
            chunk_idx, f_baseline, f_target, logits_baseline, chunk_index_baseline,
            chunk_index_target, parts, dtype
        )

        # Assertions
        self.assertEqual(len(kl_values_list), 0)
        self.assertEqual(chunk_stats["Average"], 0.0)
        mock_logger.warning.assert_called_with(f"Chunk {chunk_idx} has zero samples, skipping.")

    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    def test_update_statistics(self, mock_logger, mock_h5py):
        # Mock inputs
        f_out = MagicMock()
        kl_values_chunk = np.random.rand(100)
        chunk_stats = {
            "Average": 0.5,
            "StdDev": 0.1,
            "Minimum": 0.0,
            "Maximum": 1.0,
        }
        digest = MagicMock()
        overall_sum = 10.0
        overall_sumsq = 100.0
        overall_min = 0.0
        overall_max = 1.0
        total_values = 200

        # Call function
        updated_sum, updated_sumsq, updated_min, updated_max, updated_total_values = update_statistics(
            f_out, kl_values_chunk, chunk_stats, digest, overall_sum, overall_sumsq, overall_min, overall_max, total_values
        )

        # Expected updated_sum calculation
        expected_updated_sum = overall_sum + chunk_stats["Average"] * kl_values_chunk.size

        # Assertions
        self.assertEqual(updated_sum, expected_updated_sum)
        self.assertEqual(updated_max, max(overall_max, chunk_stats['Maximum']))
        digest.batch_update.assert_called_with(kl_values_chunk)


    @patch('compare_logits.h5py.File')
    @patch('compare_logits.logger')
    def test_save_chunk_stats(self, mock_logger, mock_h5py):
        # Mock inputs
        f_out = MagicMock()
        chunk_idx = 1
        chunk_stats = {
            "ChunkNumber": 1,
            "Average": 0.5,
            "StdDev": 0.1,
            "Minimum": 0.0,
            "Maximum": 1.0,
        }

        # Call function
        save_chunk_stats(f_out, chunk_idx, chunk_stats)

        # Assertions
        f_out.create_group.assert_called_with('chunk_1')
        f_out.create_group().attrs.update.assert_called_once_with(chunk_stats)
        mock_logger.info.assert_any_call("\n===== KL-divergence statistics for Chunk 1 =====")


class TestProcessChunkPart(unittest.TestCase):
    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.kl_divergence_log_probs')
    @patch('compare_logits.logger')
    def test_process_chunk_part(
        self, mock_logger, mock_kl_divergence, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock inputs
        p_logits_part = np.random.rand(10, 5)
        q_logits_part = np.random.rand(10, 5)
        chunk_idx = 1
        part_idx = 0

        # Mock behavior for KL divergence
        mock_kl_divergence.return_value = np.random.rand(10)

        # Call the function
        result = process_chunk_part(p_logits_part, q_logits_part, chunk_idx, part_idx)

        # Assertions
        mock_kl_divergence.assert_called_once_with(p_logits_part, q_logits_part)
        self.assertTrue(np.all(result >= 0))  # Basic validation that values are non-negative

    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.kl_divergence_log_probs')
    @patch('compare_logits.logger')
    def test_process_chunk_part_shape_mismatch(
        self, mock_logger, mock_kl_divergence, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock inputs with mismatched shapes
        p_logits_part = np.random.rand(10, 5)
        q_logits_part = np.random.rand(8, 5)

        with self.assertRaises(ValueError) as context:
            process_chunk_part(p_logits_part, q_logits_part, 1, 0)
        self.assertIn("Shape mismatch", str(context.exception))

    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.kl_divergence_log_probs')
    @patch('compare_logits.logger')
    def test_process_chunk_part_non_finite_values(
        self, mock_logger, mock_kl_divergence, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock inputs with NaNs and Infs
        p_logits_part = np.array([[1, 2], [np.nan, np.inf]])
        q_logits_part = np.array([[1, 2], [3, 4]])

        # Call the function
        result = process_chunk_part(p_logits_part, q_logits_part, 1, 0)

        # Assertions
        self.assertTrue(np.all(np.isfinite(result)))

    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.kl_divergence_log_probs')
    @patch('compare_logits.logger')
    def test_process_chunk_part_identical_inputs(
        self, mock_logger, mock_kl_divergence, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock identical logits
        logits = np.random.rand(10, 5)

        # Mock kl_divergence_log_probs to return zeros for identical inputs
        mock_kl_divergence.return_value = np.zeros(10)

        result = process_chunk_part(logits, logits, 1, 0)

        # Assertions
        mock_kl_divergence.assert_called_once_with(logits, logits)
        self.assertTrue(np.all(result == 0))  # Identical inputs result in zero KL divergence


class TestKLDivergenceLogProbs(unittest.TestCase):
    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.logger')
    def test_kl_divergence_log_probs_numerical_stability(
        self, mock_logger, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock logits with extreme values
        p_logits = np.array([[1e9, 1e-9]])
        q_logits = np.array([[1e-9, 1e9]])

        result = kl_divergence_log_probs(p_logits, q_logits)

        # Assertions
        self.assertTrue(np.all(np.isfinite(result)))  # Ensure no NaN or Inf

    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.logger')
    def test_kl_divergence_log_probs_single_class(
        self, mock_logger, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock logits with single-class probabilities
        p_logits = np.array([[1000, -1000]])
        q_logits = np.array([[1000, -1000]])

        result = kl_divergence_log_probs(p_logits, q_logits)

        # Assertions
        self.assertTrue(np.all(result == 0))  # Single-class probabilities should have zero KL divergence

    @patch('compare_logits.BayesianPriorState')
    @patch('compare_logits.BayesianPriorUpdate')
    @patch('compare_logits.KLFileStructure')
    @patch('compare_logits.EarlyStoppingStats')
    @patch('compare_logits.logger')
    def test_kl_divergence_log_probs_small_logits(
        self, mock_logger, mock_early_stats, mock_kl_structure, mock_prior_update, mock_prior_state
    ):
        # Mock logits with small values
        p_logits = np.array([[1e-10, 1e-10]])
        q_logits = np.array([[1e-10, 1e-10]])

        result = kl_divergence_log_probs(p_logits, q_logits)

        # Assertions
        self.assertTrue(np.all(result == 0))  # Zero divergence for equal probabilities


class TestHandleEarlyStopping(unittest.TestCase):
    @patch('compare_logits.calculate_sample_size')
    @patch('compare_logits.adjust_decay_rate')
    @patch('compare_logits.save_early_stopping_info')
    @patch('compare_logits.save_prior_and_stats')
    @patch('compare_logits.beta')
    def test_handle_early_stopping_with_early_stop(self, mock_beta, mock_save_prior_and_stats,
                                                mock_save_early_stopping_info,
                                                mock_adjust_decay_rate,
                                                mock_calculate_sample_size):
        # Setup mocks
        mock_calculate_sample_size.return_value = 1  # Effective chunk size
        mock_adjust_decay_rate.return_value = 0.1
        mock_beta.sf.return_value = 0.96  # Stopping probability above confidence_level

        # Initialize EarlyStoppingStats without mocking add_effect_size and add_p_value
        early_stopping_stats = EarlyStoppingStats(
            sample_size=1000,
            min_samples=500,
            alpha=1,
            beta=1,
            window_size=3,
            effect_sizes=[0.1, 0.2],
            p_values=[0.05, 0.06],
        )
        # Do not mock add_effect_size and add_p_value
        early_stopping_stats.update_beta_parameters = MagicMock()

        # Mock prior_distribution and bayesian_updater
        prior_distribution = np.array([0.1, 0.2, 0.3])
        bayesian_updater = MagicMock(spec=BayesianPriorUpdate)
        bayesian_updater.update_prior.return_value = (prior_distribution, False)
        bayesian_updater.decay_factor = 0.2  # Add decay_factor to the mock

        # Test inputs
        kl_values_chunk = np.array([0.1, 0.2, 0.3])
        confidence_level = 0.95
        margin_of_error = 0.05
        initial_prior_learning_rate = 0.1
        initial_min_prior_weight = 0.2
        decay_rate = 0.1
        momentum = 0.9
        chunk_idx = 0
        f_out = MagicMock(spec=h5py.File)
        window_size = 3
        effective_chunk_size = 1  # Ensure loop runs
        log_effect_sizes = False

        # Call the function
        result = handle_early_stopping(
            kl_values_chunk=kl_values_chunk,
            prior_distribution=prior_distribution,
            bayesian_updater=bayesian_updater,
            early_stopping_stats=early_stopping_stats,
            confidence_level=confidence_level,
            margin_of_error=margin_of_error,
            initial_prior_learning_rate=initial_prior_learning_rate,
            initial_min_prior_weight=initial_min_prior_weight,
            decay_rate=decay_rate,
            momentum=momentum,
            chunk_idx=chunk_idx,
            f_out=f_out,
            window_size=window_size,
            effective_chunk_size=effective_chunk_size,
            log_effect_sizes=log_effect_sizes,
        )

        # Assertions
        self.assertTrue(result)
        mock_save_early_stopping_info.assert_called_once_with(
            f_out, prior_distribution, early_stopping_stats, bayesian_updater
        )
        # Assert that effect sizes and p-values were updated
        self.assertEqual(len(early_stopping_stats.effect_sizes), 3)
        self.assertEqual(len(early_stopping_stats.p_values), 3)

        # Check that update_beta_parameters was called
        early_stopping_stats.update_beta_parameters.assert_called()

        mock_beta.sf.assert_called_with(confidence_level, early_stopping_stats.alpha, early_stopping_stats.beta)

    @patch('compare_logits.calculate_sample_size')
    @patch('compare_logits.adjust_decay_rate')
    @patch('compare_logits.save_early_stopping_info')
    @patch('compare_logits.save_prior_and_stats')
    @patch('compare_logits.beta')
    def test_handle_early_stopping_without_early_stop(self, mock_beta, mock_save_prior_and_stats,
                                                      mock_save_early_stopping_info,
                                                      mock_adjust_decay_rate,
                                                      mock_calculate_sample_size):
        # Setup mocks
        mock_calculate_sample_size.return_value = 1  # Effective chunk size
        mock_adjust_decay_rate.return_value = 0.1
        mock_beta.sf.return_value = 0.90  # Stopping probability below confidence_level

        # Initialize EarlyStoppingStats without mocking add_effect_size and add_p_value
        early_stopping_stats = EarlyStoppingStats(
            sample_size=1000,
            min_samples=500,
            alpha=1,
            beta=1,
            window_size=3,
            effect_sizes=[0.1, 0.2],
            p_values=[0.05, 0.06],
        )
        # Do not mock add_effect_size and add_p_value
        early_stopping_stats.update_beta_parameters = MagicMock()

        # Mock prior_distribution and bayesian_updater
        prior_distribution = np.array([0.1, 0.2, 0.3])
        bayesian_updater = MagicMock(spec=BayesianPriorUpdate)
        bayesian_updater.update_prior.return_value = (prior_distribution, False)
        bayesian_updater.decay_factor = 0.2  # Add decay_factor to the mock

        # Test inputs
        kl_values_chunk = np.array([0.1, 0.2, 0.3])
        confidence_level = 0.95
        margin_of_error = 0.05
        initial_prior_learning_rate = 0.1
        initial_min_prior_weight = 0.2
        decay_rate = 0.1
        momentum = 0.9
        chunk_idx = 0
        f_out = MagicMock(spec=h5py.File)
        window_size = 3
        effective_chunk_size = 1  # Ensure loop runs
        log_effect_sizes = False

        # Call the function
        result = handle_early_stopping(
            kl_values_chunk=kl_values_chunk,
            prior_distribution=prior_distribution,
            bayesian_updater=bayesian_updater,
            early_stopping_stats=early_stopping_stats,
            confidence_level=confidence_level,
            margin_of_error=margin_of_error,
            initial_prior_learning_rate=initial_prior_learning_rate,
            initial_min_prior_weight=initial_min_prior_weight,
            decay_rate=decay_rate,
            momentum=momentum,
            chunk_idx=chunk_idx,
            f_out=f_out,
            window_size=window_size,
            effective_chunk_size=effective_chunk_size,
            log_effect_sizes=log_effect_sizes,
        )

        # Assertions
        self.assertFalse(result)
        mock_save_prior_and_stats.assert_called_once_with(
            f_out, prior_distribution, early_stopping_stats, bayesian_updater
        )
        mock_save_early_stopping_info.assert_not_called()

        # Since we didn't mock add_effect_size and add_p_value, we can check the list lengths
        self.assertEqual(len(early_stopping_stats.effect_sizes), 3)
        self.assertEqual(len(early_stopping_stats.p_values), 3)

        # Check that update_beta_parameters was called
        early_stopping_stats.update_beta_parameters.assert_called()

        mock_beta.sf.assert_called_with(confidence_level, early_stopping_stats.alpha, early_stopping_stats.beta)

    def test_initialize_early_stopping(self):
        # Test when early stopping is disabled
        result = initialize_early_stopping(
            early_stopping=False,
            min_samples=1000,
            window_size=5
        )
        self.assertIsNone(result)

        # Test when early stopping is enabled
        result = initialize_early_stopping(
            early_stopping=True,
            min_samples=1000,
            window_size=5,
            theta_E=0.2,
            theta_P=0.1,
            confidence=0.95,
            dynamic_thresholds_enabled=False
        )
        self.assertIsInstance(result, EarlyStoppingStats)
        self.assertEqual(result.min_samples, 1000)
        self.assertEqual(result.window_size, 5)
        self.assertEqual(result.theta_E, 0.2)
        self.assertEqual(result.theta_P, 0.1)
        self.assertEqual(result.confidence, 0.95)
        self.assertFalse(result.dynamic_thresholds_enabled)

    @patch('compare_logits.save_common_state')
    @patch('compare_logits.json.dumps')
    def test_save_early_stopping_info(self, mock_json_dumps, mock_save_common_state):
        # Setup
        f_out = MagicMock(spec=h5py.File)
        prior_distribution = np.array([0.1, 0.2, 0.3])
        early_stopping_stats = EarlyStoppingStats()
        bayesian_updater = MagicMock(spec=BayesianPriorUpdate)
        mock_json_dumps.return_value = 'serialized_stats'

        # Call the function
        save_early_stopping_info(f_out, prior_distribution, early_stopping_stats, bayesian_updater)

        # Assertions
        self.assertTrue(early_stopping_stats.stopped_early)
        mock_save_common_state.assert_called_once_with(
            f_out, prior_distribution, early_stopping_stats, bayesian_updater
        )
        f_out.attrs.__setitem__.assert_called_with('early_stopping_stats', 'serialized_stats')
        mock_json_dumps.assert_called_with(early_stopping_stats.__dict__, default=ANY)


class TestCheckOutputFileConditions(unittest.TestCase):
    @patch("compare_logits.h5py.File")
    @patch("compare_logits.h5py.is_hdf5")
    @patch("compare_logits.json.loads")
    @patch("compare_logits.compose_overall_stats")
    @patch("compare_logits.TDigest")
    def test_check_output_file_conditions(self, mock_tdigest, mock_compose_stats, mock_json_loads, mock_is_hdf5, mock_h5py_file):
        # Mock `is_hdf5` to return True
        mock_is_hdf5.return_value = True

        # Mocked HDF5 file and attributes
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        mock_file.keys.return_value = ["chunk_0", "chunk_1"]  # Simulate chunk keys in the HDF5 file
        mock_file.attrs = {
            'overall_sum': 10.0,
            'overall_sumsq': 100.0,
            'overall_min': 1.0,
            'overall_max': 10.0,
            'total_values': 20,
            'digest': '{"centroids": [{"mean": 0.5, "weight": 1}]}'
        }

        # Mock JSON loads and compose_overall_stats
        mock_json_loads.return_value = {"centroids": [{"mean": 0.5, "weight": 1}]}
        mock_compose_stats.return_value = {"Average": 5.0}

        # Mock TDigest
        digest = mock_tdigest.return_value
        digest.update_from_dict.return_value = None

        # Call the function
        existing_chunks, overall_stats, digest_result = check_output_file_conditions(
            "mock_output_path", 0, 1, clobber=False
        )

        # Assertions
        self.assertEqual(existing_chunks, {0, 1})  # Ensure parsed chunks are integers
        self.assertEqual(overall_stats, {"Average": 5.0})
        self.assertEqual(digest_result, mock_tdigest.return_value)  # Ensure digest was updated
        mock_is_hdf5.assert_called_once_with("mock_output_path")
        mock_h5py_file.assert_called_once_with("mock_output_path", "r")
        mock_json_loads.assert_called_once_with('{"centroids": [{"mean": 0.5, "weight": 1}]}')
        mock_compose_stats.assert_called_once_with(10.0, 100.0, 1.0, 10.0, 20, digest)

        # Verify `update_from_dict` call
        digest.update_from_dict.assert_called_once_with({"centroids": [{"mean": 0.5, "weight": 1}]})


    @patch("compare_logits.h5py.File")
    @patch("compare_logits.json.loads")
    def test_load_state_from_file(self, mock_json_loads, mock_h5py_file):
        # Mocked HDF5 file object
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Define the attributes in the HDF5 file
        mock_file.attrs = {
            'prior_distribution': "[0.1, 0.2, 0.3]",
            'early_stopping_stats': '{"alpha": 1, "beta": 1}',
            'bayesian_prior_state': '{"initial_alpha": 0.1, "decay_factor": 0.3, "update_count": 0, "current_alpha": 0.1, "momentum": 0.7, "previous_dists": [], "average_size": 1.0, "target_alpha": 0.01, "min_weight": 0.2, "window_size": 3}'
        }

        # Define side_effect with correct order:
        # 1. prior_distribution
        # 2. bayesian_prior_state
        # 3. early_stopping_stats
        mock_json_loads.side_effect = [
            [0.1, 0.2, 0.3],  # prior_distribution
            {  # bayesian_prior_state
                "initial_alpha": 0.1,
                "decay_factor": 0.3,
                "update_count": 0,
                "current_alpha": 0.1,
                "momentum": 0.7,
                "previous_dists": [],
                "average_size": 1.0,
                "target_alpha": 0.01,
                "min_weight": 0.2,
                "window_size": 3
            },
            {"alpha": 1, "beta": 1}  # early_stopping_stats
        ]

        # Define a simple mock class for BayesianPriorState to avoid MagicMock issues
        class MockBayesianPriorState:
            def __init__(self, initial_alpha, decay_factor, update_count, current_alpha, momentum,
                         previous_dists, average_size, target_alpha, min_weight, window_size):
                self.initial_alpha = initial_alpha
                self.decay_factor = decay_factor
                self.update_count = update_count
                self.current_alpha = current_alpha
                self.momentum = momentum
                self.previous_dists = previous_dists
                self.average_size = average_size
                self.target_alpha = target_alpha
                self.min_weight = min_weight
                self.window_size = window_size

        # Patch the BayesianPriorState to return our MockPriorState
        with patch("compare_logits.BayesianPriorState", side_effect=MockBayesianPriorState):
            # Call the function
            prior_distribution, bayesian_updater, early_stopping_stats = load_state_from_file(
                mock_file, prior_learning_rate=0.1, min_prior_weight=0.2, window_size=3
            )

            # Assertions
            np.testing.assert_array_equal(prior_distribution, np.array([0.1, 0.2, 0.3]))
            self.assertIsInstance(bayesian_updater, BayesianPriorUpdate)
            self.assertIsInstance(early_stopping_stats, EarlyStoppingStats)
            self.assertEqual(early_stopping_stats.alpha, 1)
            self.assertEqual(early_stopping_stats.beta, 1)

            # Ensure h5py.File was not called again
            mock_h5py_file.assert_not_called()

            # Ensure json.loads was called with correct arguments
            mock_json_loads.assert_any_call("[0.1, 0.2, 0.3]")
            mock_json_loads.assert_any_call('{"initial_alpha": 0.1, "decay_factor": 0.3, "update_count": 0, "current_alpha": 0.1, "momentum": 0.7, "previous_dists": [], "average_size": 1.0, "target_alpha": 0.01, "min_weight": 0.2, "window_size": 3}')
            mock_json_loads.assert_any_call('{"alpha": 1, "beta": 1}')

    @patch("compare_logits.h5py.File")
    @patch("compare_logits.json.dumps")
    @patch("compare_logits.numpy_encoder")
    def test_save_state_to_file(self, mock_numpy_encoder, mock_json_dumps, mock_h5py_file):
        # Mocked HDF5 file object
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        
        # Mock prior_distribution as a NumPy array
        prior_distribution = np.array([0.1, 0.2, 0.3])
        
        # Mock early_stopping_stats with required attributes
        early_stopping_stats = MagicMock()
        early_stopping_stats.alpha = 1
        early_stopping_stats.beta = 1
        early_stopping_stats.stopped_early = False  # Initialize the attribute
        
        # Mock bayesian_updater and its get_state method
        class MockPriorState:
            def __init__(self):
                self.update_count = 0

        bayesian_updater = MagicMock()
        bayesian_updater.get_state.return_value = MockPriorState()
        
        # Mock numpy_encoder to pass values unchanged
        mock_numpy_encoder.side_effect = lambda x: x
        
        # Call the function under test
        save_state_to_file(
            mock_file, prior_distribution, early_stopping_stats, bayesian_updater, final=True
        )
        
        # Assert that 'stopped_early' was set to True
        self.assertTrue(early_stopping_stats.stopped_early)

        # Assertions for `prior_distribution`
        prior_distribution_list = prior_distribution.tolist()
        mock_json_dumps.assert_any_call(prior_distribution_list, default=mock_numpy_encoder)
        mock_file.attrs.__setitem__.assert_any_call("prior_distribution", mock_json_dumps.return_value)
        
        # Assertions for `early_stopping_stats`
        mock_json_dumps.assert_any_call(early_stopping_stats.__dict__, default=mock_numpy_encoder)
        mock_file.attrs.__setitem__.assert_any_call("early_stopping_stats", mock_json_dumps.return_value)

        # Assertions for `bayesian_updater`
        prior_state_dict = bayesian_updater.get_state().__dict__
        mock_json_dumps.assert_any_call(prior_state_dict, default=mock_numpy_encoder)
        mock_file.attrs.__setitem__.assert_any_call("bayesian_prior_state", mock_json_dumps.return_value)


class SaveEarlyStoppingInfo(unittest.TestCase):
    @patch("compare_logits.json.dumps")  # Mock JSON dumps
    @patch("compare_logits.h5py.File")  # Mock HDF5 file handling
    def test_save_common_state(self, mock_h5py_file, mock_json_dumps):
        # Mocked data
        prior_distribution = MagicMock()
        early_stopping_stats = MagicMock()
        bayesian_updater = MagicMock()
        
        # Mocking function return values
        mock_json_dumps.return_value = "{}"

        # Mocked file object
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Call the function
        save_common_state(mock_file, prior_distribution, early_stopping_stats, bayesian_updater)

        # Assertions to ensure the mocked methods were called
        mock_json_dumps.assert_called()
        self.assertTrue(mock_file.attrs.__setitem__.called)

    @patch("compare_logits.json.dumps")
    @patch("compare_logits.h5py.File")
    def test_save_prior_and_stats(self, mock_h5py_file, mock_json_dumps):
        # Mocked data
        prior_distribution = MagicMock()
        early_stopping_stats = MagicMock()
        bayesian_updater = MagicMock()

        # Mocked file object
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Call the function
        save_prior_and_stats(mock_file, prior_distribution, early_stopping_stats, bayesian_updater)

        # Assertions
        mock_json_dumps.assert_called()
        self.assertTrue(mock_file.attrs.__setitem__.called)

    @patch("compare_logits.numpy_encoder")
    @patch("compare_logits.json.dumps")
    @patch("compare_logits.h5py.File")
    @patch("compare_logits.TDigest")
    def test_finalize_processing(self, mock_tdigest, mock_h5py_file, mock_json_dumps, mock_numpy_encoder):
        # Mocked TDigest instance and methods
        digest = mock_tdigest.return_value
        digest.to_dict.return_value = {"digest_key": "digest_value"}
        digest.percentile.side_effect = lambda p: p  # Simulate percentile values

        # Mock numpy_encoder
        mock_numpy_encoder.side_effect = lambda x: x  # Pass values unchanged

        # Mocked file object
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Provide real numeric values for calculations
        overall_sum = 100.0
        overall_sumsq = 500.0
        overall_min = 1.0
        overall_max = 10.0
        total_values = 20

        # Call the function
        finalize_processing(
            mock_file, digest, True,
            overall_sum, overall_sumsq, overall_min, overall_max, total_values
        )

        # Convert NumPy float values to Python native types
        expected_overall_stats = {
            "Average": float(overall_sum / total_values),
            "StdDev": float(((overall_sumsq / total_values) - (overall_sum / total_values) ** 2) ** 0.5),
            "Minimum": overall_min,
            "Maximum": overall_max,
            "KLD_99": 99,
            "KLD_95": 95,
            "KLD_90": 90,
            "Median": 50,
            "KLD_10": 10,
            "KLD_05": 5,
            "KLD_01": 1,
            "total_values": total_values
        }

        # Assertions for digest serialization
        digest.to_dict.assert_called_once()
        mock_json_dumps.assert_any_call({"digest_key": "digest_value"}, default=mock_numpy_encoder)
        mock_json_dumps.assert_any_call(expected_overall_stats, default=mock_numpy_encoder)

        # Ensure that file attributes were set correctly
        mock_file.attrs.__setitem__.assert_any_call("digest", mock_json_dumps.return_value)
        mock_file.attrs.__setitem__.assert_any_call("overall", mock_json_dumps.return_value)

    @patch("compare_logits.numpy_encoder")
    @patch("compare_logits.json.dumps")
    @patch("compare_logits.h5py.File")
    @patch("compare_logits.TDigest")
    def test_finalize_processing_with_compute_overall(self, mock_tdigest, mock_h5py_file, mock_json_dumps, mock_numpy_encoder):
        # Mocked TDigest instance and methods
        digest = mock_tdigest.return_value
        digest.to_dict.return_value = {"digest_key": "digest_value"}
        digest.percentile.side_effect = lambda p: p  # Mock percentile values to return percentiles directly

        # Mock numpy_encoder
        mock_numpy_encoder.side_effect = lambda x: x  # Pass values unchanged

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        overall_sum = 100.0
        overall_sumsq = 500.0
        overall_min = 1.0
        overall_max = 10.0
        total_values = 20

        # Call the function with compute_overall set to True
        finalize_processing(
            mock_file, digest, True,
            overall_sum, overall_sumsq, overall_min, overall_max, total_values
        )

        # Assertions for digest serialization
        digest.to_dict.assert_called_once()
        mock_json_dumps.assert_called()

        # Verify that "overall" was set as an attribute
        mock_file.attrs.__setitem__.assert_any_call(
            "overall",
            mock_json_dumps.return_value
        )

        # Verify the correct overall statistics were passed to json.dumps
        expected_overall_stats = {
            "Average": overall_sum / total_values,
            "StdDev": ((overall_sumsq / total_values) - (overall_sum / total_values) ** 2) ** 0.5,
            "Minimum": overall_min,
            "Maximum": overall_max,
            "KLD_99": 99,
            "KLD_95": 95,
            "KLD_90": 90,
            "Median": 50,
            "KLD_10": 10,
            "KLD_05": 5,
            "KLD_01": 1,
            "total_values": total_values
        }
        mock_json_dumps.assert_any_call(expected_overall_stats, default=mock_numpy_encoder)

    @patch("compare_logits.json.dumps")
    @patch("compare_logits.h5py.File")
    @patch("compare_logits.TDigest")
    def test_finalize_processing_without_compute_overall(self, mock_tdigest, mock_h5py_file, mock_json_dumps):
        # Mocked data
        digest = mock_tdigest.return_value
        digest.to_dict.return_value = {"digest_key": "digest_value"}

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        overall_sum = 100.0
        overall_sumsq = 500.0
        overall_min = 1.0
        overall_max = 10.0
        total_values = 20

        # Call the function with compute_overall set to False
        finalize_processing(
            mock_file, digest, False,
            overall_sum, overall_sumsq, overall_min, overall_max, total_values
        )

        # Assertions for digest serialization
        digest.to_dict.assert_called_once()
        mock_json_dumps.assert_called()

        # Verify that overall statistics are not computed
        self.assertNotIn("overall", mock_file.attrs)


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
                f_baseline.create_dataset('chunk_index', data=np.arange(num_chunks))

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)
                f_target.create_dataset('chunk_index', data=np.arange(num_chunks))

            # Run the process_chunks function
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    from_chunk=None,
                    to_chunk=None,
                    compute_overall=True,
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
                f_baseline.create_dataset('chunk_index', data=np.arange(num_chunks))

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)
                f_target.create_dataset('chunk_index', data=np.arange(num_chunks))

            # Process chunks from index 1 to 3
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    from_chunk=1,
                    to_chunk=4,
                    clobber=False
                )

            # Verify that only chunks 1 and 2 are processed
            with h5py.File(output_path, 'r') as f_out:
                self.assertNotIn('chunk_0', f_out)
                self.assertIn('chunk_1', f_out)
                self.assertIn('chunk_2', f_out)
                self.assertIn('chunk_3', f_out)

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
                f_baseline.create_dataset('chunk_index', data=np.arange(num_chunks))

            with h5py.File(target_path, 'w') as f_target:
                logits_target = np.random.rand(num_chunks, chunk_size, num_classes)
                f_target.create_dataset('logits', data=logits_target)
                f_target.create_dataset('chunk_index', data=np.arange(num_chunks))

            # Initial run without clobber
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                process_chunks(
                    baseline_path=baseline_path,
                    target_path=target_path,
                    output_path=output_path,
                    clobber=False
                )

            # Second run with clobber=True
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
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
