import contextlib
import json
import logging
import psutil
import tempfile
import torch
import unittest
from unittest.mock import MagicMock, patch

from mock_model import MockModel
from best_bub import (
    setup_study, initialize_batch_and_model_config, tokenize, chunk_text,
    execute_trials, report_results, get_model_size_gb, get_model_config,
    get_available_memory_gb, estimate_max_batch_size, create_trial, 
    update_best_chunk_time_with_probability, update_bayesian_mean_variance, 
    objective_wrapper, prepare_llama_args, objective, main, ExponentRange
)


logger = logging.getLogger("best_bub")

class TestInitializeBatchAndModelConfig(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @contextlib.contextmanager
    def temporary_logger_config(self, logger, level):
        logging.disable(logging.NOTSET)
        
        original_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logging.disable(logging.CRITICAL)
            logger.setLevel(original_level)

    @patch('os.path.getsize')
    def test_get_model_size_gb(self, mock_getsize):
        test_cases = [
            # (file size in bytes, expected size in GB)
            (1024**3, 1.0),           # 1 GB
            (2.5 * 1024**3, 2.5),     # 2.5 GB
            (0.5 * 1024**3, 0.5),     # 500 MB
            (15.7 * 1024**3, 15.7),   # 15.7 GB
        ]

        for bytes_size, expected_gb in test_cases:
            with self.subTest(bytes_size=bytes_size):
                mock_getsize.return_value = bytes_size
                result = get_model_size_gb("dummy/path/model.bin")
                self.assertAlmostEqual(result, expected_gb, places=6)

    @patch('os.path.exists')
    def test_get_model_config(self, mock_exists):
        """Test when the config.json file exists."""
        
        # Mock the os.path.exists function to return True
        mock_exists.return_value = True
        
        # Mock the open function to return a MagicMock that simulates file reading
        with patch('builtins.open', new_callable=MagicMock) as mock_open:
            # Set up the mock to return a specific JSON content
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                'hidden_size': 1024,
                'num_hidden_layers': 8
            })
            
            # Call the function with a model path
            hidden_size, num_layers = get_model_config('/path/to/model')
            
            # Assert that the returned values match the expected configuration
            self.assertEqual(hidden_size, 1024)
            self.assertEqual(num_layers, 8)

    @patch('os.path.exists')
    def test_get_model_config_config_missing(self, mock_exists):
        """Test when the config.json file is missing."""
        
        # Mock the os.path.exists function to return False
        mock_exists.return_value = False
        
        # Call the function with a model path
        hidden_size, num_layers = get_model_config('/path/to/model')
        
        # Assert that the returned values match the default configuration
        self.assertEqual(hidden_size, 4096)
        self.assertEqual(num_layers, 32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_get_available_memory_gb_cuda_available(self):
        """Test when CUDA is available."""
        with patch('torch.cuda.get_device_properties') as mock_get_device_props:
            mock_prop = MagicMock()
            mock_prop.total_memory = 1073741824  # Example: 1 GB in bytes
            mock_get_device_props.return_value = mock_prop
            
            result = get_available_memory_gb()
            self.assertEqual(result, 1.0)  # Based on the example total_memory set above
            mock_get_device_props.assert_called_once_with(0)

    @unittest.skipIf(torch.cuda.is_available(), "CUDA available, skipping CPU test")
    def test_get_available_memory_gb_cuda_not_available(self):
        """Test when CUDA is not available (falls back to system memory)."""
        with patch('psutil.virtual_memory') as mock_virtual_mem:
            mock_vm = MagicMock()
            mock_vm.total = 2147483648  # Example: 2 GB in bytes
            mock_virtual_mem.return_value = mock_vm
            
            result = get_available_memory_gb()
            self.assertEqual(result, 2.0)  # Based on the example total set above
            mock_virtual_mem.assert_called_once()

    def test_get_available_memory_gb_edge_cases(self):
        """Test edge cases (zero, negative) - assuming underlying libs don't return these."""
        with patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.get_device_properties') as mock_cuda_props:
            
            mock_prop = MagicMock()

            # Edge case: Zero memory (rounding to nearest MB because of fp variance)
            mock_prop.total_memory = 0
            mock_cuda_props.return_value = mock_prop
            self.assertEqual(round(get_available_memory_gb(), 3), 0.0)

            # Edge case: Negative memory
            mock_prop.total_memory = -1
            mock_cuda_props.return_value = mock_prop
            self.assertEqual(round(get_available_memory_gb(), 3), 0.0)

    def test_get_available_memory_gb_unexpected_exceptions(self):
        """Test that unexpected exceptions are properly handled or propagated."""
        with patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.get_device_properties') as mock_cuda_props:
            
            mock_cuda_props.side_effect = RuntimeError("Mocked unexpected error")
            
            # Expecting the RuntimeError to be raised from within get_available_memory_gb
            with self.assertRaises(RuntimeError, msg="Mocked unexpected error"):
                get_available_memory_gb()

    @patch('psutil.virtual_memory')  
    def test_get_available_memory_gb_happy_path_non_cuda(self, mock_virtual_mem):
        """Happy path test for non-CUDA environment."""
        
        # Mock system memory total to a known value (e.g., 16 GB in bytes)
        mock_vm = MagicMock()
        mock_vm.total = 17179869184  
        mock_virtual_mem.return_value = mock_vm
        
        result = get_available_memory_gb()
        
        # Assert the expected outcome (16 GB, as set above)
        self.assertAlmostEqual(result, 16.0, places=1)  # 'places' for floating point comparison
        mock_virtual_mem.assert_called_once()

    def test_estimate_max_batch_size_valid_input(self):
        """Test with valid, non-edge case inputs."""
        model_size_gb = 2.0
        hidden_size = 1024
        num_layers = 12
        precision_bits = 16
        sequence_length = 512  
        available_memory_gb = 16.0

        expected_max_batch_size = (available_memory_gb - model_size_gb) * (1024 ** 3) // (hidden_size * num_layers * precision_bits / 8)

        max_batch_size = estimate_max_batch_size(model_size_gb, hidden_size, num_layers, precision_bits, sequence_length, available_memory_gb)
        
        self.assertEqual(max_batch_size, expected_max_batch_size)

    def test_estimate_max_batch_size_zero_available_memory(self):
        """Edge case: Zero available memory."""
        max_batch_size = estimate_max_batch_size(model_size_gb=2.0, hidden_size=1024, num_layers=12, precision_bits=16, sequence_length=512, available_memory_gb=0.0)

        self.assertEqual(max_batch_size, 0)

    def test_estimate_max_batch_size_model_larger_than_available_memory(self):
        """Edge case: Model size exceeds available memory."""
        max_batch_size = estimate_max_batch_size(model_size_gb=16.1, hidden_size=1024, num_layers=12, precision_bits=16, sequence_length=512, available_memory_gb=16.0)
        self.assertEqual(max_batch_size, 0)

    def test_estimate_max_batch_size_invalid_inputs(self):
        """Test function handles invalid input types gracefully."""
        with self.assertRaises(TypeError):
            estimate_max_batch_size('a', 1024, 12, 16, 512, 16.0)  # model_size_gb is not a number
        with self.assertRaises(TypeError):
            estimate_max_batch_size(2.0, 'wide', 12, 16, 512, 16.0)  # hidden_size is not a number

    @patch('best_bub.estimate_max_batch_size', return_value=1024)  # Expect max_batch_size = 1024
    @patch('best_bub.get_available_memory_gb', return_value=16)  # Assume 16 GB available
    @patch('best_bub.estimate_model_precision', return_value=16)  # Assume 16-bit precision
    @patch('best_bub.get_model_config', return_value=(1024, 32))  # Assume hidden_size=1024, num_layers=32
    @patch('best_bub.get_model_size_gb', return_value=4)  # Assume 4 GB model size
    def test_initialize_batch_and_model_config(
            self, 
            mock_get_model_size_gb, 
            mock_get_model_config, 
            mock_estimate_model_precision, 
            mock_get_available_memory_gb, 
            mock_estimate_max_batch_size):
        
        # Define kwargs as input to the function
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 512,
            'conform_to_imatrix': False
        }

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = initialize_batch_and_model_config(kwargs)

        # Verify the function returned expected exponent ranges
        self.assertEqual(batch_exponent_range, ExponentRange(min=4, max=10))  # 2^10 = 1024
        self.assertEqual(ubatch_exponent_range, ExponentRange(min=1, max=10))

    @patch('best_bub.estimate_max_batch_size', return_value=1024)  # Expect max_batch_size = 1024
    @patch('best_bub.get_available_memory_gb', return_value=16)  # Assume 16 GB available
    @patch('best_bub.estimate_model_precision', return_value=16)  # Assume 16-bit precision
    @patch('best_bub.get_model_config', return_value=(1024, 32))  # Assume hidden_size=1024, num_layers=32
    @patch('best_bub.get_model_size_gb', return_value=4)  # Assume 4 GB model size
    def test_initialize_batch_and_model_config_conform_to_imatrix(
            self, 
            mock_get_model_size_gb, 
            mock_get_model_config, 
            mock_estimate_model_precision, 
            mock_get_available_memory_gb, 
            mock_estimate_max_batch_size):
        
        # Define kwargs to set conform_to_imatrix to True, with a smaller context size
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 512,  # Smaller than max_batch_size, should limit it
            'conform_to_imatrix': True
        }

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = initialize_batch_and_model_config(kwargs)

        # Verify that batch_exponent_range max is limited by context_size exponent (2^9 = 512)
        self.assertEqual(batch_exponent_range, ExponentRange(min=4, max=9))
        self.assertEqual(ubatch_exponent_range, ExponentRange(min=1, max=9))

        # Ensure mocks were called as expected
        mock_get_model_size_gb.assert_called_once_with(kwargs['model'])
        mock_get_model_config.assert_called_once_with(kwargs['model'])
        mock_estimate_model_precision.assert_called_once_with(kwargs['model'])
        mock_get_available_memory_gb.assert_called_once()
        mock_estimate_max_batch_size.assert_called_once_with(
            4, 1024, 32, 16, kwargs['context_size'], 16
        )


    @patch('best_bub.estimate_max_batch_size', return_value=128)
    @patch('best_bub.get_available_memory_gb', return_value=2)  # Assume 2 GB available
    @patch('best_bub.estimate_model_precision', return_value=8)  # Assume 8-bit precision
    @patch('best_bub.get_model_config', return_value=(512, 8))  # Assume hidden_size=512, num_layers=8
    @patch('best_bub.get_model_size_gb', return_value=1)  # Assume 1 GB model size
    def test_conform_to_imatrix_option(
            self,
            mock_get_model_size_gb,
            mock_get_model_config,
            mock_estimate_model_precision,
            mock_get_available_memory_gb,
            mock_estimate_max_batch_size):
        
        # Define kwargs with a context size smaller than estimated max batch size
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 64,
            'conform_to_imatrix': True
        }
        
        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = initialize_batch_and_model_config(kwargs)
        
        # Check that the exponent ranges are limited by the context size
        self.assertEqual(batch_exponent_range, ExponentRange(min=4, max=6))  # 2^6 = 64 (context_size)
        self.assertEqual(ubatch_exponent_range, ExponentRange(min=1, max=6))

        # Verify max batch size is adjusted to be no larger than context size
        mock_estimate_max_batch_size.assert_called_once_with(
            1, 512, 8, 8, kwargs['context_size'], 2
        )


class TestExecuteTrials(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @contextlib.contextmanager
    def temporary_logger_config(self, logger, level):
        logging.disable(logging.NOTSET)
        
        original_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logging.disable(logging.CRITICAL)
            logger.setLevel(original_level)

    def test_create_trial(self):
        self.assertTrue(callable(create_trial))

    def test_update_best_chunk_time_with_pvalue(self):
        self.assertTrue(callable(update_best_chunk_time_with_probability))

    def test_update_bayesian_mean_variance(self):
        self.assertTrue(callable(update_bayesian_mean_variance))


class TestObjectiveWrapper(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @contextlib.contextmanager
    def temporary_logger_config(self, logger, level):
        logging.disable(logging.NOTSET)
        
        original_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logging.disable(logging.CRITICAL)
            logger.setLevel(original_level)

    def test_objective_wrapper(self):
        self.assertTrue(callable(objective_wrapper))


class TestObjective(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_prepare_llama_args(self):
        self.assertTrue(callable(prepare_llama_args))

    def test_objective(self):
        self.assertTrue(callable(objective))


class TestMainExecution(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_setup_study(self):
        self.assertTrue(callable(setup_study))

    def test_tokenize(self):
        self.assertTrue(callable(tokenize))

    def test_chunk_text(self):
        self.assertTrue(callable(chunk_text))

    def test_execute_trials(self):
        self.assertTrue(callable(execute_trials))

    def test_report_results(self):
        self.assertTrue(callable(report_results))

    def test_main_execution(self):
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
