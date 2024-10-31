import contextlib
import logging
import math
import numpy as np
import optuna
from queue import Empty
import torch
import unittest
from unittest.mock import MagicMock, Mock, patch, call

from mock_model import MockModel
import best_bub


logger = logging.getLogger("best_bub")

class TestInitializeBatchAndModelConfig(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

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
                result = best_bub.get_model_size_gb("dummy/path/model.bin")
                self.assertAlmostEqual(result, expected_gb, places=6)

    @patch('os.path.getsize')
    def test_get_model_size_gb_edge_sizes(self, mock_getsize):
        # Edge cases for model size: zero bytes and very large file size
        mock_getsize.return_value = 0
        result = best_bub.get_model_size_gb("dummy/path/model.bin")
        self.assertEqual(result, 0.0)  # Expecting 0 GB for empty file

        mock_getsize.return_value = 100 * 1024**3  # 100 GB
        result = best_bub.get_model_size_gb("dummy/path/model.bin")
        self.assertEqual(result, 100.0)  # Expecting 100 GB

    @patch('llama_cpp.Llama')
    def test_get_model_config_successful_metadata_extraction(self, mock_llama):
        """Test when the metadata is successfully extracted from gguf."""
        
        # Mock Llama instance and its metadata
        mock_model_instance = MockModel('/path/to/model')
        mock_model_instance.metadata = {
            'llama.embedding_length': 1024,
            'llama.block_count': 8
        }
        mock_llama.return_value = mock_model_instance
        
        # Call the function with a model path
        hidden_size, num_layers, model = best_bub.get_model_config('/path/to/model')
        
        # Assert that the returned values match the expected metadata configuration
        self.assertEqual(hidden_size, 1024)
        self.assertEqual(num_layers, 8)
        self.assertEqual(model, mock_model_instance)

    @patch('llama_cpp.Llama')
    def test_get_model_config_missing_metadata_key(self, mock_llama):
        """Test when a required metadata key is missing from gguf."""
        
        # Mock Llama instance with incomplete metadata
        mock_model_instance = MockModel('/path/to/model')
        mock_model_instance.metadata = {
            'llama.embedding_length': 1024
            # 'llama.block_count' is missing
        }
        mock_llama.return_value = mock_model_instance
        
        # Expect KeyError due to missing metadata
        with self.assertRaises(KeyError) as context:
            best_bub.get_model_config('/path/to/model')
        
        self.assertIn('Required key missing in gguf metadata', str(context.exception))

    @patch('llama_cpp.Llama')
    def test_get_model_config_failed_to_load_metadata(self, mock_llama):
        """Test when the model loading fails due to an exception."""
        
        # Simulate an exception when initializing the model
        mock_llama.side_effect = RuntimeError("Failed to load model")
        
        with self.assertRaises(RuntimeError) as context:
            best_bub.get_model_config('/path/to/model')
        
        self.assertIn("Failed to retrieve model configuration", str(context.exception))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @patch('torch.cuda.get_device_properties')
    def test_get_available_memory_gb_cuda_available(self, mock_get_device_props):
        """Test when CUDA is available."""
        mock_prop = MagicMock()
        mock_prop.total_memory = 1073741824  # Example: 1 GB in bytes
        mock_get_device_props.return_value = mock_prop
        
        result = best_bub.get_available_memory_gb()
        self.assertEqual(result, 1.0)  # Based on the example total_memory set above
        mock_get_device_props.assert_called_once_with(0)

    @unittest.skipIf(torch.cuda.is_available(), "CUDA available, skipping CPU test")
    @patch('psutil.virtual_memory')
    def test_get_available_memory_gb_cuda_not_available(self, mock_virtual_mem):
        """Test when CUDA is not available (falls back to system memory)."""
        mock_vm = MagicMock()
        mock_vm.total = 2147483648  # Example: 2 GB in bytes
        mock_virtual_mem.return_value = mock_vm
        
        result = best_bub.get_available_memory_gb()
        self.assertEqual(result, 2.0)  # Based on the example total set above
        mock_virtual_mem.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_get_available_memory_gb_edge_cases(self, mock_cuda_props, mock_availability):
        """Test edge cases (zero, negative) - assuming underlying libs don't return these."""            
        # Edge case: Zero memory (rounding to nearest MB because of fp variance)
        mock_prop = MagicMock()
        mock_prop.total_memory = 0
        mock_cuda_props.return_value = mock_prop

        self.assertEqual(round(best_bub.get_available_memory_gb(), 3), 0.0)

        # Edge case: Negative memory
        mock_prop.total_memory = -1
        mock_cuda_props.return_value = mock_prop
        self.assertEqual(round(best_bub.get_available_memory_gb(), 3), 0.0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties', side_effect=RuntimeError("Mocked unexpected error"))
    def test_get_available_memory_gb_unexpected_exceptions(self, mock_cuda_props, mock_availability):
        """Test that unexpected exceptions are properly handled or propagated."""
        with self.assertRaises(RuntimeError, msg="Mocked unexpected error"):
            best_bub.get_available_memory_gb()

    @patch('psutil.virtual_memory')  
    def test_get_available_memory_gb_happy_path_non_cuda(self, mock_virtual_mem):
        """Happy path test for non-CUDA environment."""
        # Mock system memory total to a known value (e.g., 16 GB in bytes)
        mock_vm = MagicMock()
        mock_vm.total = 17179869184  
        mock_virtual_mem.return_value = mock_vm
        
        result = best_bub.get_available_memory_gb()
        
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

        max_batch_size = best_bub.estimate_max_batch_size(model_size_gb, hidden_size, num_layers, precision_bits, sequence_length, available_memory_gb)
        
        self.assertEqual(max_batch_size, expected_max_batch_size)

    def test_estimate_max_batch_size_zero_available_memory(self):
        """Edge case: Zero available memory."""
        max_batch_size = best_bub.estimate_max_batch_size(model_size_gb=2.0, hidden_size=1024, num_layers=12, precision_bits=16, sequence_length=512, available_memory_gb=0.0)

        self.assertEqual(max_batch_size, 0)

    def test_estimate_max_batch_size_model_larger_than_available_memory(self):
        """Edge case: Model size exceeds available memory."""
        max_batch_size = best_bub.estimate_max_batch_size(model_size_gb=16.1, hidden_size=1024, num_layers=12, precision_bits=16, sequence_length=512, available_memory_gb=16.0)
        self.assertEqual(max_batch_size, 0)

    def test_estimate_max_batch_size_invalid_inputs(self):
        """Test function handles invalid input types gracefully."""
        with self.assertRaises(TypeError):
            best_bub.estimate_max_batch_size('a', 1024, 12, 16, 512, 16.0)  # model_size_gb is not a number
        with self.assertRaises(TypeError):
            best_bub.estimate_max_batch_size(2.0, 'wide', 12, 16, 512, 16.0)  # hidden_size is not a number

    def test_estimate_max_batch_size_min_precision(self):
        model_size_gb = 2.0
        hidden_size = 1024
        num_layers = 12
        precision_bits = 1  # Minimal precision bits
        sequence_length = 512
        available_memory_gb = 16.0

        # Expected max batch size based on smallest precision bit usage
        expected_max_batch_size = (available_memory_gb - model_size_gb) * (1024 ** 3) // (hidden_size * num_layers * precision_bits / 8)
        
        max_batch_size = best_bub.estimate_max_batch_size(model_size_gb, hidden_size, num_layers, precision_bits, sequence_length, available_memory_gb)
        self.assertEqual(max_batch_size, expected_max_batch_size)

    @patch('best_bub.estimate_max_batch_size', return_value=2**10)  # Expect max_batch_size = 1024
    @patch('best_bub.get_available_memory_gb', return_value=16)  # Assume 16 GB available
    @patch('best_bub.estimate_model_precision', return_value=16)  # Assume 16-bit precision
    @patch('best_bub.get_model_config', return_value=(1024, 32, None))  # Assume hidden_size=1024, num_layers=32
    @patch('best_bub.get_model_size_gb', return_value=4)  # Assume 4 GB model size
    def test_initialize_batch_and_model_config_lower_context(
            self, 
            mock_get_model_size_gb, 
            mock_get_model_config, 
            mock_estimate_model_precision, 
            mock_get_available_memory_gb, 
            mock_estimate_max_batch_size):
        
        # Define kwargs as input to the function
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 2**9,
            'conform_to_imatrix': False
        }

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = best_bub.initialize_batch_and_model_config(kwargs)

        # Verify the function returned expected exponent ranges
        self.assertEqual(batch_exponent_range, best_bub.ExponentRange(min=4, max=9))
        self.assertEqual(ubatch_exponent_range, best_bub.ExponentRange(min=1, max=9))

    @patch('best_bub.estimate_max_batch_size', return_value=2**9)  # Expect max_batch_size = 1024
    @patch('best_bub.get_available_memory_gb', return_value=16)  # Assume 16 GB available
    @patch('best_bub.estimate_model_precision', return_value=16)  # Assume 16-bit precision
    @patch('best_bub.get_model_config', return_value=(1024, 32, None))  # Assume hidden_size=1024, num_layers=32
    @patch('best_bub.get_model_size_gb', return_value=4)  # Assume 4 GB model size
    def test_initialize_batch_and_model_config_lower_max_batch(
            self, 
            mock_get_model_size_gb, 
            mock_get_model_config, 
            mock_estimate_model_precision, 
            mock_get_available_memory_gb, 
            mock_estimate_max_batch_size):
        
        # Define kwargs as input to the function
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 2**10,
            'conform_to_imatrix': False
        }

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = best_bub.initialize_batch_and_model_config(kwargs)

        # Verify the function returned expected exponent ranges
        self.assertEqual(batch_exponent_range, best_bub.ExponentRange(min=4, max=9))
        self.assertEqual(ubatch_exponent_range, best_bub.ExponentRange(min=1, max=9))

    @patch('best_bub.estimate_max_batch_size', return_value=2**10)  # Expect max_batch_size = 1024
    @patch('best_bub.get_available_memory_gb', return_value=16)  # Assume 16 GB available
    @patch('best_bub.estimate_model_precision', return_value=16)  # Assume 16-bit precision
    @patch('best_bub.get_model_config', return_value=(1024, 32, None))  # Assume hidden_size=1024, num_layers=32
    @patch('best_bub.get_model_size_gb', return_value=4)  # Assume 4 GB model size
    def test_initialize_batch_and_model_config_equal_max_batch_and_context(
            self, 
            mock_get_model_size_gb, 
            mock_get_model_config, 
            mock_estimate_model_precision, 
            mock_get_available_memory_gb, 
            mock_estimate_max_batch_size):
        
        # Define kwargs as input to the function
        kwargs = {
            'model': 'dummy_model_path',
            'context_size': 2**10,
            'conform_to_imatrix': False
        }

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = best_bub.initialize_batch_and_model_config(kwargs)

        # Verify the function returned expected exponent ranges
        self.assertEqual(batch_exponent_range, best_bub.ExponentRange(min=4, max=10))
        self.assertEqual(ubatch_exponent_range, best_bub.ExponentRange(min=1, max=10))

    @patch('llama_cpp.Llama')  # Replace llama_cpp.Llama directly with MockModel
    @patch('best_bub.get_model_size_gb', return_value=4)
    @patch('best_bub.get_available_memory_gb', return_value=16)
    @patch('best_bub.estimate_model_precision', return_value=16)
    def test_initialize_batch_with_mock_model_metadata_conformant_to_imatrix(
        self, 
        mock_estimate_model_precision,
        mock_get_available_memory_gb, 
        mock_get_model_size_gb, 
        MockLlama
    ):
        # Define test arguments
        kwargs = {
            'model': 'dummy model',
            'context_size': 2**9,  # Set context size as needed for test
            'conform_to_imatrix': True
        }

        # Create and configure a mock model instance
        mock_model_instance = MockModel('dummy model path')
        mock_model_instance.metadata['llama.embedding_length'] = 512
        mock_model_instance.metadata['llama.block_count'] = 8
        mock_model_instance.n_ctx = 2**7  # Model context size
        MockLlama.return_value = mock_model_instance

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = best_bub.initialize_batch_and_model_config(kwargs)

        # Assertions for batch and ubatch exponent ranges
        self.assertEqual(batch_exponent_range.min, 4)
        self.assertEqual(batch_exponent_range.max, 9)
        self.assertEqual(ubatch_exponent_range.min, 1)
        self.assertEqual(ubatch_exponent_range.max, 7)

        # Additional assertions to confirm correct usage of MockModel
        self.assertIsInstance(batch_exponent_range, best_bub.ExponentRange)
        self.assertIsInstance(ubatch_exponent_range, best_bub.ExponentRange)

    @patch('llama_cpp.Llama')
    @patch('best_bub.estimate_max_batch_size', return_value=128)
    @patch('best_bub.get_available_memory_gb', return_value=2)  # Assume 2 GB available
    @patch('best_bub.estimate_model_precision', return_value=8)  # Assume 8-bit precision
    @patch('best_bub.get_model_size_gb', return_value=1)  # Assume 1 GB model size
    def test_initialize_batch_conform_to_imatrix_option(
            self,
            mock_get_model_size_gb,
            mock_estimate_model_precision,
            mock_get_available_memory_gb,
            mock_estimate_max_batch_size,
            MockLlama):
        
        # Define kwargs with a context size smaller than estimated max batch size
        kwargs = {
            'model': 'dummy model path',
            'context_size': 2**6, # smaller than model context 2^7
            'conform_to_imatrix': True
        }

        # Create and configure a mock model instance
        mock_model_instance = MockModel('dummy model path')
        mock_model_instance.metadata['llama.embedding_length'] = 512
        mock_model_instance.metadata['llama.block_count'] = 8
        mock_model_instance.n_ctx = 2**7  # Model context size
        MockLlama.return_value = mock_model_instance

        # Call the function under test
        batch_exponent_range, ubatch_exponent_range = best_bub.initialize_batch_and_model_config(kwargs)
        
        # Check that the exponent ranges are limited by the context size
        self.assertEqual(batch_exponent_range, best_bub.ExponentRange(min=4, max=6))  # 2^6 = 64 (context_size)
        self.assertEqual(ubatch_exponent_range, best_bub.ExponentRange(min=1, max=6))

        # Verify max batch size is adjusted to be no larger than context size
        mock_estimate_max_batch_size.assert_called_once_with(
            1, 512, 8, 8, kwargs['context_size'], 2
        )


class TestObjectiveWrapper(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        best_bub.trial_cache.clear()

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_caching_of_results(self, mock_process, mock_queue):
        # Configure trial mock with proper pruning behavior
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1
        trial.should_prune.return_value = False  # Important: prevent pruning
        
        # Configure process mock
        mock_process.return_value.is_alive.side_effect = [True, True, True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None

        # Configure queue mock
        mock_queue_instance = mock_queue.return_value
        queue_results = [(1, 100), (2, 95), ("done", None)]
        
        def queue_side_effect():
            if queue_results:
                return queue_results.pop(0)
            raise Empty()
        
        mock_queue_instance.empty.side_effect = [False, False, False, True]
        mock_queue_instance.get_nowait.side_effect = queue_side_effect

        # Run objective_wrapper to cache the result
        result_1 = best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)
        
        # Verify the first run
        self.assertEqual(result_1, [100, 95])
        self.assertIn((1024, 512), best_bub.trial_cache)
        self.assertEqual(best_bub.trial_cache[(1024, 512)]['result'], result_1)
        self.assertEqual(best_bub.trial_cache[(1024, 512)]['read_count'], 0)

        # Verify trial interactions
        trial.report.assert_any_call(100, step=1)
        trial.report.assert_any_call(95, step=2)
        
        # Reset mocks for second call
        mock_process.reset_mock()
        mock_queue.reset_mock()
        trial.reset_mock()

        # Run objective_wrapper again, should use cached result
        result_2 = best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)

        # Verify the second result uses cache
        self.assertEqual(result_1, result_2)
        self.assertEqual(best_bub.trial_cache[(1024, 512)]['read_count'], 1)
        
        # Verify process wasn't started for cached result
        mock_process.return_value.start.assert_not_called()
        # Verify trial wasn't called for cached result
        trial.report.assert_not_called()
        trial.should_prune.assert_not_called()

    @patch("best_bub.trial_cache", {})
    def test_error_handling_for_cached_exceptions(self):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1

        # Cache an exception result
        test_error = ValueError("Cached error")
        best_bub.trial_cache[(1024, 512)] = {
            'result': ('exception', test_error),
            'read_count': 0
        }

        with self.assertRaises(ValueError) as context:
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)
            
        self.assertEqual(str(context.exception), "Cached error")
        self.assertEqual(best_bub.trial_cache[(1024, 512)]['read_count'], 1)

    @patch('best_bub.time.time')
    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_trial_execution_with_timeout(self, mock_process, mock_queue, mock_time):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1

        
        # Configure process to stay alive longer than the timeout
        def mock_objective():
            while True:
                yield True
        mock_process.return_value.is_alive.side_effect = mock_objective  # Multiple True responses to ensure timeout
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None
        def mock_is_alive():
            return True
        mock_process.return_value.is_alive.side_effect = mock_is_alive  # Prints debug and returns True

        # Configure queue to simulate slow processing
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.get_nowait.side_effect = Empty()

        # Set a very small timeout to ensure we hit it
        # Simulate time passing between checks
        def time_generator():
            yield 0  # Initial start time
            while True:
                yield 2  # Always return 2 to simulate a timeout condition
        mock_time.side_effect = time_generator()
        
        with self.assertRaises(optuna.TrialPruned):
            b = best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=1)
        
        # Verify process was terminated due to timeout
        mock_process.return_value.terminate.assert_called_once()
        mock_process.return_value.join.assert_called()

    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_result_reporting_and_intermediate_pruning(self, mock_process, mock_queue):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1
        
        # Configure process mock
        mock_process.return_value.is_alive.side_effect = [True, True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None

        # Configure queue to return one result
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.side_effect = [False, True]
        mock_queue_instance.get_nowait.side_effect = [(1, 100), Empty()]

        # Configure trial to trigger pruning after first report
        trial.should_prune.side_effect = [True]  # Return True on first call
        
        with self.assertRaises(optuna.TrialPruned):
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)

        # Verify process was terminated due to pruning
        mock_process.return_value.terminate.assert_called_once()
        trial.report.assert_called_once_with(100, step=1)
        trial.should_prune.assert_called_once()

    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_empty_result_handling(self, mock_process, mock_queue):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1
        trial.should_prune.return_value = False

        # Configure process to complete without producing results
        mock_process.return_value.is_alive.side_effect = [True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None

        # Configure queue to never return results
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.get_nowait.side_effect = Empty()

        # Process finishes without producing results
        with self.assertRaises(optuna.TrialPruned):
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)

    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_runtime_error_handling(self, mock_process, mock_queue):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1
        
        # Test OOM error
        mock_process.return_value.is_alive.side_effect = [True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None
        
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.side_effect = [False, True]
        mock_queue_instance.get_nowait.side_effect = [RuntimeError("CUDA out of memory")]
        
        with self.assertRaises(optuna.TrialPruned) as context:
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)
        self.assertEqual(str(context.exception), "OOM")
        
        # Verify OOM was cached
        self.assertIn((1024, 512), best_bub.trial_cache)
        cached_result = best_bub.trial_cache[(1024, 512)]
        self.assertEqual(cached_result['result'][0], 'exception')
        self.assertTrue(isinstance(cached_result['result'][1], optuna.TrialPruned))
        
        # Test other runtime error
        mock_process.reset_mock()
        mock_queue.reset_mock()
        best_bub.trial_cache.clear()
        
        mock_process.return_value.is_alive.side_effect = [True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None
        
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.side_effect = [False, True]
        mock_queue_instance.get_nowait.side_effect = [RuntimeError("Other runtime error")]
        
        with self.assertRaises(RuntimeError) as context:
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)
        self.assertEqual(str(context.exception), "Other runtime error")
        
        # Verify runtime error was cached
        self.assertIn((1024, 512), best_bub.trial_cache)
        cached_result = best_bub.trial_cache[(1024, 512)]
        self.assertEqual(cached_result['result'][0], 'exception')
        self.assertTrue(isinstance(cached_result['result'][1], RuntimeError))

    @patch("best_bub.mp.Queue")
    @patch("best_bub.mp.Process")
    def test_unexpected_exception_handling(self, mock_process, mock_queue):
        trial = MagicMock()
        trial.params = {'n_batch': 1024, 'n_ubatch': 512}
        trial.user_attrs = {}
        trial.number = 1
        
        mock_process.return_value.is_alive.side_effect = [True, False]
        mock_process.return_value.start.return_value = None
        mock_process.return_value.join.return_value = None
        
        mock_queue_instance = mock_queue.return_value
        mock_queue_instance.empty.side_effect = [False, True]
        mock_queue_instance.get_nowait.side_effect = [ValueError("Unexpected error")]
        
        with self.assertRaises(ValueError) as context:
            best_bub.objective_wrapper(trial, pre_chunked_text=None, kwargs={}, best_chunk_time=None)
        self.assertEqual(str(context.exception), "Unexpected error")
        
        # Verify unexpected exception was cached
        self.assertIn((1024, 512), best_bub.trial_cache)
        cached_result = best_bub.trial_cache[(1024, 512)]
        self.assertEqual(cached_result['result'][0], 'exception')
        self.assertTrue(isinstance(cached_result['result'][1], ValueError))


class TestObjective(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @patch('best_bub.time.time')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective(self, mock_prepare_llama_args, mock_Llama, mock_time):
        # Set up mocks
        mock_prepare_llama_args.return_value = {'n_batch': 1024, 'n_ubatch': 512}
        mock_model_instance = MagicMock()
        mock_Llama.return_value = mock_model_instance

        # Simulate model(chunk) call
        def mock_model_call(chunk):
            return "processed " + chunk

        mock_model_instance.side_effect = mock_model_call

        # Set up inputs
        queue = MagicMock()
        pre_chunked_text = ["chunk1", "chunk2"]
        kwargs = {'chunks': 2}
        n_batch = 1024
        n_ubatch = 512
        best_chunk_time = None

        # Simulate time.time() returning increasing values
        mock_time.side_effect = [0, 0.1, 0.2, 0.3]

        # Run objective
        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time)

        # Define expected calls for comparison
        expected_calls = [
            (0, 100.0),
            (1, 100.0),
            ("done", 100.0)
        ]

        # Retrieve actual calls from the queue
        actual_calls = [args[0] for args, _ in queue.put.call_args_list]

        # Assert that each actual call approximately matches the expected call
        for actual, expected in zip(actual_calls, expected_calls):
            self.assertEqual(actual[0], expected[0])  # Chunk number or "done" should match exactly
            if isinstance(actual[1], float) and isinstance(expected[1], float):
                self.assertAlmostEqual(actual[1], expected[1], places=1)  # Allow small differences in float comparison

        # Verify that model was called with correct chunks
        mock_model_instance.assert_has_calls([call("chunk1"), call("chunk2")], any_order=False)

    @patch('best_bub.time.time', side_effect=[0, 0.1])
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_exceed_best_chunk_time(self, mock_prepare_llama_args, mock_Llama, mock_time):
        # Set up mocks
        mock_prepare_llama_args.return_value = {'n_batch': 1024, 'n_ubatch': 512}
        mock_model_instance = MagicMock()
        mock_Llama.return_value = mock_model_instance

        # Simulate model(chunk) call
        def mock_model_call(chunk):
            return "processed " + chunk

        mock_model_instance.side_effect = mock_model_call

        # Set up inputs
        queue = MagicMock()
        pre_chunked_text = ["chunk1", "chunk2"]
        kwargs = {'chunks': 2}
        n_batch = 1024
        n_ubatch = 512
        best_chunk_time = 50  # milliseconds

        # Run objective
        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time)

        # Check that the queue received the RuntimeError
        queue.put.assert_called_once()
        args, _ = queue.put.call_args
        self.assertIsInstance(args[0], RuntimeError)
        self.assertEqual(str(args[0]), "Chunk time exceeded best_chunk_time")

        # Verify that model was called once (since it should exit after exceeding best_chunk_time)
        mock_model_instance.assert_called_once_with("chunk1")

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_exception_handling(self, mock_prepare_llama_args, mock_Llama):
        # Set up mocks
        mock_prepare_llama_args.return_value = {'n_batch': 1024, 'n_ubatch': 512}
        mock_model_instance = MagicMock()
        mock_Llama.return_value = mock_model_instance

        # Simulate model(chunk) raising an exception
        def mock_model_call(chunk):
            raise ValueError("Test exception")

        mock_model_instance.side_effect = mock_model_call

        # Set up inputs
        queue = MagicMock()
        pre_chunked_text = ["chunk1", "chunk2"]
        kwargs = {'chunks': 2}
        n_batch = 1024
        n_ubatch = 512
        best_chunk_time = None

        # Run objective
        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time)

        # Check that the queue received the exception
        queue.put.assert_called_once()
        args, _ = queue.put.call_args
        self.assertIsInstance(args[0], ValueError)
        self.assertEqual(str(args[0]), "Test exception")

        # Verify that model was called once (since exception occurred during first chunk)
        mock_model_instance.assert_called_once_with("chunk1")

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_oom_handling(self, mock_prepare_llama_args, mock_Llama):
        # Set up mocks
        mock_prepare_llama_args.return_value = {'n_batch': 1024, 'n_ubatch': 512}
        mock_model_instance = MagicMock()
        mock_Llama.return_value = mock_model_instance

        # Simulate model(chunk) raising an OOM exception
        def mock_model_call(chunk):
            raise RuntimeError("CUDA out of memory")

        mock_model_instance.side_effect = mock_model_call

        # Set up inputs
        queue = MagicMock()
        pre_chunked_text = ["chunk1", "chunk2"]
        kwargs = {'chunks': 2}
        n_batch = 1024
        n_ubatch = 512
        best_chunk_time = None

        # Run objective
        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time)

        # Check that the queue received RuntimeError("OOM")
        queue.put.assert_called_once()
        args, _ = queue.put.call_args
        self.assertIsInstance(args[0], RuntimeError)
        self.assertEqual(str(args[0]), "OOM")

        # Verify that model was called once
        mock_model_instance.assert_called_once_with("chunk1")

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_model_initialization_failure(self, mock_prepare_llama_args, mock_Llama):
        # Set up mocks
        mock_prepare_llama_args.return_value = {'n_batch': 1024, 'n_ubatch': 512}
        mock_Llama.side_effect = Exception("Initialization failed")

        # Set up inputs
        queue = MagicMock()
        pre_chunked_text = ["chunk1", "chunk2"]
        kwargs = {'chunks': 2}
        n_batch = 1024
        n_ubatch = 512
        best_chunk_time = None

        # Run objective
        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch, best_chunk_time)

        # Check that the queue received the exception
        queue.put.assert_called_once()
        args, _ = queue.put.call_args
        self.assertIsInstance(args[0], Exception)
        self.assertEqual(str(args[0]), "Initialization failed")

        # Verify that model initialization was attempted
        mock_Llama.assert_called_once_with(n_batch=1024, n_ubatch=512)

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_empty_input(self, mock_prepare_llama_args, mock_Llama):
        """Test behavior with empty input text"""
        queue = MagicMock()
        pre_chunked_text = []
        kwargs = {'chunks': 0}
        n_batch = 1024
        n_ubatch = 512

        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch)

        # Check that 'done' with None as time was put in the queue once
        queue.put.assert_called_once_with(("done", None))

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('builtins.open')
    def test_objective_file_redirection_failure(self, mock_open, mock_prepare_llama_args, mock_Llama):
        """Test handling of file redirection failures"""
        # Cause open to raise an IOError
        mock_open.side_effect = IOError("File operation failed")
        queue = MagicMock()
        pre_chunked_text = ["chunk1"]
        kwargs = {'chunks': 1}
        n_batch = 1024
        n_ubatch = 512

        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch)

        # Verify the IOError was sent to the queue
        queue.put.assert_called_once()
        args, _ = queue.put.call_args
        self.assertIsInstance(args[0], IOError)
        self.assertEqual(str(args[0]), "File operation failed")

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    def test_objective_queue_failure(self, mock_prepare_llama_args, mock_Llama):
        """Test handling of queue.put failures"""
        queue = MagicMock()
        queue.put.side_effect = Exception("Queue operation failed")
        pre_chunked_text = ["chunk1"]
        kwargs = {'chunks': 1}
        n_batch = 1024
        n_ubatch = 512

        with self.assertRaises(Exception) as context:
            best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch)

        self.assertEqual(str(context.exception), "Queue operation failed")

    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.time.time', side_effect=[0, float('inf')])
    def test_objective_extreme_timing(self, mock_prepare_llama_args, mock_Llama, mock_time):
        """Test handling of extreme timing scenarios"""
        queue = MagicMock()
        pre_chunked_text = ["chunk1"]
        kwargs = {'chunks': 1}
        n_batch = 1024
        n_ubatch = 512

        best_bub.objective(queue, pre_chunked_text, kwargs, n_batch, n_ubatch)

        # Verify that extreme timing entries were put in the queue
        expected_calls = [
            call((0, float('inf'))),
            call(("done", float('inf')))
        ]
        queue.put.assert_has_calls(expected_calls, any_order=False)


class TestExecuteTrials(unittest.TestCase):
    def setUp(self):
        best_bub.trial_cache.clear()

        best_bub.near_best_trials = []
        best_bub.prior_mean = None
        best_bub.prior_variance = None

        # Optionally, reset the logger's handlers to avoid duplicate logs in tests
        best_bub.logger.handlers = []
        best_bub.logger.addHandler(logging.NullHandler())

        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_prepare_llama_args_fixed_properties(self):
        """Test for properties that should remain unchanged"""
        kwargs = {
            'context_size': 1024,
            'n_gpu_layers': 4,
            'temp': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'min_p': 0.1,
            'repeat_last_n': 100,
            'repeat_penalty': 1.2,
            'presence_penalty': 0.5,
            'frequency_penalty': 0.4,
            'seed': 42,
        }
        expected = {
            'n_ctx': 1024,
            'n_gpu_layers': 4,
            'temp': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'min_p': 0.1,
            'repeat_last_n': 100,
            'repeat_penalty': 1.2,
            'presence_penalty': 0.5,
            'frequency_penalty': 0.4,
            'seed': 42,
        }
        result = best_bub.prepare_llama_args(kwargs)
        self.assertEqual(result, expected)

    def test_prepare_llama_args_dynamic_properties(self):
        """Test for properties that are optional or dynamic"""
        kwargs = {
            'model': 'path/to/model',
            'context_size': 512,
            'dynatemp_range': [0.2, 0.8],
            'dynatemp_exp': 1.5,
            'mirostat': 1,
            'mirostat_lr': 0.01,
            'mirostat_ent': 4.5,
            'threads': 8
        }
        expected = {
            'model_path': 'path/to/model',
            'n_ctx': 512,
            'dynatemp_range': [0.2, 0.8],
            'dynatemp_exp': 1.5,
            'mirostat': 1,
            'mirostat_lr': 0.01,
            'mirostat_ent': 4.5,
            'n_threads': 8
        }
        result = best_bub.prepare_llama_args(kwargs)
        self.assertEqual(result, expected)

    def test_prepare_llama_args_with_batch_and_ubatch(self):
        """Test when both n_batch and n_ubatch are provided"""
        kwargs = {
            'model': 'path/to/model',
            'context_size': 1024,
            'n_batch': 32,
            'n_ubatch': 16
        }
        expected = {
            'model_path': 'path/to/model',
            'n_ctx': 1024,
            'n_batch': 32,
            'n_ubatch': 16
        }
        result = best_bub.prepare_llama_args(kwargs)
        self.assertEqual(result, expected)

    def test_prepare_llama_args_missing_optional_arguments(self):
        kwargs = {
            'model': 'path/to/model',
            'context_size': 1024,
            'temp': 0.7,
            'seed': 42
        }
        expected = {
            'model_path': 'path/to/model',
            'n_ctx': 1024,
            'temp': 0.7,
            'seed': 42
        }

        result = best_bub.prepare_llama_args(kwargs)
        self.assertEqual(result, expected)

    def test_prepare_llama_args_removes_extraneous(self):
        """Test when both n_batch and n_ubatch are provided"""
        kwargs = {
            'model': 'path/to/model',
            'context_size': 1024,
            'n_batch': 32,
            'n_ubatch': 16,
            'removable_agrument': True
        }
        expected = {
            'model_path': 'path/to/model',
            'n_ctx': 1024,
            'n_batch': 32,
            'n_ubatch': 16
        }
        result = best_bub.prepare_llama_args(kwargs)
        self.assertEqual(result, expected)

    def test_create_trial_with_defaults(self):
        """Test create_trial when default batch sizes are provided"""
        # Setup
        study = Mock()
        study.ask.return_value = Mock()
        batch_range = Mock(min=1, max=5)
        ubatch_range = Mock(min=1, max=4)
        
        # Test with valid defaults
        result = best_bub.create_trial(
            study=study,
            batch_exponent_range=batch_range,
            ubatch_exponent_range=ubatch_range,
            default_n_batch=32,
            default_n_ubatch=16
        )
        
        # Assertions
        study.enqueue_trial.assert_called_once_with(
            params={'n_batch': 32, 'n_ubatch': 16},
            user_attrs={'n_batch': 32, 'n_ubatch': 16}
        )
        assert study.ask.called
        assert result == study.ask.return_value

    def test_create_trial_without_defaults(self):
        """Test create_trial when suggesting new values"""
        # Setup
        study = Mock()
        trial = Mock()
        study.ask.return_value = trial
        batch_range = Mock(min=2, max=4)
        ubatch_range = Mock(min=1, max=3)
        
        # Configure trial to suggest valid values (2^3=8 and 2^2=4)
        trial.suggest_int.side_effect = [3, 2]
        
        # Test
        result = best_bub.create_trial(
            study=study,
            batch_exponent_range=batch_range,
            ubatch_exponent_range=ubatch_range
        )
        
        # Assertions
        assert trial.suggest_int.call_count == 2
        trial.set_user_attr.assert_any_call('n_batch', 8)
        trial.set_user_attr.assert_any_call('n_ubatch', 4)
        assert result == trial

    def test_create_trial_handles_invalid_batch_sizes(self):
        """Test create_trial when batch sizes are not divisible"""
        # Setup
        study = Mock()
        first_trial = Mock()
        second_trial = Mock()
        study.ask.side_effect = [first_trial, second_trial]
        batch_range = Mock(min=3, max=5)
        ubatch_range = Mock(min=1, max=4)
        
        # Configure trials:
        first_trial.suggest_int.side_effect = [3, 4] # First trial: 2^3=8 and 2^4=16 (invalid combination)
        second_trial.suggest_int.side_effect = [3, 2] # Second trial: 2^3=8 and 2^2=4 (valid combination)
        
        # Test
        result = best_bub.create_trial(
            study=study,
            batch_exponent_range=batch_range,
            ubatch_exponent_range=ubatch_range
        )
        
        # Assertions
        study.tell.assert_called_once_with(first_trial, state=optuna.trial.TrialState.PRUNED)
        assert study.ask.call_count == 2
        assert result == second_trial

    def test_create_trial_edge_cases(self):
        """Test create_trial with edge cases"""
        # Setup
        study = Mock()
        trial = Mock()
        study.ask.return_value = trial
        batch_range = Mock(min=1, max=1)
        ubatch_range = Mock(min=1, max=1)
        
        # Test with minimum possible values
        trial.suggest_int.side_effect = [1, 1]
        
        result = best_bub.create_trial(
            study=study,
            batch_exponent_range=batch_range,
            ubatch_exponent_range=ubatch_range
        )
        
        # Assertions
        assert trial.set_user_attr.call_count == 2
        trial.set_user_attr.assert_any_call('n_batch', 2)
        trial.set_user_attr.assert_any_call('n_ubatch', 2)

    def test_create_trial_default_validation(self):
        test_cases = [
            (32, 16, True),    # Valid: 32 is divisible by 16
            (32, 7, False),    # Invalid: 32 is not divisible by 7
            (64, 32, True),    # Valid: 64 is divisible by 32
            (128, 10, False),  # Invalid: 128 is not divisible by 10
        ]

        for default_batch, default_ubatch, expected_valid in test_cases:
            with self.subTest(default_batch=default_batch, default_ubatch=default_ubatch, expected_valid=expected_valid):
                # Setup mocks
                study = Mock()
                study.ask.return_value = Mock()
                batch_range = Mock(min=1, max=5)
                ubatch_range = Mock(min=1, max=4)

                if not expected_valid:
                    # Expecting a ValueError for invalid configurations
                    with self.assertRaises(ValueError):
                        best_bub.create_trial(
                            study=study,
                            batch_exponent_range=batch_range,
                            ubatch_exponent_range=ubatch_range,
                            default_n_batch=default_batch,
                            default_n_ubatch=default_ubatch
                        )
                else:
                    # Expecting successful creation for valid configurations
                    result = best_bub.create_trial(
                        study=study,
                        batch_exponent_range=batch_range,
                        ubatch_exponent_range=ubatch_range,
                        default_n_batch=default_batch,
                        default_n_ubatch=default_ubatch
                    )
                    self.assertEqual(result, study.ask.return_value)

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_trial_better_with_high_confidence(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [90, 95, 92]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [100, 105, 102]
        best_batch = 64
        best_ubatch = 16

        # Mock the functions to return desired values
        mock_update_bayesian.return_value = (93, 1.5)  # posterior_mean, posterior_variance
        mock_calculate_prob.return_value = 0.96  # Greater than PROBABILITY_THRESHOLD (0.95)

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertEqual(updated_best_chunk_times, trial_chunk_times)
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)
        self.assertEqual(best_bub.near_best_trials, [])  # Should be cleared

        # Ensure the internal functions were called
        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_trial_better_with_low_confidence(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [99, 101, 100]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [100, 105, 102]
        best_batch = 64
        best_ubatch = 16

        # Mock the functions to return desired values
        mock_update_bayesian.return_value = (100, 2)
        mock_calculate_prob.return_value = 0.90  # Below PROBABILITY_THRESHOLD

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertEqual(updated_best_chunk_times, best_chunk_times)  # Should not update
        self.assertEqual(updated_best_batch, best_batch)
        self.assertEqual(updated_best_ubatch, best_ubatch)
        self.assertEqual(len(best_bub.near_best_trials), 1)  # Should add to near_best_trials

        trial_entry = best_bub.near_best_trials[0]
        self.assertEqual(trial_entry['chunk_time'], np.mean(trial_chunk_times))
        self.assertEqual(trial_entry['params'], {'n_batch': n_batch, 'n_ubatch': n_ubatch})
        self.assertEqual(trial_entry['prob_superiority'], 0.90)
        self.assertEqual(trial_entry['p_value'], 0.90)

        # Ensure the internal functions were called
        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_trial_worse_than_best(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [110, 115, 112]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [100, 105, 102]
        best_batch = 64
        best_ubatch = 16

        mock_update_bayesian.return_value = (112, 3)
        mock_calculate_prob.return_value = 0.05  # Low probability of superiority

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertEqual(updated_best_chunk_times, best_chunk_times)
        self.assertEqual(updated_best_batch, best_batch)
        self.assertEqual(updated_best_ubatch, best_ubatch)
        self.assertEqual(len(best_bub.near_best_trials), 1)

        trial_entry = best_bub.near_best_trials[0]
        self.assertEqual(trial_entry['chunk_time'], np.mean(trial_chunk_times))
        self.assertEqual(trial_entry['params'], {'n_batch': n_batch, 'n_ubatch': n_ubatch})
        self.assertEqual(trial_entry['prob_superiority'], 0.05)
        self.assertEqual(trial_entry['p_value'], 0.05)

        # Ensure the internal functions were called
        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_empty_trial_chunk_times(self,mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = []
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [100, 105, 102]
        best_batch = 64
        best_ubatch = 16

        mock_update_bayesian.return_value = (float('inf'), 1e6)
        mock_calculate_prob.return_value = 0.0

        # When
        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        # Then
        self.assertEqual(updated_best_chunk_times, best_chunk_times)
        self.assertEqual(updated_best_batch, best_batch)
        self.assertEqual(updated_best_ubatch, best_ubatch)
        self.assertEqual(len(best_bub.near_best_trials), 1)

        trial_entry = best_bub.near_best_trials[0]
        self.assertEqual(trial_entry['chunk_time'], float('inf'))
        self.assertEqual(trial_entry['params'], {'n_batch': n_batch, 'n_ubatch': n_ubatch})
        self.assertEqual(trial_entry['prob_superiority'], 0.0)
        self.assertEqual(trial_entry['p_value'], 0.0)

        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_empty_best_chunk_times(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [100, 105, 102]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = []
        best_batch = None
        best_ubatch = None

        # Mock values to avoid irrelevant calls influencing the test outcome
        mock_update_bayesian.return_value = (102.33, 2.11)
        mock_calculate_prob.return_value = 0.95

        # When
        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        # Then
        self.assertEqual(updated_best_chunk_times, trial_chunk_times)  # Should update to trial data
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)
        self.assertEqual(best_bub.near_best_trials, [])  # Ensure near_best_trials remains empty

        # Assertions on the mocked function calls
        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called()  # Adjust based on observed behavior, if required

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_variance_zero_or_negative(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [100, 100, 100]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [105, 105, 105]
        best_batch = 64
        best_ubatch = 16

        # Set up the mock returns
        mock_update_bayesian.return_value = (100, 0.0)  # This should trigger a zero variance warning
        mock_calculate_prob.return_value = 0.97

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        # Standard assertions
        self.assertEqual(updated_best_chunk_times, trial_chunk_times)
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)
        self.assertEqual(best_bub.near_best_trials, [])

        # Ensure the mocked functions were called as expected
        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_prior_mean_variance_initialization(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = [100, 105, 102]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [110, 115, 112]
        best_batch = 64
        best_ubatch = 16

        # Mock initial prior as None
        best_bub.prior_mean = None
        best_bub.prior_variance = None

        mock_update_bayesian.return_value = (102.33, 1.055)
        mock_calculate_prob.return_value = 0.97

        # When
        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        # Then
        self.assertIsNotNone(best_bub.prior_mean)
        self.assertIsNotNone(best_bub.prior_variance)
        self.assertEqual(best_bub.prior_mean, 102.33)
        self.assertEqual(best_bub.prior_variance, 1.055)
        self.assertEqual(updated_best_chunk_times, trial_chunk_times)
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)

        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_prior_mean_variance_not_none(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        best_bub.prior_mean = 100
        best_bub.prior_variance = 10
        trial_chunk_times = [90, 95, 92]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [100, 105, 102]
        best_batch = 64
        best_ubatch = 16

        mock_update_bayesian.return_value = (93.07, 4.5)
        mock_calculate_prob.return_value = 0.98

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertAlmostEqual(best_bub.prior_mean, 93.07)
        self.assertAlmostEqual(best_bub.prior_variance, 4.5)
        self.assertEqual(updated_best_chunk_times, trial_chunk_times)
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)
        self.assertEqual(best_bub.near_best_trials, [])

        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_empty_both_trial_and_best_chunk_times(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        trial_chunk_times = []
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = []
        best_batch = None
        best_ubatch = None

        mock_update_bayesian.return_value = (float('inf'), 1e6)
        mock_calculate_prob.return_value = float('nan')

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertEqual(updated_best_chunk_times, best_chunk_times)  # Remain empty
        self.assertEqual(updated_best_batch, best_batch)
        self.assertEqual(updated_best_ubatch, best_ubatch)
        self.assertEqual(len(best_bub.near_best_trials), 1)

        trial_entry = best_bub.near_best_trials[0]
        self.assertEqual(trial_entry['chunk_time'], float('inf'))
        self.assertEqual(trial_entry['params'], {'n_batch': n_batch, 'n_ubatch': n_ubatch})
        self.assertTrue(np.isnan(trial_entry['prob_superiority']))
        self.assertIsNone(trial_entry['p_value'])

        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    @patch('best_bub.update_bayesian_mean_variance')
    @patch('best_bub.calculate_probability_of_superiority')
    def test_update_best_chunk_time_with_probability_prior_mean_variance_not_none_initialization(self, mock_calculate_prob, mock_update_bayesian):
        # Given
        best_bub.prior_mean = 100
        best_bub.prior_variance = 10
        trial_chunk_times = [100, 105, 102]
        n_batch = 128
        n_ubatch = 32
        best_chunk_times = [110, 115, 112]
        best_batch = 64
        best_ubatch = 16

        mock_update_bayesian.return_value = (102.11, 4.75)
        mock_calculate_prob.return_value = 0.97

        updated_best_chunk_times, updated_best_batch, updated_best_ubatch = best_bub.update_best_chunk_time_with_probability(
            trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch
        )

        self.assertAlmostEqual(best_bub.prior_mean, 102.11)
        self.assertAlmostEqual(best_bub.prior_variance, 4.75)
        self.assertEqual(updated_best_chunk_times, trial_chunk_times)
        self.assertEqual(updated_best_batch, n_batch)
        self.assertEqual(updated_best_ubatch, n_ubatch)
        self.assertEqual(best_bub.near_best_trials, [])

        mock_update_bayesian.assert_called_once()
        mock_calculate_prob.assert_called_once()

    def test_update_bayesian_mean_variance_multiple_data_points(self):
        prior_mean = 5.0
        prior_variance = 2.0
        new_data = np.array([4.0, 5.0, 6.0, 5.5])
        
        posterior_mean, posterior_variance = best_bub.update_bayesian_mean_variance(prior_mean, prior_variance, new_data)
        
        self.assertTrue(np.isfinite(posterior_mean), "Posterior mean should be finite.")
        self.assertTrue(np.isfinite(posterior_variance), "Posterior variance should be finite.")
        self.assertGreater(posterior_variance, 0, "Posterior variance should be positive.")

    def test_update_bayesian_mean_variance_single_data_point(self):
        prior_mean = 10.0
        prior_variance = 5.0
        new_data = np.array([9.0])
        
        posterior_mean, posterior_variance = best_bub.update_bayesian_mean_variance(prior_mean, prior_variance, new_data)
        
        self.assertTrue(np.isfinite(posterior_mean), "Posterior mean should be finite.")
        self.assertTrue(np.isfinite(posterior_variance), "Posterior variance should be finite.")
        self.assertGreater(posterior_variance, 0, "Posterior variance should be positive.")
        
    def test_update_bayesian_mean_variance_zero_prior_variance(self):
        prior_mean = 7.0
        prior_variance = 0  # Invalid prior variance, should use epsilon
        new_data = np.array([7.5, 8.0, 7.0])
        
        posterior_mean, posterior_variance = best_bub.update_bayesian_mean_variance(prior_mean, prior_variance, new_data)
        
        self.assertTrue(np.isfinite(posterior_mean), "Posterior mean should be finite.")
        self.assertTrue(np.isfinite(posterior_variance), "Posterior variance should be finite.")
        self.assertGreater(posterior_variance, 0, "Posterior variance should be positive.")

    def test_update_bayesian_mean_variance_invalid_prior_variance(self):
        prior_mean = 5.0
        prior_variance = None  # None prior variance, should use epsilon
        new_data = np.array([4.0, 6.0])
        
        posterior_mean, posterior_variance = best_bub.update_bayesian_mean_variance(prior_mean, prior_variance, new_data)
        
        self.assertTrue(np.isfinite(posterior_mean), "Posterior mean should be finite.")
        self.assertTrue(np.isfinite(posterior_variance), "Posterior variance should be finite.")
        self.assertGreater(posterior_variance, 0, "Posterior variance should be positive.")
    
    def test_update_bayesian_mean_variance_empty_data_array(self):
        prior_mean = 4.0
        prior_variance = 2.0
        new_data = np.array([])  # No data
        
        with self.assertRaises(ValueError, msg="Empty data array should raise an error."):
            best_bub.update_bayesian_mean_variance(prior_mean, prior_variance, new_data)

    def test_calculate_probability_of_superiority_positive_variances(self):
        current_best_mean = 10.0
        current_best_variance = 2.0
        trial_mean = 9.5
        trial_variance = 1.5
        
        prob_superiority = best_bub.calculate_probability_of_superiority(
            current_best_mean, current_best_variance, trial_mean, trial_variance
        )
        
        # Check if probability is a finite number between 0 and 1
        self.assertTrue(0 <= prob_superiority <= 1, "Probability should be between 0 and 1.")
        self.assertTrue(np.isfinite(prob_superiority), "Probability should be finite.")

    def test_calculate_probability_of_superiority_zero_variance_adjustment(self):
        current_best_mean = 8.0
        current_best_variance = 0  # Should trigger epsilon adjustment
        trial_mean = 8.5
        trial_variance = 0  # Should also trigger epsilon adjustment
        
        prob_superiority = best_bub.calculate_probability_of_superiority(
            current_best_mean, current_best_variance, trial_mean, trial_variance
        )
        
        # Probability should still be finite and between 0 and 1
        self.assertTrue(0 <= prob_superiority <= 1, "Probability should be between 0 and 1.")
        self.assertTrue(np.isfinite(prob_superiority), "Probability should be finite.")
        
    def test_calculate_probability_of_superiority_high_difference_in_means(self):
        current_best_mean = 20.0
        current_best_variance = 5.0
        trial_mean = 5.0  # Significant difference in means
        trial_variance = 3.0
        
        prob_superiority = best_bub.calculate_probability_of_superiority(
            current_best_mean, current_best_variance, trial_mean, trial_variance
        )
        
        # Expected to be lower than 0.5 as current_best_mean is significantly higher
        self.assertGreater(prob_superiority, 0.5, "Probability should be greater than 0.5.")
        
    def test_calculate_probability_of_superiority_high_trial_mean(self):
        current_best_mean = 10.0
        current_best_variance = 4.0
        trial_mean = 15.0  # Trial mean is significantly higher
        trial_variance = 3.0
        
        prob_superiority = best_bub.calculate_probability_of_superiority(
            current_best_mean, current_best_variance, trial_mean, trial_variance
        )
        
        # Expected to be greater than 0.5 as trial_mean is significantly higher
        self.assertLess(prob_superiority, 0.5, "Probability should be less than 0.5.")

    def test_calculate_probability_of_superiority_nearly_equal_means(self):
        current_best_mean = 10.0
        current_best_variance = 1.0
        trial_mean = 10.01  # Trial mean is very close to current best
        trial_variance = 1.0
        
        prob_superiority = best_bub.calculate_probability_of_superiority(
            current_best_mean, current_best_variance, trial_mean, trial_variance
        )
        
        # Probability should be close to 0.5 for nearly equal means
        self.assertAlmostEqual(prob_superiority, 0.5, delta=0.05, msg="Probability should be close to 0.5.")

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_execute_trials_normal_execution(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks for normal execution
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.return_value = [100, 200, 300]  # Simulated chunk times
        mock_update_best.return_value = ([100, 200, 300], 2048, 512)  # Simulated best times

        # Set up dummy values for parameters
        n_trials = 5
        pre_chunked_text = ['chunk1', 'chunk2', 'chunk3']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function under test
        best_bub.execute_trials(
            mock_study, n_trials, pre_chunked_text, kwargs,
            batch_exponent_range, ubatch_exponent_range
        )

        # Assertions
        self.assertEqual(mock_create_trial.call_count, n_trials)
        self.assertEqual(mock_objective_wrapper.call_count, n_trials)
        self.assertEqual(mock_update_best.call_count, n_trials)
        self.assertEqual(mock_study.tell.call_count, n_trials)

        # Verify that trials are told with correct values
        mock_study.tell.assert_called_with(mock_trial, 200)  # Average of [100, 200, 300]

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_execute_trials_with_pruned_trial(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks to simulate a pruned trial
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial

        def objective_side_effect(*args, **kwargs):
            raise optuna.TrialPruned()

        mock_objective_wrapper.side_effect = objective_side_effect

        # Set up dummy values for parameters
        n_trials = 1
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function under test
        best_bub.execute_trials(
            mock_study, n_trials, pre_chunked_text, kwargs,
            batch_exponent_range, ubatch_exponent_range
        )

        # Assertions
        self.assertEqual(mock_create_trial.call_count, 1)
        self.assertEqual(mock_objective_wrapper.call_count, 1)
        self.assertEqual(mock_update_best.call_count, 0)
        mock_logger.warning.assert_called_with(f"Trial {mock_trial.number} was pruned")
        mock_study.tell.assert_called_with(mock_trial, float('inf'))

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_execute_trials_with_exception(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks to simulate an exception
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial

        def objective_side_effect(*args, **kwargs):
            raise RuntimeError("Some error")

        mock_objective_wrapper.side_effect = objective_side_effect

        # Set up dummy values for parameters
        n_trials = 1
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function under test
        best_bub.execute_trials(
            mock_study, n_trials, pre_chunked_text, kwargs,
            batch_exponent_range, ubatch_exponent_range
        )

        # Assertions
        self.assertEqual(mock_create_trial.call_count, 1)
        self.assertEqual(mock_objective_wrapper.call_count, 1)
        self.assertEqual(mock_update_best.call_count, 0)
        mock_logger.warning.assert_called_with(f"Trial {mock_trial.number} failed with exception: Some error")
        mock_study.tell.assert_called_with(mock_trial, state=optuna.trial.TrialState.FAIL)

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_max_attempts_reached(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.side_effect = optuna.TrialPruned  # Simulate pruned trials

        # Set up parameters
        n_trials = 5
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function
        best_bub.execute_trials(
            mock_study, n_trials, pre_chunked_text, kwargs,
            batch_exponent_range, ubatch_exponent_range
        )

        # Since completed_trials increments every time, create_trial is called n_trials times
        self.assertEqual(mock_create_trial.call_count, n_trials)

        # Verify that the warning about pruning was logged
        mock_logger.warning.assert_any_call(f"Trial {mock_trial.number} was pruned")

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_initial_default_trial(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.return_value = [100, 200, 300]  # Simulate chunk times

        # Set up parameters
        n_trials = 1
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function
        best_bub.execute_trials(mock_study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Check that the default trial was created with specific batch sizes
        mock_create_trial.assert_called_once_with(
            mock_study, batch_exponent_range, ubatch_exponent_range,
            default_n_batch=2 ** best_bub.DEFAULT_BATCH_EXPONENT,
            default_n_ubatch=2 ** best_bub.DEFAULT_UBATCH_EXPONENT
        )

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_execute_trials_with_oom_error(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks to simulate OOM error
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial

        def objective_side_effect(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        mock_objective_wrapper.side_effect = objective_side_effect

        # Set up parameters
        n_trials = 1
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function
        best_bub.execute_trials(mock_study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Check for OOM error handling
        mock_logger.warning.assert_called_with(f"Trial {mock_trial.number} pruned due to OOM error: CUDA out of memory")
        mock_study.tell.assert_called_with(mock_trial, np.inf)

    @patch('best_bub.create_trial')
    @patch('best_bub.objective_wrapper')
    @patch('best_bub.update_best_chunk_time_with_probability')
    @patch('best_bub.logger')
    def test_best_chunk_times_initialization(
        self, mock_logger, mock_update_best, mock_objective_wrapper, mock_create_trial
    ):
        # Configure mocks to ensure best_chunk_times starts as empty
        mock_study = MagicMock()
        mock_trial = MagicMock(number=1, user_attrs={'n_batch': 2048, 'n_ubatch': 512})
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.return_value = [100, 200, 300]  # Simulated chunk times

        # Set up parameters
        n_trials = 1
        pre_chunked_text = ['chunk1']
        kwargs = {}
        batch_exponent_range = best_bub.ExponentRange(10, 12)
        ubatch_exponent_range = best_bub.ExponentRange(8, 10)

        # Run the function
        best_bub.execute_trials(mock_study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Verify that `update_best_chunk_time_with_probability` is called with initial empty best_chunk_times
        mock_update_best.assert_called_once_with(
            [100, 200, 300], 2048, 512, [], None, None
        )

class TestMainExecution(unittest.TestCase):
    def setUp(self):
        best_bub.trial_cache.clear()

        best_bub.near_best_trials.clear()
        best_bub.prior_mean = None
        best_bub.prior_variance = None

        # Optionally, reset the logger's handlers to avoid duplicate logs in tests
        best_bub.logger.handlers = []
        best_bub.logger.addHandler(logging.NullHandler())

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

    def test_setup_study_returns_study(self):
        study = best_bub.setup_study()
        self.assertIsInstance(study, optuna.study.Study, "setup_study should return an instance of optuna.study.Study")

    def test_generate_random_tokens(self):
        target_num_tokens = 1000  # Target number of tokens
        acceptable_overshoot = 0.1  # 10% overshoot allowed
        max_total_tokens = target_num_tokens * (1 + acceptable_overshoot)

        # Initialize the mock model
        mock_model = MockModel('dummy path')

        # Generate text with the target number of tokens
        text = best_bub.generate_random_tokens(target_num_tokens, mock_model)

        # Tokenize the generated text to get the actual token count
        actual_tokens = mock_model.tokenize(text.encode("utf-8"))
        actual_token_count = len(actual_tokens)

        # Check that the generated token count is at least the target
        self.assertGreaterEqual(
            actual_token_count, target_num_tokens,
            f"Token count ({actual_token_count}) is less than the target ({target_num_tokens})"
        )

        # Check that the generated token count does not exceed the acceptable limit
        self.assertLessEqual(
            actual_token_count, max_total_tokens,
            f"Token count ({actual_token_count}) exceeds the acceptable limit ({int(max_total_tokens)})"
        )        

    @patch("best_bub.generate_random_tokens")
    @patch("best_bub.llama_cpp.Llama", new=MockModel)
    def test_tokenize_returns_integer_list(self, mock_generate_random_tokens):
        # Mock the output of generate_random_tokens
        mock_generate_random_tokens.return_value = "test text"

        # Set up kwargs for tokenize function
        kwargs = {
            'context_size': 128,
            'chunks': 10
        }

        # Initialize MockModel and call the tokenize function
        model = MockModel('dummy model')
        tokenized_text = best_bub.tokenize(model, kwargs)

        # Verify that the result is a list of integers
        self.assertIsInstance(tokenized_text, list, "The result should be a list.")
        self.assertTrue(all(isinstance(token, int) for token in tokenized_text), "All items in the result should be integers.")

    def test_chunk_text_basic_functionality(self):
        tokenized_text = list(range(50))  # A simple sequence of tokens [0, 1, ..., 49]
        context_size = 10
        chunks = best_bub.chunk_text(tokenized_text, context_size)
        
        # Check that the result is a list
        self.assertIsInstance(chunks, list, "The result should be a list.")

        # Check each chunk size (except possibly the last one)
        expected_chunk_size = context_size - 1
        for chunk in chunks[:-1]:  # All except the last should be exactly the expected size
            self.assertEqual(len(chunk), expected_chunk_size, f"Each chunk should be of size {expected_chunk_size}.")
        
        # Verify that all tokens are covered without duplication or loss
        reconstructed_text = [token for chunk in chunks for token in chunk]
        self.assertEqual(reconstructed_text, tokenized_text, "The chunked and reconstructed text should match the original.")

    def test_chunk_text_with_edge_cases(self):
        # Test with empty tokenized_text
        self.assertEqual(best_bub.chunk_text([], 10), [], "Chunking an empty list should return an empty list.")

        # Test with context_size of 1 (smallest valid context)
        tokenized_text = [1, 2, 3]
        context_size = 2  # `context_size - 1` will be 1, so each chunk should contain one element
        self.assertEqual(best_bub.chunk_text(tokenized_text, context_size), [[1], [2], [3]], "Each chunk should contain exactly one token.")

        # Test with tokenized_text that isn't a multiple of context_size - 1
        tokenized_text = list(range(7))
        context_size = 4
        expected_chunks = [[0, 1, 2], [3, 4, 5], [6]]  # Last chunk should be shorter
        self.assertEqual(best_bub.chunk_text(tokenized_text, context_size), expected_chunks, "The last chunk should contain the remaining tokens.")

    def test_report_results(self):
        self.assertTrue(callable(best_bub.report_results))

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_happy_path(
        self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial
    ):
        # Set up the mock study
        study = MagicMock(spec=optuna.Study)
        
        # Define parameters for the test
        n_trials = 5
        pre_chunked_text = ["dummy_text_chunk"] * 10
        kwargs = {"context_size": 8192, "model": "test_model"}
        batch_exponent_range = MagicMock(min=9, max=12)
        ubatch_exponent_range = MagicMock(min=8, max=10)

        # Set return values for mocks
        mock_trial = MagicMock()
        mock_trial.number = 1
        mock_trial.user_attrs = {"n_batch": 1024, "n_ubatch": 512}
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.return_value = [50, 45, 47, 48]
        mock_update_best_chunk_time.return_value = ([50, 45, 47, 48], 1024, 512)

        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Assertions
        self.assertEqual(mock_create_trial.call_count, n_trials)
        self.assertEqual(mock_objective_wrapper.call_count, n_trials)
        mock_create_trial.assert_called_with(study, batch_exponent_range, ubatch_exponent_range)
        mock_update_best_chunk_time.assert_called()

        avg_best_time = sum(mock_objective_wrapper.return_value) / len(mock_objective_wrapper.return_value) * 2.5
        mock_objective_wrapper.assert_called_with(mock_trial, pre_chunked_text, kwargs, avg_best_time)

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_all_trials_complete(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        study = MagicMock(spec=optuna.Study)
        n_trials = 5
        pre_chunked_text = ["dummy_text_chunk"] * 10
        kwargs = {"context_size": 8192, "model": "test_model"}
        batch_exponent_range = MagicMock(min=9, max=12)
        ubatch_exponent_range = MagicMock(min=8, max=10)

        # Mock trial and result values
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"n_batch": 1024, "n_ubatch": 512}
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.return_value = [50, 45, 47, 48]
        mock_update_best_chunk_time.return_value = ([50, 45, 47, 48], 1024, 512)

        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Ensure n_trials were completed
        self.assertEqual(mock_create_trial.call_count, n_trials)
        self.assertEqual(mock_objective_wrapper.call_count, n_trials)

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_max_attempts(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        # Set up the mock study and params
        study = MagicMock(spec=optuna.Study)
        n_trials = 5  # Should not complete due to max_attempts
        pre_chunked_text = ["dummy_text_chunk"] * 10
        kwargs = {"context_size": 8192, "model": "test_model"}
        batch_exponent_range = MagicMock(min=9, max=12)
        ubatch_exponent_range = MagicMock(min=8, max=10)

        # Mock return values and raise an exception to prune each trial
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"n_batch": 1024, "n_ubatch": 512}
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.side_effect = optuna.TrialPruned()  # Simulate pruning on each trial

        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Verify max_attempts has prevented completion of n_trials
        self.assertTrue(mock_create_trial.call_count < n_trials * 10)

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_update_best_trial(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        # Setup the initial conditions for the test
        study = MagicMock(spec=optuna.Study)
        n_trials = 2  # Adjusted for simplicity
        pre_chunked_text = ["dummy_text_chunk"] * 10
        kwargs = {"context_size": 8192, "model": "test_model"}
        batch_exponent_range = MagicMock(min=9, max=12)
        ubatch_exponent_range = MagicMock(min=8, max=10)

        # Define the chunk times and params for each trial
        first_chunk_times = [60, 55, 58, 59]
        first_n_batch = 2048
        first_n_ubatch = 512

        second_chunk_times = [50, 45, 47, 48]
        second_n_batch = 1024
        second_n_ubatch = 512

        # Mock trial creation to return different trials for each call
        mock_trial1 = MagicMock()
        mock_trial1.user_attrs = {"n_batch": first_n_batch, "n_ubatch": first_n_ubatch}
        mock_trial1.number = 1
        mock_trial2 = MagicMock()
        mock_trial2.user_attrs = {"n_batch": second_n_batch, "n_ubatch": second_n_ubatch}
        mock_trial2.number = 2

        mock_create_trial.side_effect = [mock_trial1, mock_trial2]

        # Mock the return values for objective_wrapper for each trial
        mock_objective_wrapper.side_effect = [
            first_chunk_times,  # First trial times
            second_chunk_times  # Second trial times
        ]

        # Define a side effect function for update_best_chunk_time_with_probability
        def update_side_effect(trial_chunk_times, n_batch, n_ubatch, best_chunk_times, best_batch, best_ubatch):
            # Simulate updating best values when a better trial is found
            if not best_chunk_times or np.mean(trial_chunk_times) < np.mean(best_chunk_times):
                return trial_chunk_times, n_batch, n_ubatch
            else:
                return best_chunk_times, best_batch, best_ubatch

        # Assign the side effect to the mock
        mock_update_best_chunk_time.side_effect = update_side_effect

        # Run the function under test
        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Verify that update_best_chunk_time_with_probability was called with expected arguments

        # For the first trial
        mock_update_best_chunk_time.assert_any_call(
            first_chunk_times, first_n_batch, first_n_ubatch, [], None, None
        )

        # For the second trial
        mock_update_best_chunk_time.assert_any_call(
            second_chunk_times, second_n_batch, second_n_ubatch, first_chunk_times, first_n_batch, first_n_ubatch
        )

        # Assert that the mock for objective_wrapper was called twice
        self.assertEqual(mock_objective_wrapper.call_count, 2)

        # Assert that the mock for update_best_chunk_time_with_probability was called twice
        self.assertEqual(mock_update_best_chunk_time.call_count, 2)

        # Verify the final best values returned by the mock
        last_call = mock_update_best_chunk_time.call_args_list[-1]
        self.assertEqual(last_call[0], (second_chunk_times, second_n_batch, second_n_ubatch, first_chunk_times, first_n_batch, first_n_ubatch))

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_excludes_default_exponents_outside_range(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        # Setup the initial conditions for the test
        study = MagicMock(spec=optuna.Study)
        n_trials = 1  # Single trial for simplicity
        pre_chunked_text = ["dummy_text_chunk"] * 5
        kwargs = {"context_size": 8192, "model": "test_model"}
        
        batch_exponent_range = MagicMock(min=7, max=15)  
        ubatch_exponent_range = MagicMock(min=5, max=13) 

        # Define the chunk times and parameters for the trial that should be created
        expected_n_batch_exponent = best_bub.DEFAULT_BATCH_EXPONENT  
        expected_n_ubatch_exponent = best_bub.DEFAULT_UBATCH_EXPONENT  
        expected_n_batch = 2 ** expected_n_batch_exponent
        expected_n_ubatch = 2 ** expected_n_ubatch_exponent

        # Mock the trial creation
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"n_batch": expected_n_batch, "n_ubatch": expected_n_ubatch}  # Setting user_attrs with expected batch values
        mock_create_trial.return_value = mock_trial

        # Mock the return value for objective_wrapper for the trial
        mock_objective_wrapper.return_value = [55, 50, 53, 54]  # Example chunk times

        # Run the function under test
        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Assert that create_trial was called without default_n_batch and default_n_ubatch
        for call in mock_create_trial.call_args_list:
            self.assertIn('default_n_batch', call[1])
            self.assertEqual(call[1]['default_n_batch'], best_bub.DEFAULT_BATCH_EXPONENT)
            self.assertIn('default_n_ubatch', call[1])
            self.assertEqual(call[1]['default_n_ubatch'], best_bub.DEFAULT_UBATCH_EXPONENT)

        # Verify that the mock for create_trial was called exactly once
        self.assertEqual(mock_create_trial.call_count, 1)

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_excludes_default_exponents_outside_range(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        # Setup the initial conditions for the test
        study = MagicMock(spec=optuna.Study)
        n_trials = 1  # Single trial for simplicity
        pre_chunked_text = ["dummy_text_chunk"] * 5
        kwargs = {"context_size": 8192, "model": "test_model"}
        
        # Set up batch and ubatch exponent ranges that exclude DEFAULT_BATCH_EXPONENT and DEFAULT_UBATCH_EXPONENT
        # DEFAULT_BATCH_EXPONENT = 11, DEFAULT_UBATCH_EXPONENT = 9
        batch_exponent_range = MagicMock(min=7, max=10)  # Range that excludes the default 11
        ubatch_exponent_range = MagicMock(min=5, max=8)  # Range that excludes the default 9

        # Define DUMMY chunk times and parameters for the trial that should be created
        expected_n_batch_exponent = 10  # Should be within the provided range
        expected_n_ubatch_exponent = 8  # Should be within the provided range
        expected_n_batch = 2 ** expected_n_batch_exponent
        expected_n_ubatch = 2 ** expected_n_ubatch_exponent

        # Mock the trial creation
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"n_batch": expected_n_batch, "n_ubatch": expected_n_ubatch}  # Setting user_attrs with expected batch values
        mock_create_trial.return_value = mock_trial

        # Mock the return value for objective_wrapper for the trial
        mock_objective_wrapper.return_value = [55, 50, 53, 54]  # Example chunk times

        # Run the function under test
        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Assert that create_trial was called without default_n_batch and default_n_ubatch
        for call in mock_create_trial.call_args_list:
            self.assertNotIn('default_n_batch', call[1])
            self.assertNotIn('default_n_ubatch', call[1])

        # Verify that the mock for create_trial was called exactly once
        self.assertEqual(mock_create_trial.call_count, 1)

    @patch("best_bub.create_trial")
    @patch("best_bub.objective_wrapper")
    @patch("best_bub.update_best_chunk_time_with_probability")
    def test_execute_trials_pruned_due_to_oom(self, mock_update_best_chunk_time, mock_objective_wrapper, mock_create_trial):
        study = MagicMock(spec=optuna.Study)
        n_trials = 5
        pre_chunked_text = ["dummy_text_chunk"] * 10
        kwargs = {"context_size": 8192, "model": "test_model"}
        batch_exponent_range = MagicMock(min=9, max=12)
        ubatch_exponent_range = MagicMock(min=8, max=10)

        # Simulate OOM by raising RuntimeError during objective_wrapper call
        mock_trial = MagicMock()
        mock_trial.user_attrs = {"n_batch": 1024, "n_ubatch": 512}
        mock_create_trial.return_value = mock_trial
        mock_objective_wrapper.side_effect = RuntimeError("CUDA out of memory")  # Simulate OOM

        from best_bub import execute_trials
        execute_trials(study, n_trials, pre_chunked_text, kwargs, batch_exponent_range, ubatch_exponent_range)

        # Confirm objective_wrapper was retried and eventually pruned each trial due to OOM
        self.assertEqual(mock_create_trial.call_count, n_trials)
        self.assertEqual(mock_objective_wrapper.call_count, n_trials)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_with_max_trials_and_chunks(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=9, max=11)
        mock_ubatch_range = best_bub.ExponentRange(min=7, max=9)

        # Define a side_effect to capture the arguments
        def capture_initialize_batch_config_call(kwargs):
            self.captured_kwargs = kwargs.copy()
            return (mock_batch_range, mock_ubatch_range)

        mock_initialize_batch_and_model_config.side_effect = capture_initialize_batch_config_call

        # Since max_trials is provided, estimate_number_of_trials should not be called
        mock_estimate_number_of_trials.return_value = 100

        mock_prepare_llama_args.return_value = {'arg1': 'value1'}

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['token1', 'token2']
        mock_chunk_text.return_value = ['chunk1', 'chunk2']

        # Define kwargs
        kwargs_initial = {
            'max_trials': 50,
            'chunks': 10,
            'conform_to_imatrix': False,
            'context_size': 2048,
            'other_param': 'value'
        }
        kwargs = kwargs_initial.copy()

        # Ensure logger.debug is disabled
        mock_logger.isEnabledFor.return_value = False

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()
        mock_initialize_batch_and_model_config.assert_called_once_with(kwargs_initial)
        mock_estimate_number_of_trials.assert_not_called()  # max_trials provided

        mock_prepare_llama_args.assert_called_once_with(kwargs)
        mock_llama_class.assert_called_once_with(arg1='value1')
        mock_model_instance.close.assert_called_once()

        mock_tokenize.assert_called_once_with(mock_model_instance, kwargs)
        mock_chunk_text.assert_called_once_with(['token1', 'token2'], kwargs['context_size'])

        mock_execute_trials.assert_called_once_with(
            mock_study, 50, ['chunk1', 'chunk2'], kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

        # Check logger.debug not called since logging level is not DEBUG
        mock_logger.isEnabledFor.assert_called_with(logging.DEBUG)
        mock_logger.debug.assert_not_called()

        # Since 'max_trials' is provided, 'estimate_number_of_trials' should not be called,
        # and there should be no log about automatic estimation.
        # Therefore, 'Estimated number of trials automatically' should not be in logs.

        # Check that 'Auto-estimated chunks' was not logged
        unwanted_call = call(f"Auto-estimated chunks: 10 for batch size {2 ** mock_batch_range.max} and context size {kwargs['context_size']}")
        self.assertNotIn(unwanted_call, mock_logger.info.call_args_list)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_without_max_trials_and_with_conform_to_imatrix(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=9, max=11)
        mock_ubatch_range = best_bub.ExponentRange(min=7, max=9)

        # Define side effect functions to capture arguments
        captured_initialize_kwargs = None
        def initialize_batch_and_model_config_side_effect(kwargs):
            nonlocal captured_initialize_kwargs
            captured_initialize_kwargs = kwargs.copy()
            return (mock_batch_range, mock_ubatch_range)

        captured_prepare_kwargs = None
        def prepare_llama_args_side_effect(kwargs):
            nonlocal captured_prepare_kwargs
            captured_prepare_kwargs = kwargs.copy()
            return {'arg1': 'value1'}

        mock_initialize_batch_and_model_config.side_effect = initialize_batch_and_model_config_side_effect
        mock_prepare_llama_args.side_effect = prepare_llama_args_side_effect

        mock_estimate_number_of_trials.return_value = 150

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['tokenA', 'tokenB']
        mock_chunk_text.return_value = ['chunkA', 'chunkB']

        # Define kwargs without max_trials and chunks, with conform_to_imatrix=True
        kwargs_initial = {
            'max_trials': None,
            'chunks': None,
            'conform_to_imatrix': True,
            'context_size': 1024,
            'other_param': 'value2'
        }
        kwargs = kwargs_initial.copy()

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()
        mock_initialize_batch_and_model_config.assert_called_once()

        # Compare captured initialize kwargs to kwargs_initial
        self.assertEqual(captured_initialize_kwargs, kwargs_initial)

        mock_estimate_number_of_trials.assert_called_once_with(mock_batch_range, mock_ubatch_range)

        # Check captured kwargs for prepare_llama_args
        for k, v in captured_prepare_kwargs.items():
            if k != 'chunks':
                self.assertEqual(v, kwargs_initial[k])

        mock_llama_class.assert_called_once_with(arg1='value1')
        mock_model_instance.close.assert_called_once()

        mock_tokenize.assert_called_once_with(mock_model_instance, captured_prepare_kwargs)
        mock_chunk_text.assert_called_once_with(['tokenA', 'tokenB'], kwargs['context_size'])

        # Check that chunks were auto-estimated to 5 due to conform_to_imatrix=True
        self.assertEqual(captured_prepare_kwargs['chunks'], 5)
        mock_logger.info.assert_any_call(
            f"Estimated number of trials automatically: 150"
        )
        mock_logger.info.assert_any_call(
            f"Auto-estimated chunks: 5 for batch size {2 ** mock_batch_range.max} and context size {kwargs['context_size']}"
        )

        mock_execute_trials.assert_called_once_with(
            mock_study, 150, ['chunkA', 'chunkB'], captured_prepare_kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

        # Ensure debug logs were emitted if logging level is DEBUG
        mock_logger.isEnabledFor.assert_called_with(logging.DEBUG)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_with_debug_logging(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=10, max=12)
        mock_ubatch_range = best_bub.ExponentRange(min=8, max=10)

        # Define side effect functions to capture arguments
        captured_initialize_kwargs = []
        def initialize_batch_and_model_config_side_effect(kwargs):
            captured_initialize_kwargs.append(kwargs.copy())
            return (mock_batch_range, mock_ubatch_range)

        captured_prepare_kwargs = []
        def prepare_llama_args_side_effect(kwargs):
            captured_prepare_kwargs.append(kwargs.copy())
            return {'argX': 'valueX'}

        mock_initialize_batch_and_model_config.side_effect = initialize_batch_and_model_config_side_effect
        mock_prepare_llama_args.side_effect = prepare_llama_args_side_effect

        mock_estimate_number_of_trials.return_value = 200

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['tokenX', 'tokenY']
        mock_chunk_text.return_value = ['chunkX', 'chunkY']

        # Define kwargs without max_trials and chunks, with conform_to_imatrix=False
        kwargs_initial = {
            'max_trials': None,
            'chunks': None,
            'conform_to_imatrix': False,
            'context_size': 4096,
            'other_param': 'value3'
        }
        kwargs = kwargs_initial.copy()

        # Configure logger to have DEBUG level
        mock_logger.isEnabledFor.return_value = True

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()

        # Verify initialize_batch_and_model_config received kwargs_initial with chunks unset
        self.assertEqual(len(captured_initialize_kwargs), 1)
        self.assertEqual(captured_initialize_kwargs[0], kwargs_initial)

        mock_estimate_number_of_trials.assert_called_once_with(mock_batch_range, mock_ubatch_range)

        # Verify prepare_llama_args received the modified kwargs with chunks set
        self.assertEqual(len(captured_prepare_kwargs), 1)
        max_batch_size = 2 ** mock_batch_range.max  # 2^12 = 4096
        expected_chunks = max(5, math.ceil(max_batch_size / kwargs_initial['context_size']))  # max(5, 4096/4096) = 5
        expected_modified_kwargs = kwargs_initial.copy()
        expected_modified_kwargs['chunks'] = expected_chunks
        self.assertEqual(captured_prepare_kwargs[0], expected_modified_kwargs)

        mock_llama_class.assert_called_once_with(argX='valueX')
        mock_model_instance.close.assert_called_once()

        # Ensure chunk_text is called with the correct context size
        mock_chunk_text.assert_called_once_with(['tokenX', 'tokenY'], kwargs_initial['context_size'])

        # Logging assertions
        mock_logger.info.assert_any_call(
            f"Estimated number of trials automatically: 200"
        )
        mock_logger.info.assert_any_call(
            f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}"
        )

        # Check debug logs
        batch_sizes = [2 ** exp for exp in range(mock_batch_range.min, mock_batch_range.max + 1)]
        ubatch_sizes = [2 ** exp for exp in range(mock_ubatch_range.min, mock_ubatch_range.max + 1)]
        expected_debug_calls = [
            call(f"Batch size range (2^{mock_batch_range.min} to 2^{mock_batch_range.max}): {batch_sizes}"),
            call(f"Ubatch size range (2^{mock_ubatch_range.min} to 2^{mock_ubatch_range.max}): {ubatch_sizes}")
        ]
        mock_logger.debug.assert_has_calls(expected_debug_calls, any_order=True)

        # Confirm execute_trials is called with the final, modified kwargs
        mock_execute_trials.assert_called_once_with(
            mock_study, 200, ['chunkX', 'chunkY'], expected_modified_kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

        # Ensure that 'Auto-estimated chunks' was logged correctly
        auto_estimated_call = call(f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}")
        self.assertIn(auto_estimated_call, mock_logger.info.call_args_list)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_without_chunks_conform_false(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=8, max=10)
        mock_ubatch_range = best_bub.ExponentRange(min=6, max=8)

        # Define side effect functions to capture arguments
        captured_initialize_kwargs = []
        def initialize_batch_and_model_config_side_effect(kwargs):
            captured_initialize_kwargs.append(kwargs.copy())
            return (mock_batch_range, mock_ubatch_range)

        captured_prepare_kwargs = []
        def prepare_llama_args_side_effect(kwargs):
            captured_prepare_kwargs.append(kwargs.copy())
            return {'argY': 'valueY'}

        mock_initialize_batch_and_model_config.side_effect = initialize_batch_and_model_config_side_effect
        mock_prepare_llama_args.side_effect = prepare_llama_args_side_effect

        mock_estimate_number_of_trials.return_value = 80

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['tokenM', 'tokenN']
        mock_chunk_text.return_value = ['chunkM', 'chunkN']

        # Define kwargs without max_trials and chunks, with conform_to_imatrix=False
        kwargs_initial = {
            'max_trials': None,
            'chunks': None,
            'conform_to_imatrix': False,
            'context_size': 1024,
            'other_param': 'value4'
        }
        kwargs = kwargs_initial.copy()

        # Configure logger to not have DEBUG level
        mock_logger.isEnabledFor.return_value = False

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()
        mock_initialize_batch_and_model_config.assert_called_once()

        # Verify initialize_batch_and_model_config received kwargs_initial
        self.assertEqual(len(captured_initialize_kwargs), 1)
        self.assertEqual(captured_initialize_kwargs[0], kwargs_initial)

        mock_estimate_number_of_trials.assert_called_once_with(mock_batch_range, mock_ubatch_range)

        # Verify prepare_llama_args received the modified kwargs with chunks set
        self.assertEqual(len(captured_prepare_kwargs), 1)
        max_batch_size = 2 ** mock_batch_range.max  # 2^10 = 1024
        expected_chunks = max(5, math.ceil(max_batch_size / kwargs_initial['context_size']))  # max(5, 1024/1024) = 5
        expected_modified_kwargs = kwargs_initial.copy()
        expected_modified_kwargs['chunks'] = expected_chunks
        self.assertEqual(captured_prepare_kwargs[0], expected_modified_kwargs)

        mock_llama_class.assert_called_once_with(argY='valueY')
        mock_model_instance.close.assert_called_once()

        # Ensure chunk_text is called with the correct context size
        mock_chunk_text.assert_called_once_with(['tokenM', 'tokenN'], kwargs_initial['context_size'])

        # Logging assertions
        mock_logger.info.assert_any_call(
            f"Estimated number of trials automatically: 80"
        )
        mock_logger.info.assert_any_call(
            f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}"
        )

        # Ensure that 'Auto-estimated chunks' was logged correctly
        auto_estimated_call = call(f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}")
        self.assertIn(auto_estimated_call, mock_logger.info.call_args_list)

        # Check that debug logs were not called
        mock_logger.debug.assert_not_called()

        # Confirm execute_trials is called with the final, modified kwargs
        mock_execute_trials.assert_called_once_with(
            mock_study, 80, ['chunkM', 'chunkN'], expected_modified_kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_with_chunks_provided(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=10, max=14)
        mock_ubatch_range = best_bub.ExponentRange(min=8, max=12)

        def capture_initialize_batch_config_call(kwargs):
            self.captured_kwargs = kwargs.copy()
            return (mock_batch_range, mock_ubatch_range)

        mock_initialize_batch_and_model_config.side_effect = capture_initialize_batch_config_call

        # Even though max_trials is None, chunks are provided, so estimate_number_of_trials should be called
        mock_estimate_number_of_trials.return_value = 250

        mock_prepare_llama_args.return_value = {'argZ': 'valueZ'}

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['tokenP', 'tokenQ']
        mock_chunk_text.return_value = ['chunkP', 'chunkQ']

        # Define kwargs with chunks provided
        kwargs_initial = {
            'max_trials': None,
            'chunks': 20,
            'conform_to_imatrix': False,
            'context_size': 2048,
            'other_param': 'value5'
        }
        kwargs = kwargs_initial.copy()

        # Configure logger to not have DEBUG level
        mock_logger.isEnabledFor.return_value = False

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()
        mock_initialize_batch_and_model_config.assert_called_once_with(kwargs_initial)
        mock_estimate_number_of_trials.assert_called_once_with(mock_batch_range, mock_ubatch_range)

        mock_prepare_llama_args.assert_called_once_with(kwargs)
        mock_llama_class.assert_called_once_with(argZ='valueZ')
        mock_model_instance.close.assert_called_once()

        mock_tokenize.assert_called_once_with(mock_model_instance, kwargs)
        mock_chunk_text.assert_called_once_with(['tokenP', 'tokenQ'], kwargs['context_size'])

        # Chunks should remain as provided
        self.assertEqual(kwargs['chunks'], 20)
        mock_logger.info.assert_any_call(
            f"Estimated number of trials automatically: 250"
        )
        # Ensure that 'Auto-estimated chunks' was not auto-set to 5
        unexpected_call = call(f"Auto-estimated chunks: 5 for batch size {2 ** mock_batch_range.max} and context size {kwargs['context_size']}")
        self.assertNotIn(unexpected_call, mock_logger.info.call_args_list)

        # Check that debug logs were not called
        mock_logger.debug.assert_not_called()

        mock_execute_trials.assert_called_once_with(
            mock_study, 250, ['chunkP', 'chunkQ'], kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

        # Verify that 'initialize_batch_and_model_config' was called with the expected modified kwargs
        expected_kwargs = {
            'max_trials': None,
            'chunks': 20,  # Expected modification (no change since chunks were provided)
            'conform_to_imatrix': False,
            'context_size': 2048,
            'other_param': 'value5'
        }
        self.assertEqual(self.captured_kwargs, expected_kwargs)

    @patch('best_bub.report_results')
    @patch('best_bub.execute_trials')
    @patch('best_bub.chunk_text')
    @patch('best_bub.tokenize')
    @patch('best_bub.llama_cpp.Llama')
    @patch('best_bub.prepare_llama_args')
    @patch('best_bub.estimate_number_of_trials')
    @patch('best_bub.initialize_batch_and_model_config')
    @patch('best_bub.setup_study')
    @patch('best_bub.logger')
    def test_main_with_debug_logging_enabled(
        self,
        mock_logger,
        mock_setup_study,
        mock_initialize_batch_and_model_config,
        mock_estimate_number_of_trials,
        mock_prepare_llama_args,
        mock_llama_class,
        mock_tokenize,
        mock_chunk_text,
        mock_execute_trials,
        mock_report_results
    ):
        # Setup mocks
        mock_study = MagicMock()
        mock_setup_study.return_value = mock_study

        mock_batch_range = best_bub.ExponentRange(min=10, max=12)
        mock_ubatch_range = best_bub.ExponentRange(min=8, max=10)

        # Define side effect functions to capture arguments
        captured_initialize_kwargs = []
        def initialize_batch_and_model_config_side_effect(kwargs):
            captured_initialize_kwargs.append(kwargs.copy())
            return (mock_batch_range, mock_ubatch_range)

        captured_prepare_kwargs = []
        def prepare_llama_args_side_effect(kwargs):
            captured_prepare_kwargs.append(kwargs.copy())
            return {'argW': 'valueW'}

        mock_initialize_batch_and_model_config.side_effect = initialize_batch_and_model_config_side_effect
        mock_prepare_llama_args.side_effect = prepare_llama_args_side_effect

        mock_estimate_number_of_trials.return_value = 300

        mock_model_instance = MagicMock()
        mock_llama_class.return_value = mock_model_instance

        mock_tokenize.return_value = ['tokenR', 'tokenS']
        mock_chunk_text.return_value = ['chunkR', 'chunkS']

        # Define kwargs without max_trials and chunks, with conform_to_imatrix=False
        kwargs_initial = {
            'max_trials': None,
            'chunks': None,
            'conform_to_imatrix': False,
            'context_size': 512,
            'other_param': 'value6'
        }
        kwargs = kwargs_initial.copy()

        # Configure logger to have DEBUG level
        mock_logger.isEnabledFor.return_value = True

        # Call main
        best_bub.main(**kwargs)

        # Assertions
        mock_setup_study.assert_called_once()

        # Verify initialize_batch_and_model_config received kwargs_initial with chunks unset
        self.assertEqual(len(captured_initialize_kwargs), 1)
        self.assertEqual(captured_initialize_kwargs[0], kwargs_initial)

        mock_estimate_number_of_trials.assert_called_once_with(mock_batch_range, mock_ubatch_range)

        # Verify prepare_llama_args received the modified kwargs with chunks set
        self.assertEqual(len(captured_prepare_kwargs), 1)
        max_batch_size = 2 ** mock_batch_range.max  # 2^12 = 4096
        expected_chunks = max(5, math.ceil(max_batch_size / kwargs_initial['context_size']))  # max(5, 4096/512) = 8
        expected_modified_kwargs = kwargs_initial.copy()
        expected_modified_kwargs['chunks'] = expected_chunks
        self.assertEqual(captured_prepare_kwargs[0], expected_modified_kwargs)

        mock_llama_class.assert_called_once_with(argW='valueW')
        mock_model_instance.close.assert_called_once()

        # Ensure chunk_text is called with the correct context size
        mock_chunk_text.assert_called_once_with(['tokenR', 'tokenS'], kwargs_initial['context_size'])

        # Logging assertions
        mock_logger.info.assert_any_call(
            f"Estimated number of trials automatically: 300"
        )
        mock_logger.info.assert_any_call(
            f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}"
        )

        # Check debug logs
        batch_sizes = [2 ** exp for exp in range(mock_batch_range.min, mock_batch_range.max + 1)]
        ubatch_sizes = [2 ** exp for exp in range(mock_ubatch_range.min, mock_ubatch_range.max + 1)]
        expected_debug_calls = [
            call(f"Batch size range (2^{mock_batch_range.min} to 2^{mock_batch_range.max}): {batch_sizes}"),
            call(f"Ubatch size range (2^{mock_ubatch_range.min} to 2^{mock_ubatch_range.max}): {ubatch_sizes}")
        ]
        mock_logger.debug.assert_has_calls(expected_debug_calls, any_order=True)

        # Confirm execute_trials is called with the final, modified kwargs
        mock_execute_trials.assert_called_once_with(
            mock_study, 300, ['chunkR', 'chunkS'], expected_modified_kwargs, mock_batch_range, mock_ubatch_range
        )
        mock_report_results.assert_called_once_with(mock_study)

        # Ensure that 'Auto-estimated chunks' was logged correctly
        auto_estimated_call = call(f"Auto-estimated chunks: {expected_chunks} for batch size {max_batch_size} and context size {kwargs_initial['context_size']}")
        self.assertIn(auto_estimated_call, mock_logger.info.call_args_list)

if __name__ == "__main__":
    unittest.main()
