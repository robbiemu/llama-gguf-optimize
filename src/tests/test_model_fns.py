import contextlib
import logging
import unittest
from unittest.mock import patch

from gguf_optimize_model_fns import estimate_model_parameters, estimate_model_precision


logger = logging.getLogger("gguf_optimize_model_fns")

class TestGenerateLogits(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

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

    def test_estimate_model_parameters_valid(self):
        # Test to ensure the estimated size is valid for the given metadata
        metadata = {
            'llama.vocab_size': 32000,
            'llama.embedding_length': 4096,
            'llama.feed_forward_length': 16384,
            'llama.block_count': 32
        }
        # Expected value calculation
        vocab_size = metadata['llama.vocab_size']
        embedding_length = metadata['llama.embedding_length']
        feed_forward_length = metadata['llama.feed_forward_length']
        num_layers = metadata['llama.block_count']

        embedding_params = vocab_size * embedding_length
        layer_params_per_layer = 4 * embedding_length**2 + 4 * embedding_length * feed_forward_length
        total_params = embedding_params + (num_layers * layer_params_per_layer)

        estimated_size = estimate_model_parameters(metadata)
        self.assertEqual(estimated_size, total_params)

    def test_estimate_model_parameters_missing_parameter(self):
        # Test to ensure the function returns None when a required parameter is missing
        metadata = {
            'llama.vocab_size': 32000,
            'llama.embedding_length': 4096,
            'llama.feed_forward_length': 16384  # Missing 'llama.block_count'
        }
        estimated_size = estimate_model_parameters(metadata)
        self.assertIsNone(estimated_size)

    @patch('os.path.getsize', return_value=1000000000)
    @patch('llama_cpp.Llama')
    def test_estimate_model_precision_valid(self, MockLlama, mock_getsize):
        # Happy path with a valid model and valid metadata
        mock_instance = MockLlama()
        mock_instance.metadata = {
            'llama.vocab_size': 32000,
            'llama.embedding_length': 4096,
            'llama.feed_forward_length': 16384,
            'llama.block_count': 32
        }

        # Expected calculation
        num_params = estimate_model_parameters(mock_instance.metadata)
        file_size_bytes = mock_getsize.return_value
        bits_per_weight = (file_size_bytes * 8) / num_params

        estimated_precision = estimate_model_precision('mock_model.gguf')
        self.assertEqual(float(estimated_precision), bits_per_weight)

    @patch('os.path.getsize', return_value=1000000000)
    @patch('llama_cpp.Llama')
    def test_estimate_model_precision_missing_metadata(self, MockLlama, mock_getsize):
        # Warning when metadata is missing or invalid
        mock_instance = MockLlama()
        mock_instance.metadata = None

        with self.temporary_logger_config(logger, logging.ERROR), self.assertLogs('gguf_optimize_model_fns', level='ERROR') as log:
            estimated_precision = estimate_model_precision('mock_model.gguf')
            self.assertEqual(float(estimated_precision), 32.0)
            self.assertTrue(any("An error occurred while processing the GGUF file: 'NoneType' object has no attribute 'get'." in msg for msg in log.output))

    @patch('os.path.getsize', return_value=1000000000)
    @patch('llama_cpp.Llama')
    def test_estimate_model_precision_zero_parameters(self, MockLlama, mock_getsize):
        # Set mock metadata to trigger zero parameters condition
        mock_instance = MockLlama()
        mock_instance.metadata = {
            'llama.vocab_size': 0,
            'llama.embedding_length': 0,
            'llama.feed_forward_length': 0,
            'llama.block_count': 0
        }

        with self.temporary_logger_config(logger, logging.WARNING), self.assertLogs('gguf_optimize_model_fns', level='WARNING') as log:
            estimated_precision = estimate_model_precision('mock_model.gguf')
            self.assertEqual(float(estimated_precision), 32.0)
            self.assertTrue(any("Unable to estimate number of parameters. Defaulting to 32.0 bits." in msg for msg in log.output))

    @patch('llama_cpp.Llama', side_effect=FileNotFoundError)
    def test_estimate_model_precision_file_not_found(self, MockLlama):
        with self.temporary_logger_config(logger, logging.ERROR), self.assertLogs('gguf_optimize_model_fns', level='ERROR') as log:
            # Call the function
            result = estimate_model_precision("non_existent_path")
            self.assertEqual(result, 32)
            self.assertTrue(any("GGUF file not found at path:" in msg for msg in log.output))

    @patch('llama_cpp.Llama', side_effect=Exception("Generic Error"))
    def test_estimate_model_precision_generic_error(self, MockLlama):
        with self.temporary_logger_config(logger, logging.ERROR), self.assertLogs('gguf_optimize_model_fns', level='ERROR') as log:
            estimated_precision = estimate_model_precision('mock_model.gguf')
            self.assertEqual(float(estimated_precision), 32.0)
            self.assertTrue(any('Generic Error' in msg for msg in log.output))


if __name__ == '__main__':
    unittest.main()
