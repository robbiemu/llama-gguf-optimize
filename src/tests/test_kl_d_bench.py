import argparse
import contextlib
import h5py
import logging
import numpy as np
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

from mock_model import MockModel
from generate_logits import generate_logits_with_llama_cpp 
from compare_logits import process_chunks
from gguf_optimize_logging import setup_logging
import kl_d_bench

class TestLogitsProcessing(unittest.TestCase):
    def setUp(self):
        if os.path.exists('baseline_logits.h5'):
            os.remove('baseline_logits.h5')
        if os.path.exists('taget_logits.h5'):
            os.remove('taget_logits.h5')

        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        if os.path.exists('baseline_logits.h5'):
            os.remove('baseline_logits.h5')
        if os.path.exists('taget_logits.h5'):
            os.remove('taget_logits.h5')

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

    @patch('kl_d_bench.h5py.File')
    def test_reset_chunk_in_hdf5_create_freed_chunks_dataset(self, mock_h5py_file):
        """
        Test reset_chunk_in_hdf5 when 'freed_chunks' dataset does not exist and needs to be created.
        """
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.create_dataset = MagicMock()

        # Call the function
        kl_d_bench.reset_chunk_in_hdf5('test.h5', 10)

        # Check that the HDF5 file was opened in append mode
        mock_h5py_file.assert_called_once_with('test.h5', 'a')

        # Ensure the dataset was created with the correct name and data
        mock_file.create_dataset.assert_called_once_with(
            'freed_chunks', data=[10], maxshape=(None,), dtype='int64'
        )

    @patch('kl_d_bench.h5py.File')
    def test_reset_chunk_in_hdf5_append_to_existing_freed_chunks_dataset(self, mock_h5py_file):
        """
        Test reset_chunk_in_hdf5 when 'freed_chunks' dataset already exists.
        """
        # Setup mock file and dataset
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_dataset = MagicMock()
        mock_dataset.shape = (5,)
        
        # Simulate the existence of the 'freed_chunks' dataset
        mock_file.__contains__.return_value = True
        mock_file.__getitem__.return_value = mock_dataset

        # Call the function under test
        kl_d_bench.reset_chunk_in_hdf5('test.h5', 5)

        # Assertions
        # Ensure the file was opened in append mode
        mock_h5py_file.assert_called_once_with('test.h5', 'a')
        
        # Ensure 'freed_chunks' dataset was accessed
        mock_file.__getitem__.assert_called_once_with('freed_chunks')
        
        # Ensure dataset was resized to hold the new index
        mock_dataset.resize.assert_called_once_with(6, axis=0)
        
        # Ensure the new chunk index was appended correctly
        mock_dataset.__setitem__.assert_called_once_with(-1, 5)

    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    def test_generate_logits_for_model(self, mock_generate_logits):
        """
        Test generate_logits_for_model function.
        """
        generate_args = {
            'model_path': 'dummy model', 
            'output': 'test.h5', 
            'clobber': False
        }
        kl_d_bench.generate_logits_for_model(generate_args, 3)

        mock_generate_logits.assert_called_once_with(
            model_path='dummy model', output='test.h5', clobber=False, from_chunk=3, to_chunk=4
        )

    @patch('kl_d_bench.process_chunks')
    def test_compare_logits_for_chunk(self, mock_process_chunks):
        """
        Test compare_logits_for_chunk function.
        """
        compare_args = {'output_path': 'kl_divergence.h5'}
        kl_d_bench.compare_logits_for_chunk(compare_args, 2)

        mock_process_chunks.assert_called_once_with(
            output_path='kl_divergence.h5', from_chunk=2, to_chunk=3, clobber=False
        )

    @patch('builtins.open', new_callable=mock_open, read_data='test data')
    @patch('kl_d_bench.h5py.File')
    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    @patch('kl_d_bench.process_chunks')
    def test_process_generate_both(self, mock_process_chunks, mock_generate_logits, mock_h5py_file, mock_open_file):
        """
        Test process_generate_both function.
        """
        args = argparse.Namespace(
            baseline_model='baseline.gguf', 
            target_model='target.gguf', 
            dataset='dataset.txt', 
            output_file='kl_divergence.h5',
            baseline_logits_output='baseline_logits.h5',
            target_logits_output='target_logits.h5'
        )
        generate_args_baseline={'output_file': 'baseline_logits.h5'}
        generate_args_target={ 'output_file': 'taget_logits.h5'}
        total_chunks = 3
        compare_args = {'output_path': 'kl_divergence.h5'}

        # Mock the h5py File context manager to return a mock object
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': total_chunks}  # Set total_chunks attribute

        kl_d_bench.process_generate_both(args, total_chunks, generate_args_baseline, generate_args_target, compare_args)

        self.assertEqual(mock_generate_logits.call_count, total_chunks * 2)  # Called for both models
        self.assertEqual(mock_process_chunks.call_count, total_chunks)

    # ... (Similarly add tests for process_generate_one and process_generate_neither) ...

    @patch('builtins.open', new_callable=mock_open, read_data='test data')
    @patch('kl_d_bench.h5py.File')
    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    @patch('kl_d_bench.process_generate_both')
    def test_main_generate_both(self, mock_process_generate_both, mock_generate_logits_with_llama_cpp, mock_h5py_file, mock_open_file):
        """
        Test main function when both logits need to be generated.
        """
        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model='target.gguf',
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits_output=kl_d_bench.DEFAULT_BASELINE_LOGITS_FILE,
            target_logits_output=kl_d_bench.DEFAULT_TARGET_LOGITS_FILE,
            baseline_logits=None,
            target_logits=None,
            # Adding missing attributes with default or placeholder values
            context_size=2048,
            n_gpu_layers=1,
            threads=1,
            batch_size=2048,
            ubatch_size=512,
            precision=16,
            rope_freq_base=None,
            repeat_last_n=64,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            dynatemp_range=0.0,
            dynatemp_exp=1.0,
            mirostat=0,
            mirostat_lr=0.1,
            mirostat_ent=5.0,
            top_p=1.0,
            top_k=1,
            temp=1.0,
            seed=42,
            verbosity='INFO'
        )

        # Mock the h5py File context manager to return a mock object
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': 3}  # Set total_chunks attribute

        kl_d_bench.main(args)

        # Add assertions to check if the correct functions were called with the right arguments
        # For example:
        mock_generate_logits_with_llama_cpp.assert_called()
        mock_process_generate_both.assert_called()  # Check if generate_logits_with_llama_cpp was called

    @patch('builtins.open', new_callable=mock_open, read_data='test data')
    @patch('kl_d_bench.h5py.File')
    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    @patch('kl_d_bench.process_generate_one')
    def test_main_generate_baseline(self, mock_process_generate_one, mock_generate_logits_with_llama_cpp, mock_h5py_file, mock_open_file):
        """
        Test main function when both logits need to be generated.
        """
        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model=None,
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits=None,
            baseline_logits_output=kl_d_bench.DEFAULT_BASELINE_LOGITS_FILE,
            target_logits='target.h5',
            # Adding missing attributes with default or placeholder values
            context_size=2048,
            n_gpu_layers=1,
            threads=1,
            batch_size=2048,
            ubatch_size=512,
            precision=16,
            rope_freq_base=None,
            repeat_last_n=64,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            dynatemp_range=0.0,
            dynatemp_exp=1.0,
            mirostat=0,
            mirostat_lr=0.1,
            mirostat_ent=5.0,
            top_p=1.0,
            top_k=1,
            temp=1.0,
            seed=42,
            verbosity='INFO'
        )

        # Mock the h5py File context manager to return a mock object
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': 3}  # Set total_chunks attribute

        kl_d_bench.main(args)

        # Add assertions to check if the correct functions were called with the right arguments
        # For example:
        mock_generate_logits_with_llama_cpp.assert_not_called()
        mock_process_generate_one.assert_called()  # Check if generate_logits_with_llama_cpp was called

    @patch('builtins.open', new_callable=mock_open, read_data='test data')
    @patch('kl_d_bench.h5py.File')
    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    @patch('kl_d_bench.process_generate_one')
    def test_main_generate_target(self, mock_process_generate_one, mock_generate_logits_with_llama_cpp, mock_h5py_file, mock_open_file):
        """
        Test main function when both logits need to be generated.
        """
        args = argparse.Namespace(
            baseline_model=None,
            target_model='target.gguf',
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits='baseline.h5',
            target_logits_output=kl_d_bench.DEFAULT_TARGET_LOGITS_FILE,
            target_logits=None,
            # Adding missing attributes with default or placeholder values
            context_size=2048,
            n_gpu_layers=1,
            threads=1,
            batch_size=2048,
            ubatch_size=512,
            precision=16,
            rope_freq_base=None,
            repeat_last_n=64,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            dynatemp_range=0.0,
            dynatemp_exp=1.0,
            mirostat=0,
            mirostat_lr=0.1,
            mirostat_ent=5.0,
            top_p=1.0,
            top_k=1,
            temp=1.0,
            seed=42,
            verbosity='INFO'
        )

        # Mock the h5py File context manager to return a mock object
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': 3}  # Set total_chunks attribute

        kl_d_bench.main(args)

        mock_generate_logits_with_llama_cpp.assert_not_called()
        mock_process_generate_one.assert_called()  # Check if generate_logits_with_llama_cpp was called

    @patch('builtins.open', new_callable=mock_open, read_data='test data')
    @patch('kl_d_bench.h5py.File')
    @patch('kl_d_bench.generate_logits_with_llama_cpp')
    @patch('kl_d_bench.process_generate_neither')
    def test_main_generate_neither(self, mock_process_generate_neither, mock_generate_logits_with_llama_cpp, mock_h5py_file, mock_open_file):
        """
        Test main function when both logits need to be generated.
        """
        args = argparse.Namespace(
            baseline_model=None,
            target_model=None,
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            # Adding missing attributes with default or placeholder values
            baseline_logits='baseline.h5',
            target_logits='target.h5',
            context_size=2048,
            n_gpu_layers=1,
            threads=1,
            batch_size=2048,
            ubatch_size=512,
            precision=16,
            rope_freq_base=None,
            repeat_last_n=64,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            dynatemp_range=0.0,
            dynatemp_exp=1.0,
            mirostat=0,
            mirostat_lr=0.1,
            mirostat_ent=5.0,
            top_p=1.0,
            top_k=1,
            temp=1.0,
            seed=42,
            verbosity='INFO'
        )

        # Mock the h5py File context manager to return a mock object
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': 3}  # Set total_chunks attribute

        kl_d_bench.main(args)

        mock_generate_logits_with_llama_cpp.assert_not_called()
        mock_process_generate_neither.assert_called()  # Check if generate_logits_with_llama_cpp was called

    @patch('llama_cpp.Llama')
    def test_integration_kl_divergence_with_chunk_reuse(self, MockLlama):
        MockLlama.side_effect = lambda **kwargs: MockModel(**kwargs)

        # Create temporary files for dataset and output paths
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_dataset, \
             tempfile.NamedTemporaryFile('wb', delete=False) as temp_kl_output:
            try:
                # Prepare input text that results in exactly 4 chunks
                input_text = ("Chunk data. " * 42).strip()
                temp_dataset.write(input_text)
                temp_dataset.flush()

                # Create temporary file paths for baseline and target logits without opening them
                temp_baseline_output_path = tempfile.mktemp(suffix=".h5")
                temp_target_output_path = tempfile.mktemp(suffix=".h5")

                # Set up arguments for main() as if from the command line
                args = {
                    'baseline_model': 'path/to/mock/baseline/model',
                    'target_model': 'path/to/mock/target/model',
                    'baseline_logits_output': temp_baseline_output_path,
                    'target_logits_output': temp_target_output_path,
                    'dataset': temp_dataset.name,
                    'output_file': temp_kl_output.name,
                    'baseline_logits': None,
                    'target_logits': None,
                    'context_size': 128,  # With BOS and EOS tokens, we get 126 tokens per chunk
                    'rope_freq_base': None,
                    'verbosity': 'DEBUG',
                    'clobber': True,
                    'threads': 1,
                    'temp': 0,
                    'seed': 0,
                    'top_p': 0,
                    'top_k': 1,
                    'batch_size': 2048,
                    'ubatch_size': 512,
                    'from_chunk': None,
                    'to_chunk': None,
                    'repeat_last_n': 64,
                    'repeat_penalty': 1.0,
                    'presence_penalty': 0.0,
                    'frequency_penalty': 0.0,
                    'dynatemp_range': 0.0,
                    'dynatemp_exp': 1.0,
                    'mirostat': 0,
                    'mirostat_lr': 0.1,
                    'mirostat_ent': 5.0,
                    'n_gpu_layers': None,
                    'precision': None,
                }

                # Convert args to Namespace to match the signature expected by main()
                namespace_args = argparse.Namespace(**args)

                # Run the main function from kl_d_bench, which will use MockModel for llama_cpp.Llama
                kl_d_bench.main(namespace_args)

                # Verify that the output file was created
                self.assertTrue(os.path.exists(temp_kl_output.name), "KL-divergence output file was not created.")
                
                # Verify that the baseline and target logits files were created by generate_logits
                self.assertTrue(os.path.exists(temp_baseline_output_path), "Baseline logits file was not created.")
                self.assertTrue(os.path.exists(temp_target_output_path), "Target logits file was not created.")

                # Open the HDF5 output file and check its contents
                with h5py.File(temp_baseline_output_path, 'r') as h5f:
                    self.assertIn('logits', h5f, "Logits dataset not found in baseline logits output file.")
                    self.assertIn('processed_chunks', h5f, "Processed chunks dataset not found in baseline logits output file.")

                    # Check that exactly 4 chunks were processed
                    processed_chunks_count = len(h5f['processed_chunks'])
                    self.assertEqual(processed_chunks_count, 4, "Expected 4 processed chunks.")

                with h5py.File(temp_target_output_path, 'r') as h5f:
                    self.assertIn('logits', h5f, "Logits dataset not found in target logits output file.")
                    self.assertIn('processed_chunks', h5f, "Processed chunks dataset not found in target logits output file.")

                    # Check that exactly 4 chunks were processed
                    processed_chunks_count = len(h5f['processed_chunks'])
                    self.assertEqual(processed_chunks_count, 4, "Expected 4 processed chunks.")

                with h5py.File(temp_kl_output.name, 'r') as h5f:
                    self.assertIn('chunk_0', h5f, "Chunk statistics for chunk 0 not found in KL-divergence output file.")
                    self.assertIn('chunk_1', h5f, "Chunk statistics for chunk 1 not found in KL-divergence output file.")
                    self.assertIn('chunk_2', h5f, "Chunk statistics for chunk 2 not found in KL-divergence output file.")
                    self.assertIn('chunk_3', h5f, "Chunk statistics for chunk 3 not found in KL-divergence output file.")
                    self.assertIn('total_values', h5f.attrs, "Total values not found in KL-divergence output file.")
                    self.assertIn('overall', h5f.attrs, "Overall statistics not found in KL-divergence output file.")

            finally:
                # Clean up temporary files
                os.remove(temp_dataset.name)
                os.remove(temp_baseline_output_path)
                os.remove(temp_target_output_path)
                os.remove(temp_kl_output.name)

                
if __name__ == '__main__':
    unittest.main()