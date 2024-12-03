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
        if os.path.exists('target_logits.h5'):
            os.remove('target_logits.h5')

        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        if os.path.exists('baseline_logits.h5'):
            os.remove('baseline_logits.h5')
        if os.path.exists('target_logits.h5'):
            os.remove('target_logits.h5')

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
        mock_dataset.resize.assert_called_once_with((6,))  # Correct expected call
        
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
            model_path='dummy model', output='test.h5', clobber=False, from_chunk=3, to_chunk=3
        )

    @patch('kl_d_bench.process_chunks')
    def test_compare_logits_for_chunk(self, mock_process_chunks):
        """
        Test compare_logits_for_chunk function.
        """
        compare_args = {'output_path': 'kl_divergence.h5'}
        kl_d_bench.compare_logits_for_chunk(compare_args, 2)

        mock_process_chunks.assert_called_once_with(
            output_path='kl_divergence.h5', from_chunk=2, to_chunk=2
        )

    @patch('kl_d_bench.reset_previous_chunk')
    @patch('kl_d_bench.compare_logits_for_chunk')
    @patch('kl_d_bench.calculate_and_update_compare_args')
    @patch('kl_d_bench.generate_logits_for_model')
    def test_process_generate_both(
        self, mock_generate_logits_for_model, mock_calculate_and_update_compare_args,
        mock_compare_logits_for_chunk, mock_reset_previous_chunk
    ):
        """
        Test process_generate_both function.
        """
        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model='target.gguf',
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits_output='baseline_logits.h5',
            target_logits_output='target_logits.h5',
            from_chunk=0,
            to_chunk=2,
            clobber=False,
            early_stopping=False,
            compute_overall=False,
            keep=0,
        )
        generate_args_baseline = {'output': 'baseline_logits.h5'}
        generate_args_target = {'output': 'target_logits.h5'}
        total_chunks = 3
        compare_args = {'output_path': 'kl_divergence.h5'}

        # Call the function under test
        kl_d_bench.process_generate_both(
            args, total_chunks, generate_args_baseline, generate_args_target, compare_args
        )

        # Assertions
        self.assertEqual(mock_generate_logits_for_model.call_count, 6)  # 3 chunks * 2 models
        self.assertEqual(mock_compare_logits_for_chunk.call_count, 3)
        self.assertEqual(mock_reset_previous_chunk.call_count, 3)
        mock_calculate_and_update_compare_args.assert_not_called()  # early_stopping is False

    @patch('kl_d_bench.reset_previous_chunk')
    @patch('kl_d_bench.compare_logits_for_chunk')
    @patch('kl_d_bench.calculate_and_update_compare_args')
    @patch('kl_d_bench.generate_logits_for_model')
    def test_process_generate_one(
        self, mock_generate_logits_for_model, mock_calculate_and_update_compare_args,
        mock_compare_logits_for_chunk, mock_reset_previous_chunk
    ):
        """
        Test process_generate_one function.
        """
        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model=None,
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits=None,
            baseline_logits_output='baseline_logits.h5',
            target_logits='target_logits.h5',
            from_chunk=0,
            to_chunk=2,
            clobber=False,
            early_stopping=False,
            compute_overall=False,
            keep=0,
        )
        generate_args_baseline = {'output': 'baseline_logits.h5'}
        generate_args_target = {'output': 'target_logits.h5'}
        compare_args = {'output_path': 'kl_divergence.h5'}

        # Call the function under test
        kl_d_bench.process_generate_one(
            args, 3, generate_args_baseline, generate_args_target, compare_args
        )

        # Assertions
        self.assertEqual(mock_generate_logits_for_model.call_count, 2)  # Only baseline needs to be generated
        self.assertEqual(mock_compare_logits_for_chunk.call_count, 2)
        self.assertEqual(mock_reset_previous_chunk.call_count, 2)
        mock_calculate_and_update_compare_args.assert_not_called()  # early_stopping is False

    @patch('kl_d_bench.compare_logits_for_chunk')
    @patch('kl_d_bench.calculate_and_update_compare_args')
    def test_process_generate_neither(
        self, mock_calculate_and_update_compare_args, mock_compare_logits_for_chunk
    ):
        """
        Test process_generate_neither function.
        """
        args = argparse.Namespace(
            baseline_model=None,
            target_model=None,
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits='baseline.h5',
            target_logits='target.h5',
            from_chunk=0,
            to_chunk=2,
            early_stopping=False,
            compute_overall=False,
            clobber=False,
        )
        compare_args = {'output_path': 'kl_divergence.h5'}

        # Call the function under test
        kl_d_bench.process_generate_neither(args, 3, compare_args)

        # Assertions
        self.assertEqual(mock_compare_logits_for_chunk.call_count, 2)
        mock_calculate_and_update_compare_args.assert_not_called()  # early_stopping is False

    @patch('kl_d_bench.process_generate_both')
    @patch('kl_d_bench.calculate_total_chunks')
    @patch('kl_d_bench.tokenize_dataset')
    @patch('kl_d_bench.get_model')
    @patch('kl_d_bench.h5py.File')
    def test_main_generate_both(
        self, mock_h5py_file, mock_get_model, mock_tokenize_dataset, mock_calculate_total_chunks, mock_process_generate_both
    ):
        """
        Test main function when both logits need to be generated.
        """
        # Mock model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock dataset tokenization
        mock_tokenize_dataset.return_value = (None, 1000)  # Simulate tokenization output with 1000 tokens

        # Mock chunk calculation
        mock_calculate_total_chunks.return_value = 3  # Simulate 3 chunks for processing

        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model='target.gguf',
            dataset='dataset.txt',  # This file is mocked and won't actually be accessed
            output_file='kl_divergence.h5',
            baseline_logits_output='baseline_logits.h5',
            target_logits_output='target_logits.h5',
            baseline_logits=None,
            target_logits=None,
            keep=0,
            from_chunk=0,
            to_chunk=None,
            clobber=False,
            compute_overall=False,
            early_stopping=False,
            confidence=0.95,
            margin_of_error=0.05,
            min_chunks=None,
            window_size=None,
            learning_rate=0.01,
            min_prior_weight=0.1,
            momentum=0.9,
            decay_rate=None,
            theta_E=0.1,
            theta_P=0.05,
            n_gpu_layers=None,
            threads=1,
            context_size=2048,
            batch_size=2048,
            ubatch_size=512,
            model_precision=32,
            kld_precision=64,
            compression=None,
            parts=1,
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
            verbosity='INFO',
            dynamic_thresholds_enabled=False,
            log_effect_sizes=False,
        )

        # Call the function under test
        kl_d_bench.main(args)

        # Assertions
        mock_get_model.assert_called_once()
        mock_tokenize_dataset.assert_called_once_with(mock_model, 'dataset.txt')
        mock_calculate_total_chunks.assert_called_once_with(1000, 2048, mock_model)
        mock_process_generate_both.assert_called_once()

    @patch('kl_d_bench.process_generate_one')
    @patch('kl_d_bench.calculate_total_chunks')
    @patch('kl_d_bench.tokenize_dataset')
    @patch('kl_d_bench.get_model')
    @patch('kl_d_bench.h5py.File')
    def test_main_generate_baseline(
        self, mock_h5py_file, mock_get_model, mock_tokenize_dataset, mock_calculate_total_chunks, mock_process_generate_one
    ):
        """
        Test main function when only baseline logits need to be generated.
        """
        # Mock HDF5 file access for target logits
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs.get.return_value = 3  # Simulate 3 chunks from the target logits

        args = argparse.Namespace(
            baseline_model='baseline.gguf',  # Baseline model provided for generation
            target_model=None,
            dataset='dataset.txt',
            output_file='kl_divergence.h5',
            baseline_logits=None,  # No pre-generated baseline logits
            baseline_logits_output='baseline_logits.h5',
            target_logits='target.h5',  # Target logits provided
            target_logits_output='target_logits.h5',
            keep=0,
            from_chunk=0,
            to_chunk=None,
            clobber=False,
            compute_overall=False,
            early_stopping=False,
            confidence=0.95,
            margin_of_error=0.05,
            min_chunks=None,
            window_size=None,
            learning_rate=0.01,
            min_prior_weight=0.1,
            momentum=0.9,
            decay_rate=None,
            theta_E=0.1,
            theta_P=0.05,
            n_gpu_layers=None,
            threads=1,
            context_size=2048,
            batch_size=2048,
            ubatch_size=512,
            model_precision=32,
            kld_precision=64,
            compression=None,
            parts=1,
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
            verbosity='INFO',
            dynamic_thresholds_enabled=False,
            log_effect_sizes=False,
        )

        # Call the function under test
        kl_d_bench.main(args)

        # Assertions
        mock_h5py_file.assert_called_once_with('target.h5', 'r')  # Ensure target logits file is accessed
        mock_file.attrs.get.assert_called_once_with('total_chunks', 0)

        # Ensure get_model, tokenize_dataset, and calculate_total_chunks are NOT called
        mock_get_model.assert_not_called()
        mock_tokenize_dataset.assert_not_called()
        mock_calculate_total_chunks.assert_not_called()

        # Ensure process_generate_one is called
        mock_process_generate_one.assert_called_once()


    @patch('kl_d_bench.process_generate_both')
    @patch('kl_d_bench.calculate_total_chunks')
    @patch('kl_d_bench.tokenize_dataset')
    @patch('kl_d_bench.get_model')
    @patch('kl_d_bench.h5py.File')
    def test_main_generate_target(
        self, mock_h5py_file, mock_get_model, mock_tokenize_dataset, mock_calculate_total_chunks, mock_process_generate_both
    ):
        """
        Test main function when only target logits need to be generated.
        """
        # Mock model return value
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock tokenize_dataset return value
        mock_tokenize_dataset.return_value = (None, 1000)  # Simulate 1000 total tokens

        # Mock calculate_total_chunks return value
        mock_calculate_total_chunks.return_value = 10  # Simulate 10 chunks

        args = argparse.Namespace(
            baseline_model='baseline.gguf',
            target_model='target.gguf',
            baseline_logits=None,
            target_logits=None,
            baseline_logits_output='baseline_logits.h5',
            target_logits_output='target_logits.h5',
            dataset=None,
            output_file='kl_divergence.h5',
            keep=0,
            from_chunk=0,
            to_chunk=None,
            clobber=False,
            compute_overall=False,
            early_stopping=False,
            confidence=0.95,
            margin_of_error=0.05,
            min_chunks=None,
            window_size=None,
            learning_rate=0.01,
            min_prior_weight=0.1,
            momentum=0.9,
            decay_rate=None,
            theta_E=0.1,
            theta_P=0.05,
            n_gpu_layers=None,
            threads=1,
            context_size=2048,
            batch_size=2048,
            ubatch_size=512,
            model_precision=32,
            kld_precision=64,
            compression=None,
            parts=1,
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
            verbosity='INFO',
            dynamic_thresholds_enabled=False,
            log_effect_sizes=False,
        )

        mock_calculate_total_chunks.return_value = 3

        # Call the function under test
        kl_d_bench.main(args)

        # Assertions
        mock_process_generate_both.assert_called_once()
        mock_get_model.assert_called_once()
        mock_calculate_total_chunks.assert_called_once()


    @patch('kl_d_bench.process_generate_neither')
    @patch('kl_d_bench.h5py.File')
    def test_main_generate_neither(self, mock_h5py_file, mock_process_generate_neither):
        """
        Test main function when both logits files are supplied.
        """
        args = argparse.Namespace(
            baseline_model=None,
            target_model=None,
            dataset=None,
            baseline_logits='baseline.h5',
            target_logits='target.h5',
            baseline_logits_output=None,
            target_logits_output=None,
            output_file='kl_divergence.h5',
            keep=0,
            from_chunk=0,
            to_chunk=None,
            clobber=False,
            compute_overall=False,
            early_stopping=False,
            confidence=0.95,
            margin_of_error=0.05,
            min_chunks=None,
            window_size=None,
            learning_rate=0.01,
            min_prior_weight=0.1,
            momentum=0.9,
            decay_rate=None,
            theta_E=0.1,
            theta_P=0.05,
            n_gpu_layers=None,
            threads=1,
            context_size=2048,
            batch_size=2048,
            ubatch_size=512,
            model_precision=32,
            kld_precision=64,
            compression=None,
            parts=1,
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
            verbosity='INFO',
            dynamic_thresholds_enabled=False,
            log_effect_sizes=False,
        )

        # Mock total_chunks from h5py.File
        mock_file = mock_h5py_file.return_value.__enter__.return_value
        mock_file.attrs = {'total_chunks': 3}

        # Call the function under test
        kl_d_bench.main(args)

        # Assertions
        mock_process_generate_neither.assert_called_once()

    @patch('kl_d_bench.get_model')
    @patch('kl_d_bench.tokenize_dataset')
    @patch('kl_d_bench.calculate_total_chunks')
    @patch('kl_d_bench.process_generate_both')
    def test_integration_kl_divergence_with_chunk_reuse(
        self, mock_process_generate_both, mock_calculate_total_chunks, mock_tokenize_dataset, mock_get_model
    ):
        """
        Integration test for KL divergence calculation with chunk reuse.
        """
        # Mock necessary functions
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_tokenize_dataset.return_value = (None, 1000)  # Simulate 1000 tokens
        mock_calculate_total_chunks.return_value = 4  # Simulate 4 chunks

        # Create temporary files
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_dataset, \
            tempfile.NamedTemporaryFile('wb', delete=False) as temp_kl_output, \
            tempfile.NamedTemporaryFile('wb', delete=False, suffix='.h5') as temp_baseline_output, \
            tempfile.NamedTemporaryFile('wb', delete=False, suffix='.h5') as temp_target_output:

            try:
                # Write mock dataset content
                input_text = ("Chunk data. " * 42).strip()  # Simulates 4 chunks
                temp_dataset.write(input_text)
                temp_dataset.flush()

                # Set up arguments for main() as if from the command line
                namespace_args = argparse.Namespace(
                    baseline_model='path/to/mock/baseline/model',
                    target_model='path/to/mock/target/model',
                    baseline_logits_output=temp_baseline_output.name,
                    target_logits_output=temp_target_output.name,
                    dataset=temp_dataset.name,
                    output_file=temp_kl_output.name,
                    baseline_logits=None,
                    target_logits=None,
                    context_size=128,
                    rope_freq_base=None,
                    verbosity='DEBUG',
                    clobber=True,
                    threads=1,
                    temp=0,
                    seed=0,
                    top_p=0,
                    top_k=1,
                    batch_size=2048,
                    ubatch_size=512,
                    from_chunk=None,
                    to_chunk=None,
                    repeat_last_n=64,
                    repeat_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    dynatemp_range=0.0,
                    dynatemp_exp=1.0,
                    mirostat=0,
                    mirostat_lr=0.1,
                    mirostat_ent=5.0,
                    n_gpu_layers=None,
                    precision=None,
                    kld_precision=32,
                    model_precision=32,
                    compression=None,
                    parts=1,
                    dynamic_thresholds_enabled=False,
                    early_stopping=False,
                    confidence=0.95,
                    margin_of_error=0.05,
                    min_chunks=None,
                    window_size=None,
                    learning_rate=0.01,
                    min_prior_weight=0.1,
                    momentum=0.9,
                    decay_rate=None,
                    theta_E=0.1,
                    theta_P=0.05,
                    compute_overall=False,
                    log_effect_sizes=False,
                    keep=0,
                )

                # Call the function under test
                kl_d_bench.main(namespace_args)

                # Verify mock calls
                mock_get_model.assert_called()
                mock_tokenize_dataset.assert_called()
                mock_calculate_total_chunks.assert_called_once_with(1000, 128, mock_model)
                mock_process_generate_both.assert_called_once()

            finally:
                # Clean up temporary files
                os.remove(temp_dataset.name)
                os.remove(temp_kl_output.name)
                os.remove(temp_baseline_output.name)
                os.remove(temp_target_output.name)


if __name__ == '__main__':
    unittest.main()