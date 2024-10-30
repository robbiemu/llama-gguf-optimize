import contextlib 
import h5py
import llama_cpp
import logging
import numpy as np
import os
import random
import string
import tempfile
import unittest
from unittest.mock import patch

from mock_model import MockModel
from version import __version__
from generate_logits import (
    prepare_llama_args, estimate_disk_size, write_header, create_hdf5_dataset, 
    process_tokens_chunk, generate_logits_with_llama_cpp
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("generate_logits")


class TestGenerateLogits(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_prepare_llama_args(self):
        # Test to ensure unrelated arguments are not maintained in the output args
        input_args = {
            'model': 'mock_model.gguf',
            'batch_size': 1024,
            'unrelated_arg': 'should_be_removed'
        }
        expected_args = {
            'model_path': 'mock_model.gguf',
            'n_batch': 1024,
            'logits_all': True
        }
        output_args = prepare_llama_args(input_args)
        self.assertEqual(output_args, expected_args)

    def test_estimate_disk_size(self):
        # Ensure estimate_disk_size is callable
        self.assertTrue(callable(estimate_disk_size))

    def test_write_header_happy_path(self):
        # Happy path for write_header function, using a temporary HDF5 file
        with tempfile.NamedTemporaryFile() as tmp:
            with h5py.File(tmp.name, 'w') as h5f:
                context_size = 4096
                vocab_size = 32000
                write_header(h5f, context_size, vocab_size)

                # Assert that attributes were written correctly
                self.assertEqual(h5f.attrs['format'], f"generate_logits_v{__version__}")
                self.assertEqual(h5f.attrs['n_ctx'], context_size)
                self.assertEqual(h5f.attrs['n_vocab'], vocab_size)


class TestCreateHDF5Dataset(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_create_hdf5_dataset(self):
        # Parameters
        output_file = 'test_logits.h5'
        total_chunks = 10
        vocab_size = 1000
        context_size = 128
        precision = 16
        resume = False

        # Ensure output file does not exist
        if os.path.exists(output_file):
            os.remove(output_file)

        # Call the function
        h5f, dset, processed_chunks_dset = create_hdf5_dataset(
            output_file, total_chunks, vocab_size, context_size, precision, resume=resume
        )

        # Check that file exists
        self.assertTrue(os.path.exists(output_file), "Output file was not created.")

        # Check datasets exist
        self.assertIn('logits', h5f, "'logits' dataset not found in HDF5 file.")
        self.assertIn('processed_chunks', h5f, "'processed_chunks' dataset not found in HDF5 file.")

        # Check shapes
        self.assertEqual(dset.shape, (total_chunks, context_size, vocab_size), "Logits dataset shape mismatch.")
        self.assertEqual(processed_chunks_dset.shape, (total_chunks,), "Processed chunks dataset shape mismatch.")

        # Check dtypes
        expected_dtype = np.float16 if precision <= 16 else np.float32
        self.assertEqual(dset.dtype, expected_dtype, "Logits dataset dtype mismatch.")

        # Check attributes
        self.assertEqual(h5f.attrs['format'], f"generate_logits_v{__version__}", "Incorrect format attribute.")
        self.assertEqual(h5f.attrs['n_ctx'], context_size, "Incorrect n_ctx attribute.")
        self.assertEqual(h5f.attrs['n_vocab'], vocab_size, "Incorrect n_vocab attribute.")

        # Cleanup
        h5f.close()
        os.remove(output_file)

class TestProcessTokensChunk(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_process_tokens_chunk(self):
        # Create a mock model
        model = MockModel()

        # Tokens chunk
        tokens_chunk = [3, 4, 5, 6, 7]

        # Create a dummy dataset
        chunk_index = 0
        total_chunks = 1
        context_size = len(tokens_chunk) + 2  # Account for BOS and EOS
        vocab_size = model.n_vocab()
        dtype = np.float32

        # Create a dummy HDF5 dataset
        with h5py.File('test_logits.h5', 'w') as h5f:
            dset = h5f.create_dataset(
                'logits',
                shape=(total_chunks, context_size, vocab_size),
                dtype=dtype
            )
            # Call the function
            timing_info = process_tokens_chunk(model, tokens_chunk, dset, chunk_index)

            # Check that the logits have been written
            self.assertEqual(dset[chunk_index].shape, (context_size, vocab_size), "Logits data shape mismatch.")

            # Check that the data is not all zeros (since we used random data)
            self.assertTrue(np.any(dset[chunk_index] != 0), "Logits data not written.")

        # Cleanup
        os.remove('test_logits.h5')


class TestProcessTokensChunk(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_process_tokens_chunk(self):
        # Create a mock model
        model = MockModel('foo')

        # Tokens chunk
        tokens_chunk = [3, 4, 5, 6, 7]

        # Create a dummy dataset
        chunk_index = 0
        total_chunks = 1
        context_size = len(tokens_chunk) + 2  # Account for BOS and EOS
        vocab_size = model.n_vocab()
        dtype = np.float32

        # Create a dummy HDF5 dataset
        with h5py.File('test_logits.h5', 'w') as h5f:
            dset = h5f.create_dataset(
                'logits',
                shape=(total_chunks, context_size, vocab_size),
                dtype=dtype
            )
            # Call the function
            timing_info = process_tokens_chunk(model, tokens_chunk, dset, chunk_index)

            # Check that the logits have been written
            self.assertEqual(dset[chunk_index].shape, (context_size, vocab_size), "Logits data shape mismatch.")

            # Check that the data is not all zeros (since we used random data)
            self.assertTrue(np.any(dset[chunk_index] != 0), "Logits data not written.")

        # Cleanup
        os.remove('test_logits.h5')


class TestGenerateLogitsWithLlamaCpp(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_generate_logits_with_llama_cpp(self):
        # Prepare arguments
        kwargs = {
            'model': 'path/to/mock/model',
            'context_size': 128,
            'dataset': 'test_dataset.txt',
            'output': 'test_logits.h5',
            'verbosity': 'DEBUG',
            'clobber': True,
            'threads': 1,
            'batch_size': 2048,
            'ubatch_size': 512,
            'from': 0,
            'to': None,
            'rope_freq_base': None,
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

        with tempfile.NamedTemporaryFile('w', delete=True) as input_file, \
            tempfile.NamedTemporaryFile('w', delete=True) as output_file:

            input_text = ''.join(random.choices(string.ascii_letters + string.digits, k=256))
            input_file.write(input_text)
            input_file.flush()  # Ensure content is written to disk
            
            kwargs['dataset'] = input_file.name  # Pass the temporary input file
            kwargs['output'] = output_file.name  # Set the output to a temporary file

            # Mock the Llama model
            with patch('llama_cpp.Llama', new=MockModel):
                # Call the function
                generate_logits_with_llama_cpp(**kwargs)

                # Check that the output file is created and contains expected datasets
                with h5py.File(output_file.name, 'r') as h5f:
                    self.assertIn('logits', h5f, "'logits' dataset not found in output file.")
                    self.assertIn('processed_chunks', h5f, "'processed_chunks' dataset not found in output file.")
                    # Additional checks can be added here, such as verifying data contents


if __name__ == '__main__':
    unittest.main()
