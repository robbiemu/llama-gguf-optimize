import h5py
import llama_cpp
import logging
import numpy as np
import os
import random
import string
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from mock_model import MockModel
from version import __version__
from generate_logits import (
    prepare_llama_args, prepare_call_args, write_header, 
    create_hdf5_datasets, process_single_chunk, 
    generate_logits_with_llama_cpp
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("generate_logits")


class TestPrepareCallArgs(unittest.TestCase):
    def test_prepare_call_args(self):
        kwargs = {
            'temp': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'min_p': 0.1,
            'repeat_penalty': 1.1,
            'presence_penalty': 0.2,
            'frequency_penalty': 0.3,
            'seed': 42,
            'mirostat': 1,
            'mirostat_ent': 5.0,
            'mirostat_lr': 0.1,
        }
        expected_call_args = {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'min_p': 0.1,
            'repeat_penalty': 1.1,
            'presence_penalty': 0.2,
            'frequency_penalty': 0.3,
            'seed': 42,
            'mirostat_mode': 1,
            'mirostat_tau': 5.0,
            'mirostat_eta': 0.1,
        }
        call_args = prepare_call_args(kwargs)
        self.assertEqual(call_args, expected_call_args)


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

    def test_write_header_happy_path(self):
        # Happy path for write_header function, using a temporary HDF5 file
        with tempfile.NamedTemporaryFile() as tmp:
            with h5py.File(tmp.name, 'w') as h5f:
                context_size = 4096
                vocab_size = 32000
                total_chunks = 1
                write_header(h5f, context_size, vocab_size, total_chunks)

                # Assert that attributes were written correctly
                self.assertEqual(h5f.attrs['format'], f"generate_logits_v{__version__}")
                self.assertEqual(h5f.attrs['n_ctx'], context_size)
                self.assertEqual(h5f.attrs['n_vocab'], vocab_size)


class TestCreateHDF5Dataset(unittest.TestCase):
    test_logits_file = 'test_logits.h5'

    def setUp(self):
        if os.path.exists(self.test_logits_file):
            os.remove(self.test_logits_file)

        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        if os.path.exists(self.test_logits_file):
            os.remove(self.test_logits_file)

        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_create_hdf5_datasets(self):
        # Parameters
        output_file = self.test_logits_file
        total_chunks = 10
        vocab_size = 1000
        context_size = 128
        precision = 16
        compression = 'gzip'  # or any other supported compression method
        resume = False

        with h5py.File(output_file, 'w') as h5f:
            # Call the function
            dset, processed_chunks_dset, freed_chunks_dset, chunk_index_dset = create_hdf5_datasets(
                h5f, total_chunks, vocab_size, context_size, precision, compression, resume=resume
            )

            # Check that file exists
            self.assertTrue(os.path.exists(output_file), "Output file was not created.")

            # Check datasets exist
            self.assertIn('logits', h5f, "'logits' dataset not found in HDF5 file.")
            self.assertIn('processed_chunks', h5f, "'processed_chunks' dataset not found in HDF5 file.")

            # Check 'freed_chunks' dataset exists
            self.assertIn('freed_chunks', h5f, "'freed_chunks' dataset not found in HDF5 file.")

            # Check shape and dtype
            self.assertEqual(h5f['freed_chunks'].shape, (0,), "Initial 'freed_chunks' dataset shape mismatch.")
            self.assertEqual(h5f['freed_chunks'].dtype, np.int64, "'freed_chunks' dataset dtype mismatch.")

            # Check attributes
            self.assertEqual(h5f.attrs['format'], f"generate_logits_v{__version__}", "Incorrect format attribute.")
            self.assertEqual(h5f.attrs['n_ctx'], context_size, "Incorrect n_ctx attribute.")
            self.assertEqual(h5f.attrs['n_vocab'], vocab_size, "Incorrect n_vocab attribute.")


class TestProcessSingleChunk(unittest.TestCase):
    test_file = "test_logits.h5"

    def setUp(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    @patch('llama_cpp.llama_cpp.llama_kv_cache_clear', return_value=None)
    def test_process_single_chunk(self, mock_llama_kv_cache_clear):
        # Create a mock model
        model = MockModel('dummy model')

        # Tokens chunk
        tokens_chunk = [3, 4, 5, 6, 7]

        # Prepare call_args
        kwargs = {
            'temp': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'seed': 42
        }
        call_args = prepare_call_args(kwargs)

        # Create a dummy dataset
        chunk_index = 0
        total_chunks = 1
        context_size = len(tokens_chunk) + 2  # Account for BOS and EOS
        vocab_size = model.n_vocab()
        dtype = np.float32

        # Create a dummy HDF5 dataset
        with h5py.File(self.test_file, 'w') as h5f:
            dset = h5f.create_dataset(
                'logits',
                shape=(total_chunks, context_size, vocab_size),
                dtype=dtype
            )
            freed_chunks_dset = h5f.create_dataset(
                'freed_chunks',
                shape=(0,),
                dtype=np.int64,
                maxshape=(None,),
                chunks=True
            )
            chunk_index_dset = h5f.create_dataset(
                'chunk_index',
                shape=(total_chunks,),
                dtype=np.int64,
                chunks=True
            )
            chunk_index_dset[...] = -1  # Initialize with -1

            # Call the function
            timing_info = process_single_chunk(
                model, call_args, tokens_chunk, dset, chunk_index,
                freed_chunks_dset, chunk_index_dset
            )

            # Assertions
            self.assertEqual(dset[chunk_index].shape, (context_size, vocab_size))
            self.assertTrue(np.any(dset[chunk_index] != 0))

    @patch('llama_cpp.llama_cpp.llama_kv_cache_clear', return_value=None)
    def test_reset_chunk_in_hdf5(self, mock_llama_kv_cache_clear):
        # Use the MockModel
        model = MockModel(model_path="dummy model")
        model.metadata = {
            "tokenizer.ggml.add_bos_token": "false",
            "tokenizer.ggml.add_eos_token": "false",
        }
        input_text = ''.join(random.choices(string.ascii_letters + string.digits, k=100)).encode('utf-8')
        tokens_chunk = model.tokenize(input_text)

        chunk_index = 3  # Initial chunk index

        with h5py.File(self.test_file, 'w') as h5f:
            # Create datasets
            dset = h5f.create_dataset('logits', shape=(10, 100, model.n_vocab()), dtype=np.float32)
            processed_chunks_dset = h5f.create_dataset('processed_chunks', shape=(10,), dtype=bool)
            freed_chunks_dset = h5f.create_dataset('freed_chunks', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
            chunk_index_dset = h5f.create_dataset('chunk_index', shape=(10,), dtype=np.int64, chunks=True)
            chunk_index_dset[...] = -1

            llama_cpp.llama_cpp.llama_kv_cache_clear

            # Write the first chunk
            call_args = prepare_call_args({})
            process_single_chunk(
                model, call_args, tokens_chunk, dset, chunk_index,
                freed_chunks_dset, chunk_index_dset
            )

            # Reset the chunk
            freed_chunks_dset.resize(freed_chunks_dset.shape[0] + 1, axis=0)
            freed_chunks_dset[-1] = chunk_index

            # Write another chunk, reusing the freed chunk index
            process_single_chunk(
                model, call_args, tokens_chunk, dset, chunk_index,
                freed_chunks_dset, chunk_index_dset
            )

            # Assertions
            self.assertEqual(freed_chunks_dset.size, 0)
            self.assertTrue(np.array_equal(dset[chunk_index, :, :], model.scores))


class TestGenerateLogitsWithLlamaCpp(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable all logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    @patch('llama_cpp.Llama', new_callable=MagicMock)
    def test_generate_logits_with_llama_cpp(self, MockLlama):
        # Prepare arguments
        kwargs = {
            'model': 'path/to/mock/model',
            'context_size': 128,
            'dataset': 'test_dataset.txt',
            'output': 'test_logits.h5',
            'verbosity': 'DEBUG',
            'clobber': True,
            'threads': 1,
            'batch_size': 128,
            'ubatch_size': 128,
            'from_chunk': 0,
            'repeat_last_n': 64,
            'repeat_penalty': 1.0,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'temp': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'seed': 42,
            'compression': 'none',
            'precision': 32,
            # Add any other required arguments here
        }

        model_args = prepare_llama_args(kwargs)
        mock_model_instance = MockModel(**model_args)
        MockLlama.return_value = mock_model_instance

        with tempfile.NamedTemporaryFile('w', delete=True) as input_file, \
                tempfile.NamedTemporaryFile('w', delete=True) as output_file:

            input_text = ''.join(random.choices(string.ascii_letters + string.digits, k=256))
            input_file.write(input_text)
            input_file.flush()
            
            kwargs['dataset'] = input_file.name
            kwargs['output'] = output_file.name

            # Call the function
            generate_logits_with_llama_cpp(**kwargs)

            # Check that the output file is created and contains expected datasets
            with h5py.File(output_file.name, 'r') as h5f:
                self.assertIn('logits', h5f)
                self.assertIn('processed_chunks', h5f)
                self.assertIn('chunk_index', h5f)


if __name__ == '__main__':
    unittest.main()
