import numpy as np


class MockModel:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path

        self.metadata = {
            "tokenizer.ggml.add_bos_token": "true",
            "tokenizer.ggml.add_eos_token": "true",
        }
        self.n_tokens = 0
        self._vocab_size = 1000  
        self.n_ctx = 128
        self.scores = np.array([])
        # Handle any kwargs that may affect the model, if necessary

    def tokenize(self, text):
        # Simple tokenizer that converts characters to token IDs
        return [ord(c) for c in text.decode('utf-8')]

    def token_bos(self):
        return 1  # BOS token ID

    def token_eos(self):
        return 2  # EOS token ID

    def __call__(self, tokens):
        # Simulate model inference
        self.n_tokens = len(tokens)

        vocab_size = self.n_vocab()

        # Convert tokens to a NumPy array for vectorized operations
        tokens_array = np.array(tokens).reshape(-1, 1)  # Shape: (n_tokens, 1)
        vocab_indices = np.arange(vocab_size).reshape(1, -1)  # Shape: (1, vocab_size)

        # Example deterministic function:
        # scores[token][vocab_idx] = (token + vocab_idx) / (vocab_size + 1)
        # This ensures scores are between 0 and 1 and are reproducible
        self.scores = ((tokens_array + vocab_indices) % (self._vocab_size + 1) + 1) / (self._vocab_size + 1)
        self.scores = self.scores.astype(np.float32)

    def n_vocab(self):
        return self._vocab_size
