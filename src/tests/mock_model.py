import numpy as np


class MockModel:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path

        self.metadata = {
            "tokenizer.ggml.add_bos_token": "true",
            "tokenizer.ggml.add_eos_token": "true",
        }
        self.n_tokens = 0
        self.scores = np.array([])
        self._vocab_size = 1000  
        self.n_ctx = 128
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
        # Generate random scores for each token
        self.scores = np.random.rand(self.n_tokens, vocab_size).astype(np.float32)

    def n_vocab(self):
        return self._vocab_size
