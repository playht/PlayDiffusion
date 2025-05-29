from typing import List

import torch

from playdiffusion.models.tokenizer.voice_tokenizer import VoiceBpeTokenizer

class PPTokenizer:
    def __init__(self, vocab_file = None, device = None):
        self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

        # store optional device
        self.device = torch.device(device) if device else torch.device('cpu')

    def encoded_to_tensor(self, etxt, device = None):
        d = device or self.device
        return torch.tensor(etxt, dtype=torch.int32, device=d)[None]

    def encode_normalized(self, ntxt: str):
        return self.tokenizer.encode(ntxt)

    def encode_normalized_to_tensor(self, ntxt: str, device = None):
        tokens = self.encode_normalized(ntxt)
        return self.encoded_to_tensor(tokens, device)

    def decode_tokens(self, encoded: List[int]):
        return self.tokenizer.decode(encoded)

    def tensor_to_encoded(self, tensor: torch.Tensor):
        return tensor.squeeze().tolist()

    def decode_tokens_tensor(self, tensor: torch.Tensor):
        encoded = self.tensor_to_encoded(tensor)
        return self.decode_tokens(encoded)
