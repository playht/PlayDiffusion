"""The PlayTTS text tokenizer."""

import os
import torch
from tokenizers import Tokenizer

TOKENIZER_PATH = "voice_tokenizer-bpe_lowercase_asr_256.json"

class VoiceBpeTokenizer:
    def __init__(self, vocab_file = None):
        if vocab_file is None:
            vocab_file = TOKENIZER_PATH
        vocab_file = os.path.join(
            os.path.dirname(__file__), vocab_file
        )
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file)

    def encode(self, txt: str):
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")

        return txt
