# Encoder
class Enc:
    def __init__(self, vocabulary: str):
        self.vocab = sorted(list(set(vocabulary)))
        self.int_to_str = {i: c for i, c in enumerate(self.vocab)}
        self.str_to_int = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, string: str) -> list[int]:
        return [self.str_to_int[s] for s in string]

    def decode(self, encoding: list[int]) -> str:
        return "".join([self.int_to_str[i] for i in encoding])

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)
