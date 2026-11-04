"""Request timing data, independent of the CUDA execution stack."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SloStat:
    req_id: int
    # Detokenizer records milliseconds; post_process converts to seconds.
    t_prefill: float
    t_prefill_std: float
    t_decode: float
    t_tokens: List[float]
    t_token_timestamps: Optional[List[float]] = None

    def post_process(self) -> None:
        self.t_decode = (self.t_decode - self.t_prefill) * 1e-3
        self.t_prefill *= 1e-3
        self.t_prefill_std *= 1e-3
        self.t_token_timestamps = [t * 1e-3 for t in self.t_tokens]
        self.t_tokens = [(x - y) * 1e-3
                         for x, y in zip(self.t_tokens[1:], self.t_tokens[:-1])]
