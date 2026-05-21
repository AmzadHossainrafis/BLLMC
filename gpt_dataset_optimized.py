"""
Optimized GPTDataset — drop-in replacement for GPTDatasetV1.

Key improvements
----------------
1. O(1) memory overhead  — stores ONE flat token tensor; slices on __getitem__
2. Zero redundancy       — input & target share the same underlying storage
3. Fast GPU transfer     — pin_memory + num_workers in create_dataloader()
4. Optional disk cache   — tokenise once, reload in milliseconds next run
5. Deterministic seeding — worker_init_fn for reproducible multi-worker runs
"""

import random
from pathlib import Path
from typing import Optional, Union

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

# ── 1. Dataset ──────────────────────────────────────────────────────────────


class GPTDataset(Dataset):
    """
    Sliding-window dataset for causal-LM pretraining.

    Memory layout
    -------------
    Only ONE tensor (`self.tokens`) is kept.  Every __getitem__ call
    returns two *views* into it — no extra allocation per sample.

                tokens: [t0, t1, t2, t3, t4, t5, ...]
    window i=0  input : [t0, t1, t2, t3]   (max_length=4)
                target: [t1, t2, t3, t4]
    window i=1  input : [t2, t3, t4, t5]   (stride=2)
                target: [t3, t4, t5, t6]

    Parameters
    ----------
    txt        : raw text string.
    tokenizer  : tiktoken (or any object with .encode()).
    max_length : context window size.
    stride     : step between windows.  stride == max_length → no overlap.
    cache_path : optional .pt file to save/load the token tensor.
    """

    def __init__(
        self,
        txt: str,
        tokenizer,
        max_length: int,
        stride: int,
        cache_path: Optional[Union[str, Path]] = None,
    ):
        self.max_length = max_length
        self.stride = stride

        # ── tokenise (or load cache) ────────────────────────────────────────
        self.tokens = self._load_tokens(txt, tokenizer, cache_path)

        # ── precompute start indices (ints, not tensors) ────────────────────
        # This tiny list is the ONLY per-chunk data we store.
        n = len(self.tokens)
        self.starts = range(0, n - max_length, stride)

    # ── tokenisation / caching ───────────────────────────────────────────────

    @staticmethod
    def _load_tokens(
        txt: str,
        tokenizer,
        cache_path: Optional[Union[str, Path]],
    ) -> torch.Tensor:

        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f"[GPTDataset] Loading token cache: {cache_path}")
                return torch.load(cache_path)

        print("[GPTDataset] Tokenising … ", end="", flush=True)
        ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        tokens = torch.tensor(ids, dtype=torch.long)
        print(f"{len(tokens):,} tokens.")

        if cache_path:
            torch.save(tokens, cache_path)
            print(f"[GPTDataset] Cache saved → {cache_path}")

        return tokens

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        end = start + self.max_length + 1  # +1 so target needs no copy

        chunk = self.tokens[start:end]  # single contiguous slice
        input_ids = chunk[:-1]  # view  — zero extra memory
        target_ids = chunk[1:]  # view  — zero extra memory

        return input_ids, target_ids


# ── 2. DataLoader factory ────────────────────────────────────────────────────


def _worker_seed(worker_id: int) -> None:
    """Ensure each DataLoader worker gets a distinct, reproducible seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def create_dataloader(
    txt: str,
    tokenizer,
    max_length: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
) -> DataLoader:
    """
    Build an optimised DataLoader for GPT pretraining.

    Tips
    ----
    * Set num_workers = os.cpu_count() // 2  on multi-core machines.
    * pin_memory=True speeds up CPU→GPU copies (requires CUDA).
    * Use cache_path to avoid re-tokenising on every run.
    """
    dataset = GPTDataset(txt, tokenizer, max_length, stride, cache_path=cache_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_worker_seed if num_workers > 0 else None,
        # Stack pre-built tensors — no custom collation needed.
        persistent_workers=num_workers > 0,
    )


# ── 3. Memory comparison helper ──────────────────────────────────────────────


def compare_memory(txt: str, tokenizer, max_length: int = 256, stride: int = 128):
    """Print a before/after memory comparison (requires psutil)."""
    try:
        import psutil
        import os

        proc = psutil.Process(os.getpid())

        def rss_mb():
            return proc.memory_info().rss / 1024**2

        base = rss_mb()

        # Old approach — materialise all chunks
        ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        old_inputs = []
        old_targets = []
        for i in range(0, len(ids) - max_length, stride):
            old_inputs.append(torch.tensor(ids[i : i + max_length]))
            old_targets.append(torch.tensor(ids[i + 1 : i + max_length + 1]))
        old_mb = rss_mb() - base
        del old_inputs, old_targets, ids

        base2 = rss_mb()
        _ = GPTDataset(txt, tokenizer, max_length, stride)
        new_mb = rss_mb() - base2

        print(f"Original  approach : +{old_mb:.1f} MB")
        print(f"Optimised approach : +{new_mb:.1f} MB")
        print(
            f"Memory saved       : {old_mb - new_mb:.1f} MB  "
            f"({(1 - new_mb/max(old_mb,1))*100:.0f}% reduction)"
        )
    except ImportError:
        print("Install psutil to enable memory comparison:  pip install psutil")


# ── 4. Quick demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")

    # Synthetic corpus — replace with open(file).read() for real use
    corpus = "The quick brown fox jumps over the lazy dog. " * 2_000

    print("── DataLoader demo ──────────────────────────────")
    loader = create_dataloader(
        corpus,
        enc,
        max_length=64,
        stride=32,
        batch_size=8,
        num_workers=0,
    )
    x, y = next(iter(loader))
    print(f"input  shape : {x.shape}")  # [8, 64]
    print(f"target shape : {y.shape}")  # [8, 64]
    print(f"total batches: {len(loader)}")

    print("\n── Memory comparison ────────────────────────────")
    compare_memory(corpus, enc, max_length=64, stride=32)
