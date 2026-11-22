"""Deterministic page-local ordering for controlled position experiments."""

import hashlib
import random


def deterministic_page_permutation(size, seed, user_id, page):
    """Return a reproducible random permutation without touching global RNG state."""
    payload = f"position-order-v1:{seed}:{user_id}:{page}".encode("ascii")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    indices = list(range(size))
    random.Random(local_seed).shuffle(indices)
    return indices
