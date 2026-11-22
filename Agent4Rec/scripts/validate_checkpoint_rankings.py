"""Validate shipped Agent4Rec rankings against the shipped checkpoints.

The repository contains two ranking protocols:

* ``recommend_top_k.txt`` masks train + validation interactions and is the
  test-evaluation ranking written by ``AbstractRS.recommend_top_k``.
* the simulator masks train + test interactions before showing validation
  candidates to an avatar.

This script regenerates both top-20 lists from a checkpoint.  It first checks
that the checkpoint reproduces the shipped text file under the same mask, then
reports how much that list differs from the ranking actually used by the
simulation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RECOMMENDERS = ROOT / "recommenders"
sys.path.insert(0, str(RECOMMENDERS))

# Loading an inference model imports the training evaluator and its optional
# Cython backend through AbstractRS, even though neither is used here.  Avoid
# tying checkpoint validation to a compiler/runtime ABI by supplying the tiny
# interface required at import time.  Ranking itself remains the original
# PyTorch model code.
evaluator_stub = types.ModuleType("evaluator")
evaluator_stub.ProxyEvaluator = object
sys.modules.setdefault("evaluator", evaluator_stub)
cython_tools_stub = types.ModuleType("util.cython.tools")
cython_tools_stub.float_type = np.float32
cython_tools_stub.is_ndarray = lambda value, dtype=None: isinstance(value, np.ndarray)
sys.modules.setdefault("util.cython.tools", cython_tools_stub)

from data import Data  # noqa: E402
from models.base.utils import merge_user_list  # noqa: E402
from util import DataIterator, argmax_top_k  # noqa: E402


def parse_ranking_file(path: Path) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.split()
            if not fields:
                continue
            try:
                values = [int(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: expected integer fields") from exc
            rankings[values[0]] = values[1:]
    return rankings


def newest_checkpoint(model_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("epoch=*.checkpoint.pth.tar"):
        match = re.search(r"epoch=(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    return max(candidates)[1]


def load_saved_args(model_dir: Path, cuda_index: int) -> Namespace:
    with (model_dir / "args.txt").open(encoding="utf-8") as handle:
        values = json.load(handle)
    values.update({
        "data_path": str(ROOT / "datasets") + os.sep,
        "dataset": "ml-1m",
        "cuda": cuda_index,
    })
    return Namespace(**values)


def load_data_and_model(model_name: str, model_dir: Path, cuda_index: int):
    saved_args = load_saved_args(model_dir, cuda_index)
    module = importlib.import_module(f"models.{model_name}")
    data_class = getattr(module, f"{model_name}_Data", Data)
    data = data_class(saved_args)
    model_class = getattr(module, model_name)
    model = model_class(saved_args, data).cuda(torch.device(f"cuda:{cuda_index}"))

    checkpoint_path = newest_checkpoint(model_dir)
    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{cuda_index}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return saved_args, data, model, checkpoint_path, checkpoint.get("epoch")


def top_k_for_users(model, data, users: list[int], batch_size: int, k: int):
    test_dump = merge_user_list([data.train_user_list, data.valid_user_list])
    simulation_dump = merge_user_list([data.train_user_list, data.test_user_list])
    test_rankings: dict[int, list[int]] = {}
    simulation_rankings: dict[int, list[int]] = {}

    iterator = DataIterator(users, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch_users in iterator:
            scores = model.predict(batch_users, None)
            if not isinstance(scores, np.ndarray):
                scores = np.asarray(scores)
            for row, user in enumerate(batch_users):
                test_scores = scores[row].copy()
                # This is the exact non-candidate mask in recommend_top_k():
                # test items are deliberately retained even if duplicated in valid.
                masked_for_test = [
                    item for item in test_dump[user]
                    if item not in data.test_user_list[user]
                ]
                test_scores[masked_for_test] = -np.inf
                test_rankings[int(user)] = [
                    int(item) for item in argmax_top_k(test_scores, k)
                ]

                simulation_scores = scores[row].copy()
                simulation_scores[simulation_dump[user]] = -np.inf
                # The simulator uses np.argsort over the full score vector.
                simulation_rankings[int(user)] = [
                    int(item) for item in np.argsort(-simulation_scores)[:k]
                ]
    return test_rankings, simulation_rankings


def comparison(generated: dict[int, list[int]], reference: dict[int, list[int]], users: list[int]):
    exact = 0
    overlaps = []
    position_matches = []
    examples = []
    for user in users:
        actual = generated[user]
        expected = reference[user][: len(actual)]
        if actual == expected:
            exact += 1
        overlap = len(set(actual) & set(expected))
        overlaps.append(overlap / len(actual))
        position_matches.append(sum(a == b for a, b in zip(actual, expected)) / len(actual))
        if actual != expected and len(examples) < 5:
            examples.append({
                "user_id": user,
                "generated": actual,
                "reference": expected,
                "overlap": overlap,
            })
    return {
        "users": len(users),
        "exact_users": exact,
        "exact_rate": exact / len(users),
        "mean_set_overlap": float(np.mean(overlaps)),
        "mean_position_match": float(np.mean(position_matches)),
        "mismatch_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["MF", "LightGCN", "MultVAE"])
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproduction_outputs" / "checkpoint_ranking_validation.json",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    report = {
        "seed": args.seed,
        "top_k": args.top_k,
        "cuda": torch.cuda.get_device_name(args.cuda_index),
        "models": {},
    }
    for model_name in args.models:
        print(f"\n===== {model_name} =====", flush=True)
        model_dir = RECOMMENDERS / "weights" / "ml-1m" / model_name / "Saved"
        saved_args, data, model, checkpoint_path, epoch = load_data_and_model(
            model_name, model_dir, args.cuda_index
        )
        reference = parse_ranking_file(model_dir / "recommend_top_k.txt")
        users = sorted(set(reference) & set(data.test_user_list))[: args.num_users]
        if not users:
            raise RuntimeError(f"No comparable users for {model_name}")
        test_rankings, simulation_rankings = top_k_for_users(
            model, data, users, args.batch_size, args.top_k
        )
        report["models"][model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_epoch": epoch,
            "saved_args": vars(saved_args),
            "checkpoint_vs_shipped_test_topk": comparison(test_rankings, reference, users),
            "simulation_topk_vs_shipped_test_topk": comparison(
                simulation_rankings, reference, users
            ),
            "generated_simulation_topk": {
                str(user): ranking for user, ranking in simulation_rankings.items()
            },
        }
        print(json.dumps(report["models"][model_name], indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
