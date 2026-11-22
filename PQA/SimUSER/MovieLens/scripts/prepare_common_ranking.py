#!/usr/bin/env python3
"""Convert Agent4Rec cached rankings to SimUSER JSON without reordering IDs."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            if values:
                rows[values[0]] = values[1:]
    return rows


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings-npy", type=Path, required=True)
    parser.add_argument("--cf-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--items-per-user", type=int, default=20)
    args = parser.parse_args()

    rankings_path = args.rankings_npy.resolve()
    matrix = np.load(rankings_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError("ranking array must be 2-D, got %r" % (matrix.shape,))
    if matrix.shape[0] < args.num_users or matrix.shape[1] < args.items_per_user:
        raise ValueError(
            "ranking shape %r is too small for %d users x %d items"
            % (matrix.shape, args.num_users, args.items_per_user)
        )

    split_paths = {
        name: (args.cf_data_dir / (name + ".txt")).resolve()
        for name in ("train", "valid", "test")
    }
    splits = {name: load_split(path) for name, path in split_paths.items()}
    converted = {}
    validation = {}
    for uid in range(args.num_users):
        ordered = [int(value) for value in matrix[uid, : args.items_per_user]]
        if len(set(ordered)) != len(ordered):
            raise ValueError("user %d has duplicate item IDs in ranking prefix" % uid)
        masked = set(splits["train"].get(uid, [])) | set(splits["test"].get(uid, []))
        overlap = [item for item in ordered if item in masked]
        if overlap:
            raise ValueError(
                "user %d ranking contains train/test-masked items: %r" % (uid, overlap)
            )
        converted[str(uid)] = ordered
        validation[str(uid)] = {
            "valid_hits": sum(item in set(splits["valid"].get(uid, [])) for item in ordered),
            "train_test_overlap": 0,
        }

    atomic_json(args.output.resolve(), converted)
    reloaded = json.loads(args.output.resolve().read_text(encoding="utf-8"))
    for uid in range(args.num_users):
        expected = [int(value) for value in matrix[uid, : args.items_per_user]]
        if reloaded[str(uid)] != expected:
            raise RuntimeError("round-trip order mismatch for user %d" % uid)

    manifest = {
        "schema_version": 1,
        "setting": "agent4rec_common_ranking",
        "source_npy": str(rankings_path),
        "source_npy_sha256": sha256(rankings_path),
        "source_shape": list(matrix.shape),
        "source_dtype": str(matrix.dtype),
        "num_users": args.num_users,
        "items_per_user": args.items_per_user,
        "user_ids": [0, args.num_users - 1],
        "split_protocol": "mask_train_plus_test_keep_valid",
        "split_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in split_paths.items()
        },
        "converted_json": str(args.output.resolve()),
        "converted_json_sha256": sha256(args.output.resolve()),
        "validation": {
            "round_trip_order_equal": True,
            "train_test_overlap_users": 0,
            "valid_hits_total": sum(row["valid_hits"] for row in validation.values()),
            "users_with_valid_hit": sum(row["valid_hits"] > 0 for row in validation.values()),
        },
    }
    atomic_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
