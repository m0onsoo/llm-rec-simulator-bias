#!/usr/bin/env python3
"""Verify shared immutable artifacts before an API-backed experiment."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/rankings/manifest.json"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    repository = manifest_path.parents[2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    targets = {
        repository / "artifacts" / "rankings" / manifest["file"]: manifest["sha256"]
    }
    targets.update(
        {
            repository / relative: expected
            for relative, expected in manifest["dataset_hashes"].items()
        }
    )

    failed = False
    for path, expected in targets.items():
        if not path.is_file():
            print(f"MISSING  {path.relative_to(repository)}")
            failed = True
            continue
        actual = sha256(path)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} {path.relative_to(repository)}  {actual}")
        failed |= actual != expected

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

