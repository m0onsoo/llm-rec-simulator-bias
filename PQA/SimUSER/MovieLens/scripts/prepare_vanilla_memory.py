#!/usr/bin/env python3
"""Prepare SimUSER episodic and KG memories without a recommender/GPU."""

import argparse
import asyncio
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from run_vanilla_common import APIClient, atomic_json, load_env, load_module


def load_source(raw_dir, cf_dir, simulation_dir):
    with (raw_dir / "user_id_map.pkl").open("rb") as handle:
        user_map = pickle.load(handle)
    with (raw_dir / "movie_id_map.pkl").open("rb") as handle:
        movie_map = pickle.load(handle)
    reverse_user = {value: key for key, value in user_map.items()}
    reverse_movie = {value: key for key, value in movie_map.items()}
    train = {}
    with (cf_dir / "train.txt").open(encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            train[values[0]] = values[1:]
    ratings = {}
    with (raw_dir / "ratings.dat").open(encoding="latin-1") as handle:
        for line in handle:
            uid, mid, rating, _ = line.strip().split("::")
            ratings.setdefault(int(uid), {})[int(mid)] = float(rating)
    movie_df = pd.read_csv(simulation_dir / "movie_detail.csv")
    movies = {
        int(row["movie_id"]): {"title": row["title"], "genres": row["genres"]}
        for _, row in movie_df.iterrows()
    }
    return reverse_user, reverse_movie, train, ratings, movies


def history_for(uid, source):
    reverse_user, reverse_movie, train, ratings, movies = source
    original_uid = reverse_user[uid]
    rows = []
    for mid in train[uid]:
        original_mid = reverse_movie[mid]
        rows.append((movies[mid]["title"], int(ratings[original_uid][original_mid])))
    return rows


def valid_memory(path, uid):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return row.get("status") == "complete" and row.get("user_id") == uid and isinstance(row.get("memories"), list)
    except (OSError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cf-data-dir", type=Path, required=True)
    parser.add_argument("--simulation-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["episodic", "kg", "all"], required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--user-ids", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--user-batch-size", type=int, default=4)
    parser.add_argument("--batch-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--max-api-retries", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    user_ids = (
        [int(value) for value in args.user_ids.split(",") if value.strip()]
        if args.user_ids else list(range(args.num_users))
    )
    required = [
        args.raw_dir / "user_id_map.pkl", args.raw_dir / "movie_id_map.pkl",
        args.raw_dir / "ratings.dat", args.cf_data_dir / "train.txt",
        args.simulation_data_dir / "movie_detail.csv",
    ]
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    validation = {
        "ok": not missing, "missing": missing, "users": len(user_ids),
        "episodic_embedding_batches": len(user_ids),
        "kg_movie_count_expected": 3292,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "input_validation.json", validation)
    if missing:
        raise SystemExit(2)
    if args.dry_run:
        print(json.dumps(validation, indent=2))
        return
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not available")
    client = APIClient(
        key, args.base_url, args.output_dir / "api_errors.jsonl",
        retries=args.max_api_retries,
    )
    source = load_source(args.raw_dir, args.cf_data_dir, args.simulation_data_dir)
    module_dir = Path(__file__).resolve().parents[1] / "Phase_2_Simulate_Interactions"
    episodic_module = load_module("prepare_episodic", module_dir / "6_Episodic_Memory.py")
    kg_module = load_module("prepare_kg", module_dir / "7_KG_Memory.py")
    stages = ["episodic", "kg"] if args.stage == "all" else [args.stage]

    def prepare_user(uid):
        output = args.output_dir / "episodic" / ("%d.json" % uid)
        if args.resume and valid_memory(output, uid):
            return uid
        client.begin_user(uid)
        memory = episodic_module.EpisodicMemory(client)
        with client.context(uid, "episodic_initialization"):
            memory.initialize_from_history(history_for(uid, source))
        atomic_json(output, {
            "status": "complete", "user_id": uid,
            "memories": memory.to_dict(), "api_calls": client.calls(),
        })
        return uid

    async def prepare_episodic():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(user_ids), args.user_batch_size):
                batch = user_ids[start:start + args.user_batch_size]
                results = await asyncio.gather(
                    *[loop.run_in_executor(executor, prepare_user, uid) for uid in batch],
                    return_exceptions=True,
                )
                failures = [(uid, result) for uid, result in zip(batch, results) if isinstance(result, BaseException)]
                if failures:
                    for uid, error in failures:
                        print("episodic user %d failed: %s" % (uid, error), flush=True)
                    raise RuntimeError("episodic batch failed")
                print("episodic completed %d/%d" % (min(start + len(batch), len(user_ids)), len(user_ids)), flush=True)
                if start + len(batch) < len(user_ids):
                    await asyncio.sleep(args.batch_cooldown_seconds)

    for stage in stages:
        if stage == "episodic":
            asyncio.run(prepare_episodic())
        else:
            output = args.output_dir / "kg_memory.json"
            if args.resume and output.is_file():
                print("KG cache already exists; preserving it")
                continue
            client.begin_user(None)
            kg = kg_module.KGMemory(client)
            reverse_user, reverse_movie, train, ratings, movies = source
            kg.initialize(train, ratings, reverse_user, reverse_movie, movies)
            with client.context(None, "kg_movie_embeddings"):
                kg.precompute_movie_embeddings(movies)
            atomic_json(output, kg.to_dict())
            print("KG memory saved for %d movies" % len(kg.movie_embeddings))


if __name__ == "__main__":
    main()
