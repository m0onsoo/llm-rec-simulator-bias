#!/usr/bin/env python3
"""Prepare original-style SimUSER personas and profiles with resumable stages."""

import argparse
import asyncio
import json
import os
import pickle
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from run_vanilla_common import APIClient, atomic_json, load_env


NUM_HISTORY = 50
NUM_CANDIDATES = 5
J = 2
RHO = 10
N_OTHER = 5

AGES = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
OCCUPATIONS = [
    "other", "academic/educator", "artist", "clerical/admin", "college/grad student",
    "customer service", "doctor/health care", "executive/managerial", "farmer",
    "homemaker", "K-12 student", "lawyer", "programmer", "retired",
    "sales/marketing", "scientist", "self-employed", "technician/engineer",
    "tradesman/craftsman", "unemployed", "writer",
]
BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

PICKINESS = {
    "not picky": "You are not picky and tend to give high ratings to most movies you watch.",
    "moderately picky": "You are moderately picky and rate movies by how well they match your preferences.",
    "extremely picky": "You are extremely picky and rarely give high ratings unless a movie truly impresses you.",
}
ACTIVITY = {
    1: "An Incredibly Elusive Occasional Viewer with extraordinarily infrequent movie-watching habits who exits quickly when unsatisfied.",
    2: "An Occasional Viewer who mainly explores movies aligned with their taste and exits after a few unsatisfying experiences.",
    3: "A Movie Enthusiast who watches frequently and is tolerant of some unsatisfying recommendations before exiting.",
}
CONFORMITY = {
    1: "A Dedicated Follower whose ratings closely follow historical average ratings.",
    2: "A Balanced Evaluator who considers historical ratings and personal preferences.",
    3: "A Maverick Critic who largely ignores historical ratings and follows personal taste.",
}
DIVERSITY = {
    1: "A highly selective viewer who stays tightly within established tastes.",
    2: "A niche explorer who occasionally explores but mostly stays with preferred movie types.",
    3: "A cinematic trailblazer who actively explores diverse and unfamiliar styles.",
}


def load_data(raw_dir, cf_dir, simulation_dir):
    with (raw_dir / "user_id_map.pkl").open("rb") as handle:
        user_map = pickle.load(handle)
    with (raw_dir / "movie_id_map.pkl").open("rb") as handle:
        movie_map = pickle.load(handle)
    reverse_user = {value: key for key, value in user_map.items()}
    reverse_movie = {value: key for key, value in movie_map.items()}
    ratings = {}
    with (raw_dir / "ratings.dat").open(encoding="latin-1") as handle:
        for line in handle:
            uid, mid, rating, timestamp = line.strip().split("::")
            ratings.setdefault(int(uid), {})[int(mid)] = (float(rating), int(timestamp))
    train = {}
    with (cf_dir / "train.txt").open(encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            train[values[0]] = values[1:]
    movie_df = pd.read_csv(simulation_dir / "movie_detail.csv")
    movies = {
        int(row["movie_id"]): {
            "title": row["title"], "genres": row["genres"],
            "average_rating": float(row["rating"]), "summary": row.get("summary", ""),
        }
        for _, row in movie_df.iterrows()
    }
    stats_df = pd.read_csv(simulation_dir / "user_statistic.csv")
    stats = {
        int(row["user_id"]): {
            "activity": int(row["activity"]),
            "diversity": int(row["diversity"]),
            "conformity": int(row["conformity"]),
        }
        for _, row in stats_df.iterrows()
    }
    return reverse_user, reverse_movie, ratings, train, movies, stats


def records_for(uid, data):
    reverse_user, reverse_movie, ratings, train, movies, _ = data
    original_uid = reverse_user[uid]
    rows = []
    for mid in train[uid]:
        original_mid = reverse_movie[mid]
        rating, timestamp = ratings[original_uid][original_mid]
        row = dict(movies[mid])
        row.update({"item_id": mid, "user_rating": rating, "timestamp": timestamp})
        rows.append(row)
    return rows


def sample_history(uid, data, seed):
    rows = records_for(uid, data)
    rng = random.Random(seed + uid)
    return rng.sample(rows, min(NUM_HISTORY, len(rows)))


def history_text(rows, include_summary=False):
    lines = []
    for row in rows:
        text = "- %s | genres=%s | user rating=%.1f | average rating=%.2f" % (
            row["title"], row["genres"], row["user_rating"], row["average_rating"]
        )
        if include_summary and row.get("summary"):
            text += " | summary=%s" % row["summary"]
        lines.append(text)
    return "\n".join(lines)


def summary_prompt(rows):
    liked = [row for row in rows if row["user_rating"] >= 4]
    disliked = [row for row in rows if row["user_rating"] <= 2]
    return """Summarize this user's movie preferences in 1-2 sentences.
Ratings 4-5 are liked; ratings 1-2 are disliked; rating 3 is neutral and must not be labeled liked or disliked.
Only state preferences supported by the history, and do not mention movie titles.

Liked:
%s

Disliked:
%s
""" % (history_text(liked) or "(none)", history_text(disliked) or "(none)")


def candidate_prompt(summary, rows):
    return """Infer one plausible persona for a movie user from only the preference summary and interaction history.
Do not assume access to the user's real demographic labels.

Preference summary:
%s

Interaction history:
%s

Choose a plausible age range from: %s.
Choose a plausible occupation from: %s.
Assign each Big Five trait (%s) an integer level from 1 (low) to 3 (high).
Write exactly two concise sentences integrating background, personality, and movie preferences.
Output only the persona.
""" % (
        summary, history_text(rows, include_summary=True), ", ".join(AGES),
        ", ".join(OCCUPATIONS), ", ".join(BIG_FIVE),
    )


def scoring_prompt(persona, rows):
    return """Evaluate how well this movie interaction subset aligns with the persona.
Persona:
%s
Interactions:
%s
Respond exactly:
Score: <integer 1-10>
Reason: <brief explanation>
""" % (persona, history_text(rows, include_summary=True))


def parsed_score(text):
    match = re.search(r"Score:\s*(\d+)", text, re.I)
    if match:
        value = match.group(1)
    else:
        match = re.search(r"\d+", text)
        value = match.group(0) if match else None
    return max(1, min(10, int(value))) if value is not None else 5


def pickiness(avg):
    if avg >= 4.5:
        return "not picky"
    if avg >= 3.5:
        return "moderately picky"
    return "extremely picky"


def valid_user_file(path, uid, stage):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return row.get("user_id") == uid and row.get("stage") == stage and row.get("status") == "complete"
    except (OSError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cf-data-dir", type=Path, required=True)
    parser.add_argument("--simulation-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--stage", choices=["summaries", "candidates", "selection", "profiles", "all"], required=True)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--user-ids", default=None)
    parser.add_argument("--seed", type=int, default=101)
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
        args.simulation_data_dir / "user_statistic.csv",
    ]
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    estimate = {
        "users": len(user_ids), "history_sample_size": NUM_HISTORY,
        "persona_candidates": NUM_CANDIDATES,
        "self_consistency_calls_per_user": NUM_CANDIDATES * J * (1 + N_OTHER),
        "total_text_calls_per_user": 1 + NUM_CANDIDATES + NUM_CANDIDATES * J * (1 + N_OTHER),
    }
    validation = {"ok": not missing, "missing": missing, "estimate": estimate}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "input_validation.json", validation)
    if missing:
        raise SystemExit(2)
    if args.dry_run:
        print(json.dumps(validation, indent=2))
        return
    data = load_data(args.raw_dir, args.cf_data_dir, args.simulation_data_dir)
    if args.stage == "profiles":
        stages = ["profiles"]
    elif args.stage == "all":
        stages = ["summaries", "candidates", "selection", "profiles"]
    else:
        stages = [args.stage]
    client = None
    if any(stage != "profiles" for stage in stages):
        load_env(args.env_file)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not available")
        client = APIClient(
            key, args.base_url, args.output_dir / "api_errors.jsonl",
            retries=args.max_api_retries,
        )

    def run_stage_user(stage, uid):
        stage_dir = args.output_dir / stage
        output = stage_dir / ("%d.json" % uid)
        if args.resume and valid_user_file(output, uid, stage):
            return uid
        client.begin_user(uid)
        sampled = sample_history(uid, data, args.seed)
        if stage == "summaries":
            with client.context(uid, "phase1.summary"):
                raw = client.responses.create(model=args.model, input=summary_prompt(sampled)).output_text.strip()
            row = {
                "status": "complete", "stage": stage, "user_id": uid,
                "sampled_item_ids": [item["item_id"] for item in sampled],
                "liked_item_ids": [item["item_id"] for item in sampled if item["user_rating"] >= 4],
                "disliked_item_ids": [item["item_id"] for item in sampled if item["user_rating"] <= 2],
                "summary": raw, "raw_response": raw, "api_calls": client.calls(),
            }
        elif stage == "candidates":
            summary = json.loads((args.output_dir / "summaries" / ("%d.json" % uid)).read_text())["summary"]
            candidates = []
            for index in range(NUM_CANDIDATES):
                with client.context(uid, "phase1.candidate_%d" % (index + 1)):
                    raw = client.responses.create(
                        model=args.model, input=candidate_prompt(summary, sampled)
                    ).output_text.strip()
                candidates.append(raw)
            row = {
                "status": "complete", "stage": stage, "user_id": uid,
                "candidate_count": len(candidates), "candidates": candidates,
                "api_calls": client.calls(),
            }
        elif stage == "selection":
            candidates = json.loads((args.output_dir / "candidates" / ("%d.json" % uid)).read_text())["candidates"]
            all_users = sorted(data[3])
            scores = []
            traces = []
            for candidate_index, persona in enumerate(candidates):
                rng = random.Random(args.seed + uid * 1000 + candidate_index)
                target_total = 0
                candidate_trace = []
                target_rows = records_for(uid, data)
                for repeat in range(J):
                    subset = rng.sample(target_rows, min(RHO, len(target_rows)))
                    with client.context(uid, "phase1.selection_candidate_%d_target_%d" % (candidate_index + 1, repeat + 1)):
                        raw = client.responses.create(model=args.model, input=scoring_prompt(persona, subset)).output_text.strip()
                    score = parsed_score(raw)
                    target_total += score
                    candidate_trace.append({"kind": "target", "score": score, "raw_response": raw})
                others = rng.sample([other for other in all_users if other != uid], N_OTHER)
                other_sums = []
                for other_uid in others:
                    other_rows = records_for(other_uid, data)
                    subtotal = 0
                    for repeat in range(J):
                        subset = rng.sample(other_rows, min(RHO, len(other_rows)))
                        with client.context(uid, "phase1.selection_candidate_%d_other_%d_%d" % (candidate_index + 1, other_uid, repeat + 1)):
                            raw = client.responses.create(model=args.model, input=scoring_prompt(persona, subset)).output_text.strip()
                        score = parsed_score(raw)
                        subtotal += score
                        candidate_trace.append({
                            "kind": "other", "other_user_id": other_uid,
                            "score": score, "raw_response": raw,
                        })
                    other_sums.append(subtotal)
                score = target_total - sum(other_sums) / len(other_sums)
                scores.append(score)
                traces.append(candidate_trace)
            best = max(range(len(scores)), key=lambda index: scores[index])
            row = {
                "status": "complete", "stage": stage, "user_id": uid,
                "scores": scores, "best_index": best,
                "selected_persona": candidates[best], "evaluation_traces": traces,
                "api_calls": client.calls(),
            }
        else:
            raise ValueError(stage)
        atomic_json(output, row)
        return uid

    async def run_api_stage(stage):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(user_ids), args.user_batch_size):
                batch = user_ids[start:start + args.user_batch_size]
                results = await asyncio.gather(
                    *[loop.run_in_executor(executor, run_stage_user, stage, uid) for uid in batch],
                    return_exceptions=True,
                )
                failures = [(uid, value) for uid, value in zip(batch, results) if isinstance(value, BaseException)]
                if failures:
                    for uid, error in failures:
                        print("%s user %d failed: %s" % (stage, uid, error), flush=True)
                    raise RuntimeError("stage %s batch failed" % stage)
                print("%s completed %d/%d" % (stage, min(start + len(batch), len(user_ids)), len(user_ids)), flush=True)
                if start + len(batch) < len(user_ids):
                    await asyncio.sleep(args.batch_cooldown_seconds)

    for stage in stages:
        if stage != "profiles":
            asyncio.run(run_api_stage(stage))
            continue
        profiles = {}
        for uid in user_ids:
            selected = json.loads((args.output_dir / "selection" / ("%d.json" % uid)).read_text())
            summary = json.loads((args.output_dir / "summaries" / ("%d.json" % uid)).read_text())
            rows = records_for(uid, data)
            average = sum(row["user_rating"] for row in rows) / len(rows)
            level = pickiness(average)
            stat = data[5][uid]
            profiles[str(uid)] = {
                "uid": uid, "persona": selected["selected_persona"],
                "pickiness": {"level": level, "description": PICKINESS[level], "avg_rating": round(average, 4)},
                "habits": {
                    "activity": ACTIVITY[stat["activity"]],
                    "diversity": DIVERSITY[stat["diversity"]],
                    "conformity": CONFORMITY[stat["conformity"]],
                },
                "activity_group": stat["activity"],
                "diversity_group": stat["diversity"],
                "conformity_group": stat["conformity"],
                "unique_tastes": summary["summary"],
            }
        atomic_json(args.output_dir / "user_profiles.json", profiles)
        print("profiles assembled for %d users" % len(profiles))


if __name__ == "__main__":
    main()
