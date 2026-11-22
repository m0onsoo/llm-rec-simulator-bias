#!/usr/bin/env python3
"""Crash-safe vanilla SimUSER runner for an Agent4Rec cached ranking.

This runner intentionally contains no PQA inputs, page-quality labels, overlap
anchors, or PQA prompt rules. It consumes an already-masked ordered ranking and
never trains or invokes a recommender model.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ITEMS_PER_PAGE = 4
K1_INIT = 5
K2_INIT = 3
DELTA_K = 2
MAX_ROUNDS = 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_page_permutation(size, seed, user_id, page):
    """Match Agent4Rec's page permutation without changing global RNG state."""
    payload = "position-order-v1:%d:%d:%d" % (seed, user_id, page)
    local_seed = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")
    indices = list(range(size))
    random.Random(local_seed).shuffle(indices)
    return indices


def reorder_pages(pages, seed, user_id):
    reordered, metadata = [], []
    for page_number, items in enumerate(pages, start=1):
        permutation = deterministic_page_permutation(len(items), seed, user_id, page_number)
        reordered.append([items[index] for index in permutation])
        metadata.append({
            "original_exposed_item_ids": list(items),
            "position_permutation": permutation,
            "position_order": "random",
            "permutation_seed": seed,
        })
    return reordered, metadata


def directory_sha256(path, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        file_path = Path(path) / name
        digest.update(name.encode("utf-8"))
        digest.update(sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
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


def load_env(path):
    path = Path(path)
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            if values:
                rows[values[0]] = values[1:]
    return rows


class APIClient:
    """Small OpenAI-compatible client with structured retry logging."""

    def __init__(self, api_key, base_url, errors_path, retries=8, timeout=120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.errors_path = Path(errors_path)
        self.retries = retries
        self.timeout = timeout
        self.local = threading.local()
        self.log_lock = threading.Lock()
        self.responses = self.Responses(self)
        self.embeddings = self.Embeddings(self)

    class Responses:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model, input, **kwargs):
            if not isinstance(input, str):
                raise TypeError("vanilla text runner accepts string inputs only")
            body = {
                "model": model,
                "messages": [{"role": "user", "content": input}],
                "temperature": kwargs.pop("temperature", 0.0),
            }
            max_tokens = kwargs.pop("max_output_tokens", None)
            if max_tokens is not None:
                body["max_completion_tokens"] = max_tokens
            reasoning_effort = kwargs.pop("reasoning_effort", "none")
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            result, attempt = self.outer._post("/chat/completions", body)
            try:
                output = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("unexpected chat completion response") from exc
            self.outer._record_success("chat", attempt, result.get("usage", {}), output)
            return SimpleNamespace(output_text=output)

    class Embeddings:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model, input):
            result, attempt = self.outer._post(
                "/embeddings", {"model": model, "input": input}
            )
            self.outer._record_success("embedding", attempt, result.get("usage", {}), None)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=row["embedding"]) for row in result["data"]]
            )

    @contextmanager
    def context(self, user_id, stage):
        previous = getattr(self.local, "context", None)
        self.local.context = {"user_id": user_id, "stage": stage}
        try:
            yield
        finally:
            self.local.context = previous

    def begin_user(self, user_id):
        self.local.calls = []
        self.local.context = {"user_id": user_id, "stage": "initialization"}

    def calls(self):
        return list(getattr(self.local, "calls", []))

    def _record_success(self, kind, attempt, usage, output):
        context = getattr(self.local, "context", {}) or {}
        record = {
            "kind": kind,
            "stage": context.get("stage"),
            "attempt": attempt,
            "retry_count": attempt - 1,
            "usage": usage,
        }
        if output is not None:
            record["raw_response"] = output
        getattr(self.local, "calls", []).append(record)

    def _log_error(self, exc, status, attempt):
        context = getattr(self.local, "context", {}) or {}
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": context.get("user_id"),
            "stage": context.get("stage"),
            "attempt": attempt,
            "http_status": status,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock:
            with self.errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    def _post(self, endpoint, body):
        payload = json.dumps(body).encode("utf-8")
        for attempt in range(1, self.retries + 1):
            request = Request(
                self.base_url + endpoint,
                data=payload,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response), attempt
            except HTTPError as exc:
                status = exc.code
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                wrapped = RuntimeError("HTTP %s: %s" % (status, detail))
                self._log_error(wrapped, status, attempt)
                last_error = wrapped
                if status not in (408, 409, 429) and status < 500:
                    raise wrapped
            except (URLError, TimeoutError, OSError) as exc:
                self._log_error(exc, None, attempt)
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        raise last_error


class VanillaBrain:
    def __init__(self, client, model, profile, episodic_memory, kg_memory, movie_info,
                 activity_mode="included"):
        self.client = client
        self.model = model
        self.uid = int(profile["uid"])
        self.profile = profile
        self.episodic_memory = episodic_memory
        self.kg_memory = kg_memory
        self.movie_info = movie_info
        self.activity_mode = activity_mode
        self.interactions = []

    def _profile_text(self):
        habits = self.profile["habits"]
        lines = [
            "Persona: %s" % self.profile["persona"],
            "Pickiness: %s" % self.profile["pickiness"]["description"],
            "Unique tastes: %s" % self.profile["unique_tastes"],
        ]
        if self.activity_mode == "included":
            lines.append("Activity: %s" % habits["activity"])
        elif self.activity_mode != "removed":
            raise ValueError("unknown activity mode: %s" % self.activity_mode)
        lines.extend([
            "Diversity: %s" % habits["diversity"],
            "Conformity: %s" % habits["conformity"],
        ])
        return "\n".join(lines)

    def _movie(self, mid):
        info = self.movie_info.get(mid, {})
        text = "%s | Genres: %s" % (info.get("title", str(mid)), info.get("genres", ""))
        if info.get("rating") is not None:
            text += " | Avg Rating: %.2f" % float(info["rating"])
        if info.get("summary"):
            text += " | Summary: %s" % info["summary"]
        return text

    def _call(self, stage, prompt):
        with self.client.context(self.uid, stage):
            return self.client.responses.create(
                model=self.model, input=prompt, reasoning_effort="none", temperature=0.0
            ).output_text.strip()

    def _history_text(self):
        if not self.interactions:
            return "(no previous pages)"
        lines = []
        for row in self.interactions:
            lines.append(
                "Page %d: exposed %s; selected %s; final action %s"
                % (row["page"], row["exposed_item_ids"], row["selected_item_ids"], row["final_action"])
            )
        return "\n".join(lines)

    def preference_elicitation(self, page, items):
        k1, k2 = K1_INIT, K2_INIT
        previous = None
        parsed = ["SKIP"] * len(items)
        follow_ups = None
        rounds = []
        for round_index in range(MAX_ROUNDS):
            query = "Page %d movies: %s" % (
                page, "; ".join(self._movie(mid) for mid in items)
            )
            with self.client.context(self.uid, "page_%d.memory_retrieval" % page):
                episodic, follow_ups = self.episodic_memory.retrieve(
                    query, self.model, top_k=k1, follow_ups=follow_ups
                )
            kg_rows = []
            for mid in items:
                genres = self.kg_memory._genres_of(mid)
                candidates = []
                for node, relations in self.kg_memory.graph.items():
                    if not node.startswith("movie_") or node == "movie_%d" % mid:
                        continue
                    shared = relations.get("has_genre", set()) & genres
                    if shared:
                        candidates.append((int(node.split("_")[1]), len(shared)))
                candidate_ids = [row[0] for row in sorted(candidates, key=lambda row: row[1], reverse=True)[:50]]
                if candidate_ids:
                    kg_rows.extend(self.kg_memory.retrieve(self.uid, mid, candidate_ids))
            unique = {}
            for row in sorted(kg_rows, key=lambda value: value["score"], reverse=True):
                unique.setdefault(row["movie_id"], row)
            kg_rows = list(unique.values())[:k2]
            episodic_text = "\n".join("- " + row["text"] for row in episodic) or "(none)"
            kg_text = "\n".join("- " + self._movie(row["movie_id"]) for row in kg_rows) or "(none)"
            items_text = "\n".join("[%d] %s" % (i + 1, self._movie(mid)) for i, mid in enumerate(items))
            if round_index == 0:
                instruction = (
                    "For every movie, decide WATCH or SKIP from the profile and evidence. "
                    "Cite supporting evidence."
                )
            else:
                instruction = (
                    "Review the previous decisions for conflicts with the persona or insufficient evidence. "
                    "Modify only when needed. Previous response:\n%s" % previous
                )
            prompt = """You are simulating a movie user browsing a recommender.

Profile:
%s

Movies on page %d:
%s

Relevant episodic memories (top-%d):
%s

Related KG evidence (top-%d):
%s

%s
Respond one line per item exactly as:
[1] [WATCH/SKIP]: <reason> | Contradiction: <yes/no>
""" % (self._profile_text(), page, items_text, k1, episodic_text, k2, kg_text, instruction)
            raw = self._call("page_%d.preference_round_%d" % (page, round_index + 1), prompt)
            parsed = []
            for index in range(1, len(items) + 1):
                match = re.search(r"\[%d\].*?\[?(WATCH|SKIP)\]?" % index, raw, re.I)
                parsed.append(match.group(1).upper() if match else "SKIP")
            rounds.append({"round": round_index + 1, "raw_response": raw, "parsed": parsed})
            if round_index > 0 and not re.search(r"Contradiction:\s*yes", raw, re.I):
                break
            previous = raw
            k1 += DELTA_K
            k2 += DELTA_K
        return parsed, rounds

    def evaluate_item(self, page, mid):
        paths = []
        liked = self.kg_memory.graph.get("user_%d" % self.uid, {}).get("liked", set())
        target_genres = self.kg_memory._genres_of(mid)
        for node in liked:
            try:
                liked_mid = int(node.split("_")[1])
            except (IndexError, ValueError):
                continue
            shared = target_genres & self.kg_memory._genres_of(liked_mid)
            if shared:
                paths.append("%s -> %s -> %s" % (self._movie(liked_mid), sorted(shared), self._movie(mid)))
        prompt = """Simulate this user after watching a movie.
Profile:
%s
Movie: %s
KG evidence paths:
%s
Rate it considering conformity and pickiness. Respond exactly:
Rating: <integer 1-5>
Feelings: <3-4 sentences>
""" % (self._profile_text(), self._movie(mid), "\n".join(paths) or "(none)")
        raw = self._call("page_%d.item_%d_evaluation" % (page, mid), prompt)
        match = re.search(r"Rating:\s*([1-5])", raw, re.I)
        rating = int(match.group(1)) if match else 3
        feelings_match = re.search(r"Feelings:\s*(.+)", raw, re.I | re.S)
        feelings = feelings_match.group(1).strip() if feelings_match else raw
        return rating, feelings, raw

    def select_action(self, page, total_pages, items, selected):
        item_lines = "\n".join(
            "[%d] %s — %s" % (i + 1, self._movie(mid), "WATCHED" if mid in selected else "SKIPPED")
            for i, mid in enumerate(items)
        )
        prompt = """Simulate a movie user deciding what to do next.
Profile:
%s
History:
%s
Current page: %d/%d
%s
Reason sequentially about satisfaction, fatigue, emotion, then action.
Actions: NEXT, EXIT, or CLICK <number> on a skipped item.
Respond exactly:
Satisfaction: <HIGH/MEDIUM/LOW>
Fatigue: <LOW/MEDIUM/HIGH>
Emotion: <emotion>
Action: [NEXT/EXIT/CLICK <number>]
Reason: <one sentence>
""" % (self._profile_text(), self._history_text(), page, total_pages, item_lines)
        raw = self._call("page_%d.action_selection" % page, prompt)
        click = re.search(r"Action:\s*\[CLICK\s+(\d+)\]", raw, re.I)
        if click:
            return "CLICK_%s" % click.group(1), raw
        match = re.search(r"Action:\s*\[(NEXT|EXIT)\]", raw, re.I)
        return (match.group(1).upper() if match else "NEXT"), raw

    def refine_action(self, page, total_pages, tentative, action_raw):
        question_prompt = """Given this profile and browsing history, generate exactly two counterfactual
questions that test whether the tentative action is appropriate. One question per line.
Profile:
%s
History:
%s
Tentative action: %s
Action reasoning:
%s
""" % (self._profile_text(), self._history_text(), tentative, action_raw)
        questions_raw = self._call("page_%d.causal_questions" % page, question_prompt)
        questions = [line.strip() for line in questions_raw.splitlines() if line.strip()][:2]
        evaluations = []
        for index, question in enumerate(questions):
            prompt = """Evaluate whether this counterfactual supports the tentative action.
Profile:
%s
History:
%s
Tentative action: %s
Question: %s
Respond exactly:
Score: <float 0.0-1.0>
Verdict: <brief explanation>
""" % (self._profile_text(), self._history_text(), tentative, question)
            raw = self._call("page_%d.causal_evaluation_%d" % (page, index + 1), prompt)
            match = re.search(r"Score:\s*(\d+(?:\.\d+)?|\.\d+)", raw)
            score = max(0.0, min(1.0, float(match.group(1)))) if match else 0.5
            evaluations.append({"question": question, "score": score, "raw_response": raw})
        consistency = sum(row["score"] for row in evaluations) / len(evaluations) if evaluations else 1.0
        final_raw = None
        final = tentative
        if consistency < 0.5:
            prompt = """Adjust or confirm the tentative action using the causal evidence.
Profile:
%s
History:
%s
Page: %d/%d
Tentative action: %s
Evidence: %s
Respond exactly: Final Action: [NEXT/EXIT/CLICK <number>]
""" % (self._profile_text(), self._history_text(), page, total_pages, tentative, evaluations)
            final_raw = self._call("page_%d.causal_refinement" % page, prompt)
            click = re.search(r"Final Action:\s*\[CLICK\s+(\d+)\]", final_raw, re.I)
            match = re.search(r"Final Action:\s*\[(NEXT|EXIT)\]", final_raw, re.I)
            if click:
                final = "CLICK_%s" % click.group(1)
            elif match:
                final = match.group(1).upper()
        return final, {
            "questions_raw_response": questions_raw,
            "evaluations": evaluations,
            "consistency": consistency,
            "refinement_raw_response": final_raw,
        }

    def click_item(self, page, mid):
        prompt = """The user clicked a movie for more detail.
Profile:
%s
Movie: %s
Respond exactly:
Decision: [WATCH/SKIP]
Reason: <brief explanation>
""" % (self._profile_text(), self._movie(mid))
        raw = self._call("page_%d.click_%d" % (page, mid), prompt)
        match = re.search(r"Decision:\s*\[(WATCH|SKIP)\]", raw, re.I)
        return (match.group(1).upper() if match else "SKIP"), raw

    def finish(self):
        prompt = """Interview this simulated user after the recommendation session.
Profile:
%s
History:
%s
Rate satisfaction from 1 to 10. Respond exactly:
RATING: <integer 1-10>
REASON: <explanation>
""" % (self._profile_text(), self._history_text())
        interview_raw = self._call("satisfaction_interview", prompt)
        match = re.search(r"RATING:\s*(\d+)", interview_raw, re.I)
        satisfaction = max(1, min(10, int(match.group(1)))) if match else 5
        reflection_prompt = """Reflect on this browsing session as the simulated user.
Profile:
%s
History:
%s
In 2-3 sentences state what you enjoyed or disliked and what you learned about your tastes.
""" % (self._profile_text(), self._history_text())
        reflection_raw = self._call("post_interaction_reflection", reflection_prompt)
        with self.client.context(self.uid, "post_interaction_reflection_embedding"):
            self.episodic_memory.add(reflection_raw)
        return satisfaction, interview_raw, reflection_raw

    def simulate(self, pages, train_degree):
        for page_index, items in enumerate(pages, start=1):
            decisions, rounds = self.preference_elicitation(page_index, items)
            selected = [mid for mid, decision in zip(items, decisions) if decision == "WATCH"]
            evaluations = []
            for mid in list(selected):
                rating, feelings, raw = self.evaluate_item(page_index, mid)
                evaluations.append({
                    "item_id": mid, "rating": rating, "feelings": feelings,
                    "raw_response": raw, "train_degree": train_degree.get(mid, 0),
                })
                self.kg_memory.add_user_movie(self.uid, mid, rating)
                with self.client.context(self.uid, "page_%d.item_%d_memory" % (page_index, mid)):
                    self.episodic_memory.add(
                        self.episodic_memory._format_history_entry(
                            self.movie_info.get(mid, {}).get("title", str(mid)), rating
                        )
                    )
            tentative, action_raw = self.select_action(page_index, len(pages), items, selected)
            final_action, causal = self.refine_action(
                page_index, len(pages), tentative, action_raw
            )
            click = None
            if final_action.startswith("CLICK_"):
                position = int(final_action.split("_")[1])
                if 1 <= position <= len(items):
                    mid = items[position - 1]
                    decision, raw = self.click_item(page_index, mid)
                    click = {"item_id": mid, "position": position, "decision": decision, "raw_response": raw}
                    if decision == "WATCH" and mid not in selected:
                        selected.append(mid)
                        rating, feelings, raw_eval = self.evaluate_item(page_index, mid)
                        evaluations.append({
                            "item_id": mid, "rating": rating, "feelings": feelings,
                            "raw_response": raw_eval, "train_degree": train_degree.get(mid, 0),
                        })
                        self.kg_memory.add_user_movie(self.uid, mid, rating)
                final_action = "NEXT"
            page_row = {
                "page": page_index,
                "exposed_item_ids": list(items),
                "exposed_items": [
                    {"item_id": mid, "train_degree": train_degree.get(mid, 0)} for mid in items
                ],
                "decisions": [
                    {"item_id": mid, "decision": decision} for mid, decision in zip(items, decisions)
                ],
                "selected_item_ids": list(selected),
                "item_evaluations": evaluations,
                "preference_rounds": rounds,
                "tentative_action": tentative,
                "action_raw_response": action_raw,
                "causal_refinement": causal,
                "click": click,
                "final_action": final_action,
            }
            self.interactions.append(page_row)
            title = lambda mid: self.movie_info.get(mid, {}).get("title", str(mid))
            ratings = {row["item_id"]: row["rating"] for row in evaluations}
            with self.client.context(self.uid, "page_%d.interaction_memory" % page_index):
                self.episodic_memory.add_rs_interaction(
                    page_number=page_index,
                    item_type="movies",
                    name_all_items=[title(mid) for mid in items],
                    watched_items=[title(mid) for mid in selected],
                    ratings_list=[ratings[mid] for mid in selected if mid in ratings],
                    dislike_items=[title(mid) for mid in items if mid not in selected],
                )
            if final_action == "EXIT":
                break
        satisfaction, interview_raw, reflection_raw = self.finish()
        return {
            "pages": self.interactions,
            "selected_item_ids": [
                mid for row in self.interactions for mid in row["selected_item_ids"]
            ],
            "satisfaction": satisfaction,
            "interview_raw_response": interview_raw,
            "reflection_raw_response": reflection_raw,
        }


def validate_inputs(args, user_ids):
    required = {
        "ranking_json": args.ranking_json,
        "ranking_manifest": args.ranking_manifest,
        "profiles": args.profiles,
        "episodic_dir": args.episodic_dir,
        "kg": args.kg,
        "movie_detail": args.movie_detail,
        "train_split": args.train_split,
        "valid_split": args.valid_split,
        "test_split": args.test_split,
    }
    missing = [
        name for name, path in required.items()
        if not (Path(path).is_dir() if name == "episodic_dir" else Path(path).is_file())
    ]
    report = {"required_files": {name: str(Path(path).resolve()) for name, path in required.items()}, "missing": missing}
    if missing:
        return report
    rankings = json.loads(Path(args.ranking_json).read_text(encoding="utf-8"))
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    train = load_split(args.train_split)
    test = load_split(args.test_split)
    failures = []
    for uid in user_ids:
        key = str(uid)
        episodic_path = args.episodic_dir / ("%d.json" % uid)
        if key not in rankings or key not in profiles or not episodic_path.is_file():
            failures.append({"user_id": uid, "reason": "missing ranking/profile/episodic"})
            continue
        items = rankings[key]
        if len(items) != args.max_pages * args.items_per_page:
            failures.append({"user_id": uid, "reason": "ranking length", "actual": len(items)})
        overlap = set(items) & (set(train.get(uid, [])) | set(test.get(uid, [])))
        if overlap:
            failures.append({"user_id": uid, "reason": "train/test overlap", "items": sorted(overlap)})
    report.update({"users": len(user_ids), "failures": failures, "ok": not failures})
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", type=Path, required=True)
    parser.add_argument("--ranking-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--episodic-dir", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--movie-detail", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--valid-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--user-ids", default=None, help="comma-separated fixed panel")
    parser.add_argument("--items-per-page", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--user-batch-size", type=int, default=4)
    parser.add_argument("--batch-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--max-api-retries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--position-order", choices=["original", "random"], default="original",
        help="order within each page; the default preserves vanilla behavior exactly",
    )
    parser.add_argument(
        "--permutation-seed", type=int, default=0,
        help="seed used only with --position-order=random; use the same value as Agent4Rec",
    )
    parser.add_argument(
        "--activity-mode", choices=["included", "removed"], default="included",
        help="whether the explicit activity trait is injected into interaction prompts",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.items_per_page != ITEMS_PER_PAGE:
        raise ValueError("vanilla common-ranking protocol requires 4 items per page")
    if args.user_ids:
        user_ids = [int(value) for value in args.user_ids.split(",") if value.strip()]
    else:
        user_ids = list(range(args.num_users))
    report = validate_inputs(args, user_ids)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.result_dir / "input_validation.json", report)
    if not report.get("ok"):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    ranking_manifest = json.loads(args.ranking_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "setting": "agent4rec_common_ranking_vanilla_simuser_no_perception",
        "pqa_enabled": False,
        "perception_enabled": False,
        "model": args.model,
        "base_url": args.base_url,
        "user_ids": user_ids,
        "items_per_page": args.items_per_page,
        "max_pages": args.max_pages,
        "seed": args.seed,
        "ranking_json_sha256": sha256(args.ranking_json),
        "ranking_source_npy_sha256": ranking_manifest["source_npy_sha256"],
        "split_protocol": "mask_train_plus_test_keep_valid",
        "split_sha256": {
            "train": sha256(args.train_split),
            "valid": sha256(args.valid_split),
            "test": sha256(args.test_split),
        },
        "profiles_sha256": sha256(args.profiles),
        "episodic_dir_sha256": directory_sha256(
            args.episodic_dir, ["%d.json" % uid for uid in user_ids]
        ),
        "kg_sha256": sha256(args.kg),
        "runtime_tunables": ["workers", "user_batch_size", "batch_cooldown_seconds"],
    }
    # Preserve byte-for-byte legacy manifest contents in the default mode.
    if args.position_order != "original":
        manifest.update({
            "position_order": args.position_order,
            "permutation_seed": args.permutation_seed,
            "position_permutation_version": 1,
        })
    if args.activity_mode == "removed":
        manifest.update({
            "setting": "agent4rec_common_ranking_vanilla_simuser_no_activity_no_perception",
            "activity_prompt_enabled": False,
            "activity_group_retained_for_analysis": True,
        })
    manifest_path = args.result_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("result directory manifest differs from requested immutable configuration")
    else:
        atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"dry_run": "passed", "manifest": manifest}, ensure_ascii=False, indent=2))
        return

    load_env(args.env_file)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available")
    client = APIClient(
        api_key, args.base_url, args.result_dir / "api_errors.jsonl",
        retries=args.max_api_retries,
    )
    base_dir = Path(__file__).resolve().parents[1] / "Phase_2_Simulate_Interactions"
    episodic_module = load_module("vanilla_episodic", base_dir / "6_Episodic_Memory.py")
    kg_module = load_module("vanilla_kg", base_dir / "7_KG_Memory.py")
    rankings = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    kg_data = json.loads(args.kg.read_text(encoding="utf-8"))
    movie_df = pd.read_csv(args.movie_detail)
    movie_info = {
        int(row["movie_id"]): {
            "title": row["title"], "genres": row["genres"],
            "rating": row.get("rating"), "summary": row.get("summary", ""),
        }
        for _, row in movie_df.iterrows()
    }
    train = load_split(args.train_split)
    train_degree = defaultdict(int)
    for items in train.values():
        for mid in set(items):
            train_degree[mid] += 1

    users_dir = args.result_dir / "users"
    completed = set()
    if args.resume:
        for uid in user_ids:
            path = users_dir / ("%d.json" % uid)
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") == "complete" and row.get("user_id") == uid:
                    completed.add(uid)
            except (OSError, ValueError, TypeError):
                pass
    remaining = [uid for uid in user_ids if uid not in completed]
    print("resume preserved %d complete users; %d remain" % (len(completed), len(remaining)))

    def run_user(uid):
        random.seed(args.seed + uid)
        client.begin_user(uid)
        em = episodic_module.EpisodicMemory(client)
        episodic_row = json.loads(
            (args.episodic_dir / ("%d.json" % uid)).read_text(encoding="utf-8")
        )
        em.from_dict(episodic_row["memories"])
        kg = kg_module.KGMemory(client)
        kg.from_dict(kg_data)
        items = [int(value) for value in rankings[str(uid)]]
        pages = [items[index:index + args.items_per_page] for index in range(0, len(items), args.items_per_page)]
        order_metadata = None
        if args.position_order == "random":
            pages, order_metadata = reorder_pages(pages, args.permutation_seed, uid)
        brain = VanillaBrain(
            client, args.model, profiles[str(uid)], em, kg, movie_info,
            activity_mode=args.activity_mode,
        )
        simulation = brain.simulate(pages[:args.max_pages], train_degree)
        if order_metadata is not None:
            for page_row in simulation["pages"]:
                page_row.update(order_metadata[page_row["page"] - 1])
        result = {
            "schema_version": 1,
            "status": "complete",
            "user_id": uid,
            "persona": profiles[str(uid)].get("persona"),
            "activity": profiles[str(uid)].get("habits", {}).get("activity"),
            "profile": profiles[str(uid)],
            "simulation": simulation,
            "api_calls": client.calls(),
        }
        if args.activity_mode == "removed":
            result.update({
                "activity_prompt_enabled": False,
                "analysis_activity_group": profiles[str(uid)].get("activity_group"),
            })
        atomic_json(users_dir / ("%d.json" % uid), result)
        return uid

    async def run_batches():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(remaining), args.user_batch_size):
                batch = remaining[start:start + args.user_batch_size]
                tasks = [loop.run_in_executor(executor, run_user, uid) for uid in batch]
                done = await asyncio.gather(*tasks, return_exceptions=True)
                for uid, outcome in zip(batch, done):
                    if isinstance(outcome, BaseException):
                        print("user %d failed: %s" % (uid, outcome), flush=True)
                    else:
                        print("user %d complete" % uid, flush=True)
                if start + args.user_batch_size < len(remaining):
                    await asyncio.sleep(args.batch_cooldown_seconds)

    asyncio.run(run_batches())
    complete = []
    corrupt = []
    for uid in user_ids:
        path = users_dir / ("%d.json" % uid)
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "complete" and row.get("user_id") == uid:
                complete.append(uid)
            else:
                corrupt.append(uid)
        except (OSError, ValueError, TypeError):
            corrupt.append(uid)
    atomic_json(args.result_dir / "run_summary.json", {
        "requested": len(user_ids), "complete": len(complete),
        "missing_or_corrupt": corrupt, "complete_user_ids": complete,
    })
    if corrupt:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
