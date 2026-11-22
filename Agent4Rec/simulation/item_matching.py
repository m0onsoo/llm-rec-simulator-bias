"""Candidate-constrained title matching for Agent4Rec responses."""

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
import unicodedata


_YEAR = re.compile(r"\s*\((?:18|19|20)\d{2}\)\s*$")
_TRAILING_ARTICLE = re.compile(r",\s*(the|an|a)\s*$", re.IGNORECASE)
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_ARTICLES = {"the", "a", "an"}


def normalize_title(title):
    text = unicodedata.normalize("NFKC", str(title)).casefold().strip()
    text = _YEAR.sub("", text)
    text = _TRAILING_ARTICLE.sub("", text)
    text = text.replace("&", " and ")
    text = _NON_WORD.sub(" ", text)
    tokens = _SPACE.sub(" ", text).strip().split()
    if tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def title_similarity(left, right):
    left_tokens, right_tokens = set(left.split()), set(right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    # Character similarity handles a minor spelling variation in one token;
    # token overlap still prevents a superficially similar unrelated title.
    return 0.3 * jaccard + 0.7 * sequence


@dataclass(frozen=True)
class TitleMatch:
    llm_title: str
    item_id: object
    canonical_title: object
    match_method: str
    match_score: float

    def to_dict(self):
        return asdict(self)


def match_title_to_candidates(llm_title, candidate_ids, candidate_titles,
                              threshold=0.80, margin=0.15):
    if len(candidate_ids) != len(candidate_titles):
        raise ValueError("candidate_ids and candidate_titles must have equal length")

    query = normalize_title(llm_title)
    normalized_candidates = [normalize_title(title) for title in candidate_titles]
    exact = [index for index, title in enumerate(normalized_candidates) if title == query]
    if query and len(exact) == 1:
        index = exact[0]
        return TitleMatch(str(llm_title), int(candidate_ids[index]),
                          str(candidate_titles[index]), "normalized_exact", 1.0)

    scores = [title_similarity(query, title) for title in normalized_candidates]
    if not scores:
        return TitleMatch(str(llm_title), None, None, "unresolved", 0.0)
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    best_index = order[0]
    best = scores[best_index]
    second = scores[order[1]] if len(order) > 1 else 0.0
    if best >= threshold and best - second >= margin:
        return TitleMatch(str(llm_title), int(candidate_ids[best_index]),
                          str(candidate_titles[best_index]), "candidate_fuzzy", best)
    return TitleMatch(str(llm_title), None, None, "unresolved", best)


def match_titles_to_candidates(llm_titles, candidate_ids, candidate_titles):
    return [
        match_title_to_candidates(title, candidate_ids, candidate_titles)
        for title in llm_titles
    ]
