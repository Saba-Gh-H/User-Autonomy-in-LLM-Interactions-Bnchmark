"""Shared reasoning-analysis utilities.

This module is the single source of truth for extracting answer choices,
extracting rationale text, tokenizing rationales, and counting lexicon terms
for both Appendix Tables A2-A7 and Table 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_RE = re.compile(r"[.!?]+")


LEXICONS: Mapping[str, Sequence[str]] = {
    "hedges": [
        "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
        "likely", "generally", "typically", "roughly", "approximately",
        "suggests", "arguably", "often", "sometimes", "usually", "tend",
        "tends", "potentially", "somewhat",
    ],
    "confidence": [
        "must", "should", "clearly", "certainly", "definitely",
        "undoubtedly", "always", "never", "ensure", "guarantee", "best",
        "optimal", "required", "necessary",
    ],
    "justification_markers": [
        "because", "since", "therefore", "thus", "so that", "hence",
        "in order to", "given that", "due to",
    ],
    "moral": [
        "ethical", "ethics", "moral", "duty", "obligation",
        "responsibility", "right", "fair", "fairness", "justice", "harm",
        "benefit", "value", "integrity", "honesty", "respect", "consent",
        "transparency", "accountability",
    ],
    "relational": [
        "relationship", "collaborate", "collaboration", "cooperate",
        "cooperation", "negotiate", "negotiation", "compromise", "conflict",
        "harmony", "trust", "empathy", "empathize", "advisor",
        "colleague", "team", "stakeholder", "patient", "doctor", "manager",
        "supervisor", "family", "friend", "partner", "discuss",
        "discussion", "dialogue", "listen",
    ],
    "epistemic": [
        "evidence", "data", "fact", "accuracy", "accurate", "reliable",
        "source", "uncertain", "uncertainty", "verify", "validate", "check",
        "expert", "authority", "research", "experiment", "analysis",
        "reason", "proof", "knowledge",
    ],
    "positive": [
        "good", "better", "best", "benefit", "helpful", "improve",
        "positive", "support", "clear", "confident", "appropriate",
        "effective", "responsible", "fair", "safe",
    ],
    "negative": [
        "bad", "worse", "worst", "risk", "harm", "problem", "negative",
        "unsafe", "unclear", "conflict", "bias", "wrong", "fail",
        "failure", "costly", "error",
    ],
}


TOKEN_NORMALIZATION_OVERRIDES = {
    "collaborated": "collaborate",
    "collaborates": "collaborate",
    "collaborating": "collaborate",
    "collaborative": "collaborate",
    "cooperated": "cooperate",
    "cooperates": "cooperate",
    "cooperating": "cooperate",
    "discussed": "discuss",
    "discusses": "discuss",
    "discussing": "discuss",
    "discussion": "discuss",
    "discussions": "discuss",
    "empathic": "empathy",
    "empathetic": "empathy",
    "empathize": "empathy",
    "empathized": "empathy",
    "empathizes": "empathy",
    "empathizing": "empathy",
    "listened": "listen",
    "listening": "listen",
    "listens": "listen",
    "negotiated": "negotiate",
    "negotiates": "negotiate",
    "negotiating": "negotiate",
    "negotiation": "negotiate",
    "negotiations": "negotiate",
    "compromised": "compromise",
    "compromises": "compromise",
    "compromising": "compromise",
    "conflicted": "conflict",
    "conflicting": "conflict",
}

NO_SINGULARIZE = {
    "analysis",
    "bias",
    "data",
    "ethics",
    "evidence",
    "news",
    "series",
}


@dataclass(frozen=True)
class ParsedAnswer:
    choice: Optional[str]
    rationale: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def count_sentences(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, len(SENT_RE.findall(text)))


def _strip_hidden_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_choice(raw_answer: object) -> Optional[str]:
    if not isinstance(raw_answer, str):
        return None

    text = _strip_hidden_reasoning(raw_answer)

    answer_match = re.search(r"\bAnswer\s*:\s*([ABC])\b", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    leading_match = re.match(
        r"^\s*([ABC])\s*(?=(?:[-:)]|\([^)]*\)\s*[-:)]?|\Z))",
        text,
        flags=re.IGNORECASE,
    )
    if leading_match:
        return leading_match.group(1).upper()

    return None


def _strip_answer_prefix(text: str) -> str:
    without_letter = re.sub(
        r"^\s*(?:Answer\s*:\s*)?[ABC]\b",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    without_option_text = re.sub(
        r"^\s*\([^)]*\)\s*",
        "",
        without_letter,
        count=1,
        flags=re.DOTALL,
    )
    without_separator = re.sub(
        r"^\s*[-:)]\s*",
        "",
        without_option_text,
        count=1,
    )
    return without_separator.strip()


def extract_rationale(raw_answer: object) -> str:
    if not isinstance(raw_answer, str):
        return ""

    text = _strip_hidden_reasoning(raw_answer)
    reason_match = re.search(
        r"\bReason\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reason_match:
        return reason_match.group(1).strip()

    if extract_choice(text) is None:
        return ""

    return _strip_answer_prefix(text)


def parse_answer(raw_answer: object) -> ParsedAnswer:
    return ParsedAnswer(
        choice=extract_choice(raw_answer),
        rationale=extract_rationale(raw_answer),
    )


def _strip_possessive(token: str) -> str:
    token = token.replace("’", "'").lower().strip("'")
    if token.endswith("'s"):
        return token[:-2]
    if token.endswith("s'"):
        return token[:-1]
    return token


def _singularize(token: str) -> str:
    if token in NO_SINGULARIZE or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("es") and token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_token(token: str) -> str:
    normalized = _strip_possessive(token)
    normalized = TOKEN_NORMALIZATION_OVERRIDES.get(normalized, normalized)
    normalized = _singularize(normalized)
    return TOKEN_NORMALIZATION_OVERRIDES.get(normalized, normalized)


NORMALIZED_LEXICONS = {
    category: {normalize_token(term) for term in terms}
    for category, terms in LEXICONS.items()
    if category != "justification_markers"
}


def lexicon_hits(tokens: Iterable[str], category: str) -> list[str]:
    terms = NORMALIZED_LEXICONS[category]
    return [token for token in tokens if normalize_token(token) in terms]


def count_lexicon_terms(tokens: Iterable[str], category: str) -> int:
    return len(lexicon_hits(tokens, category))


def contains_justification_marker(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(marker in lower for marker in LEXICONS["justification_markers"])


def rate_per_100(count: int, total_words: int) -> float:
    return 100.0 * count / total_words if total_words > 0 else 0.0


def rate_per_1000(count: int, total_words: int) -> float:
    return 1000.0 * count / total_words if total_words > 0 else 0.0


def simple_sentiment(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    positive = NORMALIZED_LEXICONS["positive"]
    negative = NORMALIZED_LEXICONS["negative"]
    pos_count = sum(1 for token in tokens if normalize_token(token) in positive)
    neg_count = sum(1 for token in tokens if normalize_token(token) in negative)
    return 0.0 if (pos_count + neg_count) == 0 else (pos_count - neg_count) / (pos_count + neg_count)


def analyze_reasonings(reasonings: Sequence[str]) -> dict[str, float]:
    total_words = 0
    total_chars = 0
    total_sents = 0
    total_tokens = 0
    total_unique = 0

    hedge_hits = 0
    conf_hits = 0
    just_docs = 0
    moral_hits = 0
    relational_hits = 0
    epistemic_hits = 0
    sentiments = []

    for rationale in reasonings:
        text = rationale if isinstance(rationale, str) else ""
        toks = tokenize(text)
        total_words += len(toks)
        total_chars += len(text)
        total_sents += count_sentences(text)
        total_tokens += len(toks)
        total_unique += len(set(normalize_token(tok) for tok in toks))

        hedge_hits += count_lexicon_terms(toks, "hedges")
        conf_hits += count_lexicon_terms(toks, "confidence")
        moral_hits += count_lexicon_terms(toks, "moral")
        relational_hits += count_lexicon_terms(toks, "relational")
        epistemic_hits += count_lexicon_terms(toks, "epistemic")

        if contains_justification_marker(text):
            just_docs += 1
        sentiments.append(simple_sentiment(toks))

    n = len(reasonings)
    return {
        "n_reasonings": n,
        "avg_words": total_words / n if n else 0.0,
        "avg_chars": total_chars / n if n else 0.0,
        "avg_sentences": total_sents / n if n else 0.0,
        "lexical_diversity": total_unique / max(1, total_tokens) if total_tokens > 0 else 0.0,
        "hedge_rate_per100w": rate_per_100(hedge_hits, total_words),
        "confidence_rate_per100w": rate_per_100(conf_hits, total_words),
        "pct_with_justification_marker": 100.0 * just_docs / n if n else 0.0,
        "moral_terms_per1000w": rate_per_1000(moral_hits, total_words),
        "relational_terms_per1000w": rate_per_1000(relational_hits, total_words),
        "epistemic_terms_per1000w": rate_per_1000(epistemic_hits, total_words),
        "avg_sentiment_lexicon": sum(sentiments) / n if n else 0.0,
    }


def model_name_from_path(path: str | Path) -> str:
    return Path(path).name.replace("_test.csv", "")


def load_model_rationales(csv_files: Sequence[str | Path], data_dir: str | Path = ".") -> pd.DataFrame:
    rows = []
    base = Path(data_dir)

    for file_name in csv_files:
        path = base / file_name
        df = pd.read_csv(path)
        if "answer" not in df.columns:
            raise ValueError(f"{path} must have an 'answer' column.")

        model = model_name_from_path(path)
        for _, rec in df.iterrows():
            parsed = parse_answer(rec["answer"])
            if parsed.choice is None:
                continue

            row = {
                "model": model,
                "option": parsed.choice,
                "reasoning": parsed.rationale,
            }
            if "scenario_code" in df.columns:
                row["scenario_code"] = rec.get("scenario_code")
            rows.append(row)

    return pd.DataFrame(rows)


def build_doc_features(rationale: str) -> dict[str, int]:
    toks = tokenize(rationale)
    return {
        "words": len(toks),
        "hedge": count_lexicon_terms(toks, "hedges"),
        "confidence": count_lexicon_terms(toks, "confidence"),
        "epistemic": count_lexicon_terms(toks, "epistemic"),
        "relational": count_lexicon_terms(toks, "relational"),
    }


def make_doc_feature_table(extracted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in extracted.iterrows():
        rec = {
            "model": row["model"],
            "choice": row["option"],
            "reason": row["reasoning"],
            **build_doc_features(row["reasoning"]),
        }
        if "scenario_code" in extracted.columns:
            rec["scenario_code"] = row["scenario_code"]
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_by_model_option(extracted: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, option), group in extracted.groupby(["model", "option"]):
        metrics = analyze_reasonings(group["reasoning"].fillna("").astype(str).tolist())
        records.append({"model": model, "option": option, **metrics})
    return pd.DataFrame(records).sort_values(["option", "model"]).reset_index(drop=True)


def metric_from_features(df_feat: pd.DataFrame, metric: str) -> float:
    total_words = df_feat["words"].sum()
    n_docs = len(df_feat)

    if n_docs == 0:
        return np.nan
    if metric == "avg_words":
        return float(df_feat["words"].mean())
    if total_words == 0:
        return np.nan
    if metric == "confidence_rate_per100w":
        return 100.0 * df_feat["confidence"].sum() / total_words
    if metric == "hedge_rate_per100w":
        return 100.0 * df_feat["hedge"].sum() / total_words
    if metric == "epistemic_terms_per1000w":
        return 1000.0 * df_feat["epistemic"].sum() / total_words
    if metric == "relational_terms_per1000w":
        return 1000.0 * df_feat["relational"].sum() / total_words
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_metric(
    df_feat: pd.DataFrame,
    metric: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(df_feat)
    if n == 0:
        return np.nan, np.nan, np.nan

    words = df_feat["words"].to_numpy(dtype=float)
    conf = df_feat["confidence"].to_numpy(dtype=float)
    hedge = df_feat["hedge"].to_numpy(dtype=float)
    epi = df_feat["epistemic"].to_numpy(dtype=float)
    rel = df_feat["relational"].to_numpy(dtype=float)

    idx = rng.integers(0, n, size=(n_boot, n))
    sampled_words = words[idx]

    if metric == "avg_words":
        vals = sampled_words.mean(axis=1)
        mean = words.mean()
    elif metric == "confidence_rate_per100w":
        sw = sampled_words.sum(axis=1)
        vals = 100.0 * conf[idx].sum(axis=1) / sw
        mean = 100.0 * conf.sum() / words.sum()
    elif metric == "hedge_rate_per100w":
        sw = sampled_words.sum(axis=1)
        vals = 100.0 * hedge[idx].sum(axis=1) / sw
        mean = 100.0 * hedge.sum() / words.sum()
    elif metric == "epistemic_terms_per1000w":
        sw = sampled_words.sum(axis=1)
        vals = 1000.0 * epi[idx].sum(axis=1) / sw
        mean = 1000.0 * epi.sum() / words.sum()
    elif metric == "relational_terms_per1000w":
        sw = sampled_words.sum(axis=1)
        vals = 1000.0 * rel[idx].sum(axis=1) / sw
        mean = 1000.0 * rel.sum() / words.sum()
    else:
        raise ValueError(metric)

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(mean), float(lo), float(hi)


def bootstrap_diff_p(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)

    def draw_vals(df_feat: pd.DataFrame) -> np.ndarray:
        n = len(df_feat)
        words = df_feat["words"].to_numpy(dtype=float)
        conf = df_feat["confidence"].to_numpy(dtype=float)
        hedge = df_feat["hedge"].to_numpy(dtype=float)
        epi = df_feat["epistemic"].to_numpy(dtype=float)
        rel = df_feat["relational"].to_numpy(dtype=float)

        idx = rng.integers(0, n, size=(n_boot, n))
        sw = words[idx].sum(axis=1)

        if metric == "avg_words":
            return words[idx].mean(axis=1)
        if metric == "confidence_rate_per100w":
            return 100.0 * conf[idx].sum(axis=1) / sw
        if metric == "hedge_rate_per100w":
            return 100.0 * hedge[idx].sum(axis=1) / sw
        if metric == "epistemic_terms_per1000w":
            return 1000.0 * epi[idx].sum(axis=1) / sw
        if metric == "relational_terms_per1000w":
            return 1000.0 * rel[idx].sum(axis=1) / sw
        raise ValueError(metric)

    diff = draw_vals(df1) - draw_vals(df2)
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return float(min(1.0, p))


def bh_qvalues(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        prev = val
        qvals[order[i]] = val
    return qvals
