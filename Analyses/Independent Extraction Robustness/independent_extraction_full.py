"""Full independent answer-extraction evaluation.

Uses the same evaluator prompt and settings validated in the 20-row pilot.
Seeds the final cache from pilot_independent_extraction_results.csv and then
evaluates only missing rows.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Analyses" / "Independent Extraction Robustness"

PROMPT_VERSION = "independent_extraction_pilot_v1"
RUN_NAME = "full_independent_extraction_v1"
EVALUATOR_MODEL = "openai/gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 180
MAX_ATTEMPTS = 3

PILOT_RESULTS_CSV = OUT_DIR / "pilot_independent_extraction_results.csv"
RESULTS_CSV = OUT_DIR / "independent_extraction_results.csv"
SUMMARY_CSV = OUT_DIR / "independent_extraction_summary.csv"
DISAGREEMENTS_CSV = OUT_DIR / "disagreements.csv"
TABLE_TEX = OUT_DIR / "table10_independent_evaluator.tex"
METADATA_JSON = OUT_DIR / "independent_extraction_metadata.json"
README_PATH = OUT_DIR / "README.md"

INPUT_FILES = {
    "GPT-4o": ROOT / "LLM Normal Test" / "gpt4o_test.csv",
    "Gemma 3-27B": ROOT / "LLM Normal Test" / "gemma3-27b_test.csv",
    "Llama 3.1": ROOT / "LLM Normal Test" / "llama3.1_test.csv",
    "Qwen 2.5-32B": ROOT / "LLM Normal Test" / "qwen2.5-32b_test.csv",
}

DATASET_FILE = ROOT / "Dataset" / "Autonomy_Dataset.json"

SYSTEM_PROMPT = """You are an independent answer-extraction evaluator.

Judge which option is semantically supported by the complete rationale.
Do not simply repeat the leading answer letter. If the leading letter and the
meaning of the rationale conflict, select the option supported by the rationale
and flag the conflict. Return UNCLEAR when the response does not support one
option reliably.

You must inspect the full stored model response, including both any explicit
answer letter and the rationale. Do not answer the scenario yourself.

Return strict JSON only with this exact schema:
{
  "evaluator_label": "A" | "B" | "C" | "UNCLEAR",
  "explicit_letter_in_response": "A" | "B" | "C" | "NONE",
  "letter_rationale_conflict": true | false,
  "short_explanation": "brief explanation"
}
"""

USER_PROMPT_TEMPLATE = """Scenario code: {scenario_code}

Scenario:
{scenario_text}

Option A:
{option_a}

Option B:
{option_b}

Option C:
{option_c}

Complete stored model response:
{raw_response}

Return strict JSON only."""

RESULT_COLUMNS = [
    "run_name",
    "prompt_version",
    "evaluator_model",
    "temperature",
    "model",
    "scenario_code",
    "original_raw_response",
    "original_parser_label",
    "evaluator_label",
    "explicit_letter_in_response",
    "letter_rationale_conflict",
    "agreement",
    "evaluator_status",
    "short_explanation",
    "raw_evaluator_output",
    "response_format_flags",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
]


def original_extract_choice(answer: Any) -> str | None:
    if not isinstance(answer, str) or not answer.strip():
        return None
    s = re.sub(r"^\s*Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
    m = re.match(r"^\s*([ABC])\b", s, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABC])\b", s[:80], flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def classify_format(raw_response: str) -> str:
    flags: list[str] = []
    if re.search(r"Answer\s*:", raw_response, flags=re.IGNORECASE):
        flags.append("answer_prefix")
    if "\n" in raw_response or "\r" in raw_response:
        flags.append("newline")
    if not re.match(r"^\s*[ABC]\s+-\s+", raw_response):
        flags.append("not_letter_dash")
    if len(raw_response) > 180:
        flags.append("long")
    return "|".join(flags) if flags else "standard"


def load_scenarios() -> pd.DataFrame:
    data = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for domain in data:
        for scenario in domain.get("scenarios", []):
            options = scenario.get("options", {})
            rows.append(
                {
                    "scenario_code": scenario.get("scenario_code", ""),
                    "scenario_text": scenario.get("text", ""),
                    "option_a": options.get("A", ""),
                    "option_b": options.get("B", ""),
                    "option_c": options.get("C", ""),
                }
            )
    df = pd.DataFrame(rows)
    if len(df) != 120 or df["scenario_code"].nunique() != 120:
        raise RuntimeError("Scenario dataset must contain 120 unique scenario codes")
    return df


def build_rows() -> pd.DataFrame:
    scenarios = load_scenarios()
    frames: list[pd.DataFrame] = []
    for model, input_path in INPUT_FILES.items():
        df = pd.read_csv(input_path)
        if len(df) != 120 or df["scenario_code"].nunique() != 120:
            raise RuntimeError(f"{input_path} must contain 120 unique scenario codes")
        df = df.copy()
        df["model"] = model
        df["original_parser_label"] = df["answer"].apply(original_extract_choice)
        if df["original_parser_label"].isna().any():
            bad = df.loc[df["original_parser_label"].isna(), "scenario_code"].tolist()
            raise RuntimeError(f"{input_path} has unparsed original labels: {bad}")
        df["response_format_flags"] = df["answer"].astype(str).apply(classify_format)
        joined = df.merge(scenarios, on="scenario_code", how="left", indicator=True)
        if (joined["_merge"] != "both").any():
            bad = joined.loc[joined["_merge"] != "both", "scenario_code"].tolist()
            raise RuntimeError(f"{input_path} has missing scenario joins: {bad}")
        frames.append(joined.drop(columns=["_merge"]))
    all_rows = pd.concat(frames, ignore_index=True)
    if len(all_rows) != 480:
        raise RuntimeError(f"Expected 480 rows, got {len(all_rows)}")
    return all_rows


def format_user_prompt(row: pd.Series) -> str:
    return USER_PROMPT_TEMPLATE.format(
        scenario_code=row["scenario_code"],
        scenario_text=row["scenario_text"],
        option_a=row["option_a"],
        option_b=row["option_b"],
        option_c=row["option_c"],
        raw_response=row["answer"],
    )


def parse_evaluator_output(raw_output: str) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if not m:
            return None, "malformed_json"
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, "malformed_json"

    label = str(payload.get("evaluator_label", "")).strip().upper()
    explicit = str(payload.get("explicit_letter_in_response", "")).strip().upper()
    conflict = payload.get("letter_rationale_conflict", None)
    explanation = str(payload.get("short_explanation", "")).strip()

    if label not in {"A", "B", "C", "UNCLEAR"}:
        return None, "invalid_evaluator_label"
    if explicit not in {"A", "B", "C", "NONE"}:
        return None, "invalid_explicit_letter"
    if not isinstance(conflict, bool):
        return None, "invalid_conflict_flag"
    if not explanation:
        return None, "missing_explanation"

    return (
        {
            "evaluator_label": label,
            "explicit_letter_in_response": explicit,
            "letter_rationale_conflict": conflict,
            "short_explanation": explanation,
        },
        "ok",
    )


def estimated_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1_000_000 * 2.50 + completion_tokens / 1_000_000 * 10.00


def normalize_existing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {"raw_evaluator_output": "raw_evaluator_output"}
    df = df.rename(columns=rename)
    if "run_name" not in df.columns:
        df.insert(0, "run_name", "pilot_seed")
    if "agreement" in df.columns:
        df["agreement"] = df["agreement"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"letter_rationale_conflict", "agreement"} else False
    return df[RESULT_COLUMNS]


def seed_cache_from_pilot() -> None:
    pieces: list[pd.DataFrame] = []
    if RESULTS_CSV.exists():
        pieces.append(normalize_existing(pd.read_csv(RESULTS_CSV)))
    if PILOT_RESULTS_CSV.exists():
        pieces.append(normalize_existing(pd.read_csv(PILOT_RESULTS_CSV)))
    if not pieces:
        return

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined[combined["evaluator_status"] == "ok"].copy()
    combined = combined.drop_duplicates(subset=["model", "scenario_code"], keep="first")
    combined.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)


def cached_ok_keys() -> set[tuple[str, str]]:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    ok = df[df["evaluator_status"] == "ok"]
    return set(zip(ok["model"], ok["scenario_code"]))


def append_result(record: dict[str, Any]) -> None:
    for col in RESULT_COLUMNS:
        record.setdefault(col, "")
    exists = RESULTS_CSV.exists()
    pd.DataFrame([{col: record[col] for col in RESULT_COLUMNS}]).to_csv(
        RESULTS_CSV,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def write_metadata(started_utc: str, completed_utc: str | None) -> None:
    metadata = {
        "run_name": RUN_NAME,
        "prompt_version": PROMPT_VERSION,
        "evaluator_model": EVALUATOR_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_attempts": MAX_ATTEMPTS,
        "response_format": {"type": "json_object"},
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "input_files": {model: str(path.relative_to(ROOT)) for model, path in INPUT_FILES.items()},
        "dataset_file": str(DATASET_FILE.relative_to(ROOT)),
        "pilot_seed_file": str(PILOT_RESULTS_CSV.relative_to(ROOT)),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "cost_basis": "Estimated from OpenRouter GPT-4o listed price: $2.50/M input tokens, $10/M output tokens.",
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    if df[["model", "scenario_code"]].duplicated().any():
        df = df.drop_duplicates(subset=["model", "scenario_code"], keep="first").copy()
        df.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    if len(df) != 480:
        raise RuntimeError(f"Expected 480 result rows, got {len(df)}")
    rows: list[dict[str, Any]] = []
    for model in INPUT_FILES:
        group = df[df["model"] == model]
        agreement_count = int(group["agreement"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        total = int(len(group))
        rows.append(
            {
                "model": model,
                "evaluated_responses": total,
                "original_A": int((group["original_parser_label"] == "A").sum()),
                "original_B": int((group["original_parser_label"] == "B").sum()),
                "original_C": int((group["original_parser_label"] == "C").sum()),
                "evaluator_A": int((group["evaluator_label"] == "A").sum()),
                "evaluator_B": int((group["evaluator_label"] == "B").sum()),
                "evaluator_C": int((group["evaluator_label"] == "C").sum()),
                "evaluator_UNCLEAR": int((group["evaluator_label"] == "UNCLEAR").sum()),
                "agreement_count": agreement_count,
                "agreement_pct": round(100 * agreement_count / total, 2) if total else 0.0,
                "disagreement_count": total - agreement_count,
                "letter_rationale_conflicts": int(group["letter_rationale_conflict"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
                "unclear_count": int((group["evaluator_label"] == "UNCLEAR").sum()),
                "failed_or_malformed": int((group["evaluator_status"] != "ok").sum()),
                "prompt_tokens": int(pd.to_numeric(group["prompt_tokens"], errors="coerce").fillna(0).sum()),
                "completion_tokens": int(pd.to_numeric(group["completion_tokens"], errors="coerce").fillna(0).sum()),
                "estimated_cost_usd": round(float(pd.to_numeric(group["estimated_cost_usd"], errors="coerce").fillna(0).sum()), 6),
            }
        )

    agreement_count = int(df["agreement"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    total = int(len(df))
    rows.append(
        {
            "model": "ALL",
            "evaluated_responses": total,
            "original_A": int((df["original_parser_label"] == "A").sum()),
            "original_B": int((df["original_parser_label"] == "B").sum()),
            "original_C": int((df["original_parser_label"] == "C").sum()),
            "evaluator_A": int((df["evaluator_label"] == "A").sum()),
            "evaluator_B": int((df["evaluator_label"] == "B").sum()),
            "evaluator_C": int((df["evaluator_label"] == "C").sum()),
            "evaluator_UNCLEAR": int((df["evaluator_label"] == "UNCLEAR").sum()),
            "agreement_count": agreement_count,
            "agreement_pct": round(100 * agreement_count / total, 2) if total else 0.0,
            "disagreement_count": total - agreement_count,
            "letter_rationale_conflicts": int(df["letter_rationale_conflict"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
            "unclear_count": int((df["evaluator_label"] == "UNCLEAR").sum()),
            "failed_or_malformed": int((df["evaluator_status"] != "ok").sum()),
            "prompt_tokens": int(pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).sum()),
            "completion_tokens": int(pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).sum()),
            "estimated_cost_usd": round(float(pd.to_numeric(df["estimated_cost_usd"], errors="coerce").fillna(0).sum()), 6),
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    disagreements = df[~df["agreement"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    disagreements.to_csv(DISAGREEMENTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    tex_cols = [
        "model",
        "evaluated_responses",
        "original_A",
        "original_B",
        "original_C",
        "evaluator_A",
        "evaluator_B",
        "evaluator_C",
        "evaluator_UNCLEAR",
        "agreement_count",
        "agreement_pct",
        "disagreement_count",
        "letter_rationale_conflicts",
        "unclear_count",
        "failed_or_malformed",
    ]
    summary[tex_cols].to_latex(TABLE_TEX, index=False, escape=True, float_format="%.2f")
    return summary


def write_readme(summary: pd.DataFrame, completed_utc: str) -> None:
    total = summary[summary["model"] == "ALL"].iloc[0]
    content = f"""# Independent Extraction Robustness

This isolated analysis compares the original leading-letter parser with an
independent semantic evaluator on the same stored baseline responses used by
the main baseline analysis and Tables 8-9.

## Inputs

- `LLM Normal Test/gpt4o_test.csv`
- `LLM Normal Test/gemma3-27b_test.csv`
- `LLM Normal Test/llama3.1_test.csv`
- `LLM Normal Test/qwen2.5-32b_test.csv`
- `Dataset/Autonomy_Dataset.json`

Each response CSV contains 120 unique scenario codes. All 480 responses join to
the scenario dataset and exact Option A/B/C text.

## Evaluator

- Evaluator: OpenRouter chat completion
- Evaluator model/checkpoint: `{EVALUATOR_MODEL}`
- Prompt version: `{PROMPT_VERSION}`
- Decoding: `temperature={TEMPERATURE}`, `max_tokens={MAX_TOKENS}`
- Response format: strict JSON object
- Conversational history: none; each response is evaluated independently.
- Completed UTC: `{completed_utc}`

## Evaluator Prompt

System prompt:

```text
{SYSTEM_PROMPT}
```

User prompt template:

```text
{USER_PROMPT_TEMPLATE}
```

## Outputs

- `independent_extraction_results.csv`
- `independent_extraction_summary.csv`
- `disagreements.csv`
- `table10_independent_evaluator.tex`
- `independent_extraction_metadata.json`

## Result Snapshot

- Evaluated responses: {int(total["evaluated_responses"])}
- Agreement: {int(total["agreement_count"])} / {int(total["evaluated_responses"])} ({float(total["agreement_pct"]):.2f}%)
- Disagreements: {int(total["disagreement_count"])}
- Letter/rationale conflicts: {int(total["letter_rationale_conflicts"])}
- UNCLEAR: {int(total["unclear_count"])}
- Failed or malformed evaluator outputs: {int(total["failed_or_malformed"])}
- Prompt tokens: {int(total["prompt_tokens"])}
- Completion tokens: {int(total["completion_tokens"])}
- Estimated cost: ${float(total["estimated_cost_usd"]):.6f}

## Limitations

- This analysis does not rerun the evaluated LLMs.
- OpenRouter/GPT-4o is an LLM-based evaluator. It is independent from the
  stored response-generation calls, but not model-family-independent for the
  GPT-4o response subset.
- Costs are estimated from usage tokens and listed OpenRouter GPT-4o pricing at
  the time of execution.
"""
    README_PATH.write_text(content, encoding="utf-8")


def run() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    started_utc = datetime.now(timezone.utc).isoformat()
    write_metadata(started_utc=started_utc, completed_utc=None)
    rows = build_rows()
    seed_cache_from_pilot()
    cached = cached_ok_keys()
    print(f"Cached successful rows: {len(cached)}")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    for index, row in rows.iterrows():
        key = (row["model"], row["scenario_code"])
        if key in cached:
            continue

        user_prompt = format_user_prompt(row)
        last_status = "not_attempted"
        last_raw = ""
        last_usage = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.chat.completions.create(
                    model=EVALUATOR_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_status = f"api_error:{type(exc).__name__}"
                last_raw = str(exc)
                time.sleep(2 * attempt)
                continue

            raw_output = (response.choices[0].message.content or "").strip()
            parsed, status = parse_evaluator_output(raw_output)
            last_status = status
            last_raw = raw_output
            last_usage = response.usage

            if parsed is None:
                continue

            prompt_tokens = int(getattr(last_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(last_usage, "completion_tokens", 0) or 0)
            record = {
                "run_name": RUN_NAME,
                "prompt_version": PROMPT_VERSION,
                "evaluator_model": EVALUATOR_MODEL,
                "temperature": TEMPERATURE,
                "model": row["model"],
                "scenario_code": row["scenario_code"],
                "original_raw_response": row["answer"],
                "original_parser_label": row["original_parser_label"],
                "evaluator_label": parsed["evaluator_label"],
                "explicit_letter_in_response": parsed["explicit_letter_in_response"],
                "letter_rationale_conflict": parsed["letter_rationale_conflict"],
                "agreement": parsed["evaluator_label"] == row["original_parser_label"],
                "evaluator_status": "ok",
                "short_explanation": parsed["short_explanation"],
                "raw_evaluator_output": raw_output,
                "response_format_flags": row["response_format_flags"],
                "attempts": attempt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_cost_usd": estimated_cost(prompt_tokens, completion_tokens),
            }
            append_result(record)
            cached.add(key)
            print(f"[{len(cached):03d}/480] OK {row['model']} {row['scenario_code']} -> {parsed['evaluator_label']}")
            break

        if key not in cached:
            prompt_tokens = int(getattr(last_usage, "prompt_tokens", 0) or 0) if last_usage else 0
            completion_tokens = int(getattr(last_usage, "completion_tokens", 0) or 0) if last_usage else 0
            record = {
                "run_name": RUN_NAME,
                "prompt_version": PROMPT_VERSION,
                "evaluator_model": EVALUATOR_MODEL,
                "temperature": TEMPERATURE,
                "model": row["model"],
                "scenario_code": row["scenario_code"],
                "original_raw_response": row["answer"],
                "original_parser_label": row["original_parser_label"],
                "evaluator_label": "UNCLEAR",
                "explicit_letter_in_response": "NONE",
                "letter_rationale_conflict": False,
                "agreement": False,
                "evaluator_status": last_status,
                "short_explanation": f"Evaluator did not return valid output after {MAX_ATTEMPTS} attempts.",
                "raw_evaluator_output": last_raw,
                "response_format_flags": row["response_format_flags"],
                "attempts": MAX_ATTEMPTS,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_cost_usd": estimated_cost(prompt_tokens, completion_tokens),
            }
            append_result(record)
            cached.add(key)
            print(f"[{len(cached):03d}/480] FAILED {row['model']} {row['scenario_code']} status={last_status}")

    completed_utc = datetime.now(timezone.utc).isoformat()
    write_metadata(started_utc=started_utc, completed_utc=completed_utc)
    summary = summarize_results()
    write_readme(summary=summary, completed_utc=completed_utc)
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DISAGREEMENTS_CSV}")
    print(f"Wrote {TABLE_TEX}")
    print(f"Wrote {METADATA_JSON}")
    print(f"Wrote {README_PATH}")


if __name__ == "__main__":
    run()
