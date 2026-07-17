"""20-row pilot for independent answer-extraction robustness.

Creates only pilot outputs in this directory. It never reruns the evaluated
LLMs and never overwrites the original response CSVs.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Analyses" / "Independent Extraction Robustness"

PROMPT_VERSION = "independent_extraction_pilot_v1"
EVALUATOR_MODEL = "openai/gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 180
MAX_ATTEMPTS = 3

RESULTS_CSV = OUT_DIR / "pilot_independent_extraction_results.csv"
SUMMARY_CSV = OUT_DIR / "pilot_independent_extraction_summary.csv"
DISAGREEMENTS_CSV = OUT_DIR / "pilot_disagreements.csv"
METADATA_JSON = OUT_DIR / "pilot_prompt_metadata.json"

INPUT_FILES = {
    "GPT-4o": ROOT / "LLM Normal Test" / "gpt4o_test.csv",
    "Gemma 3-27B": ROOT / "LLM Normal Test" / "gemma3-27b_test.csv",
    "Llama 3.1": ROOT / "LLM Normal Test" / "llama3.1_test.csv",
    "Qwen 2.5-32B": ROOT / "LLM Normal Test" / "qwen2.5-32b_test.csv",
}

DATASET_FILE = ROOT / "Dataset" / "Autonomy_Dataset.json"

PILOT_SCENARIOS = {
    "GPT-4o": ["HC_EPI01", "EDU01_EPI02", "EDU01_EPI01", "LAW03_REL01", "SPR01_EPI02"],
    "Gemma 3-27B": ["EDU03_REL02", "FIN01_EPI01", "HC_MOR02", "SPR03_REL01", "ART02_MOR01"],
    "Llama 3.1": ["HC_EPI01", "FIN01_EPI01", "EDU02_MOR02", "LAW02_MOR02", "WRK03_REL02"],
    "Qwen 2.5-32B": ["FIN02_REL02", "CPR02_MOR02", "WRK03_REL01", "SPR01_EPI01", "MED03_REL01"],
}

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


def original_extract_choice(answer: Any) -> str | None:
    if not isinstance(answer, str) or not answer.strip():
        return None
    s = re.sub(r"^\s*Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
    m = re.match(r"^\s*([ABC])\b", s, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABC])\b", s[:80], flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


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


def build_pilot_rows() -> pd.DataFrame:
    scenarios = load_scenarios()
    all_rows: list[pd.DataFrame] = []
    for model, input_path in INPUT_FILES.items():
        df = pd.read_csv(input_path)
        if len(df) != 120 or df["scenario_code"].nunique() != 120:
            raise RuntimeError(f"{input_path} must contain 120 unique scenario codes")
        df["model"] = model
        df["original_parser_label"] = df["answer"].apply(original_extract_choice)
        df["response_format_flags"] = df["answer"].astype(str).apply(classify_format)
        wanted = PILOT_SCENARIOS[model]
        sub = df[df["scenario_code"].isin(wanted)].copy()
        if len(sub) != 5:
            missing = sorted(set(wanted) - set(sub["scenario_code"]))
            raise RuntimeError(f"Missing pilot rows for {model}: {missing}")
        sub["pilot_order"] = sub["scenario_code"].map({code: i for i, code in enumerate(wanted)})
        sub = sub.sort_values("pilot_order")
        joined = sub.merge(scenarios, on="scenario_code", how="left", indicator=True)
        if (joined["_merge"] != "both").any():
            missing = joined.loc[joined["_merge"] != "both", "scenario_code"].tolist()
            raise RuntimeError(f"Missing scenario joins for {model}: {missing}")
        all_rows.append(joined)

    pilot = pd.concat(all_rows, ignore_index=True)
    if len(pilot) != 20:
        raise RuntimeError(f"Expected 20 pilot rows, got {len(pilot)}")
    return pilot


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


def get_cached_keys() -> set[tuple[str, str]]:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    ok = df[df["evaluator_status"] == "ok"]
    return set(zip(ok["model"], ok["scenario_code"]))


def append_result(record: dict[str, Any]) -> None:
    exists = RESULTS_CSV.exists()
    pd.DataFrame([record]).to_csv(
        RESULTS_CSV,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def write_metadata() -> None:
    metadata = {
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
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize_results() -> None:
    df = pd.read_csv(RESULTS_CSV)
    rows: list[dict[str, Any]] = []
    for model, group in df.groupby("model", sort=False):
        agreement_count = int(group["agreement"].sum())
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
                "letter_rationale_conflicts": int(group["letter_rationale_conflict"].sum()),
                "unclear_count": int((group["evaluator_label"] == "UNCLEAR").sum()),
                "prompt_tokens": int(group["prompt_tokens"].sum()),
                "completion_tokens": int(group["completion_tokens"].sum()),
                "estimated_cost_usd": round(float(group["estimated_cost_usd"].sum()), 6),
            }
        )
    agreement_count = int(df["agreement"].sum())
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
            "letter_rationale_conflicts": int(df["letter_rationale_conflict"].sum()),
            "unclear_count": int((df["evaluator_label"] == "UNCLEAR").sum()),
            "prompt_tokens": int(df["prompt_tokens"].sum()),
            "completion_tokens": int(df["completion_tokens"].sum()),
            "estimated_cost_usd": round(float(df["estimated_cost_usd"].sum()), 6),
        }
    )
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    df[~df["agreement"]].to_csv(DISAGREEMENTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)


def estimated_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1_000_000 * 2.50 + completion_tokens / 1_000_000 * 10.00


def run() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    write_metadata()
    pilot = build_pilot_rows()
    cached = get_cached_keys()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    for _, row in pilot.iterrows():
        key = (row["model"], row["scenario_code"])
        if key in cached:
            print(f"SKIP cached {key[0]} {key[1]}")
            continue

        user_prompt = format_user_prompt(row)
        last_status = "not_attempted"
        last_raw = ""
        last_usage = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
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
            raw_output = (response.choices[0].message.content or "").strip()
            parsed, status = parse_evaluator_output(raw_output)
            last_status = status
            last_raw = raw_output
            last_usage = response.usage

            if parsed is not None:
                prompt_tokens = int(getattr(last_usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(last_usage, "completion_tokens", 0) or 0)
                record = {
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
                    "short_explanation": parsed["short_explanation"],
                    "raw_evaluator_output": raw_output,
                    "evaluator_status": "ok",
                    "agreement": parsed["evaluator_label"] == row["original_parser_label"],
                    "response_format_flags": row["response_format_flags"],
                    "attempts": attempt,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": estimated_cost(prompt_tokens, completion_tokens),
                }
                append_result(record)
                cached.add(key)
                print(f"OK {row['model']} {row['scenario_code']} -> {parsed['evaluator_label']}")
                break

            print(f"RETRY {row['model']} {row['scenario_code']} attempt={attempt} status={status}")

        if key not in cached:
            prompt_tokens = int(getattr(last_usage, "prompt_tokens", 0) or 0) if last_usage else 0
            completion_tokens = int(getattr(last_usage, "completion_tokens", 0) or 0) if last_usage else 0
            record = {
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
                "short_explanation": f"Evaluator failed after retries: {last_status}",
                "raw_evaluator_output": last_raw,
                "evaluator_status": last_status,
                "agreement": False,
                "response_format_flags": row["response_format_flags"],
                "attempts": MAX_ATTEMPTS,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_cost_usd": estimated_cost(prompt_tokens, completion_tokens),
            }
            append_result(record)
            cached.add(key)

    summarize_results()
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DISAGREEMENTS_CSV}")
    print(f"Wrote {METADATA_JSON}")


if __name__ == "__main__":
    run()
