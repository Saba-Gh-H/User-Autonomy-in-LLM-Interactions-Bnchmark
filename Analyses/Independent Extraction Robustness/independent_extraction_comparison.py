"""Independent answer-extraction robustness analysis.

This script evaluates stored model responses against the original scenario and
option text. It never asks the evaluated LLMs to answer the scenarios again.

Default mode validates inputs and reports the planned evaluator configuration.
Use --run-evaluator only after explicit approval for external evaluator calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Analyses" / "Independent Extraction Robustness"

INPUT_FILES = {
    "GPT-4o": ROOT / "LLM Normal Test" / "gpt4o_test.csv",
    "Gemma 3-27B": ROOT / "LLM Normal Test" / "gemma3-27b_test.csv",
    "Llama 3.1": ROOT / "LLM Normal Test" / "llama3.1_test.csv",
    "Qwen 2.5-32B": ROOT / "LLM Normal Test" / "qwen2.5-32b_test.csv",
}

DATASET_FILE = ROOT / "Dataset" / "Autonomy_Dataset.json"

RESULTS_CSV = OUT_DIR / "independent_extraction_results.csv"
SUMMARY_CSV = OUT_DIR / "independent_extraction_summary.csv"
DISAGREEMENTS_CSV = OUT_DIR / "disagreements.csv"
TABLE_TEX = OUT_DIR / "table10_independent_evaluator.tex"

DEFAULT_EVALUATOR = "openai/gpt-4o"

SYSTEM_PROMPT = """You are an independent answer-extraction evaluator.
Your task is to determine which of the three provided options is supported by
the stored model response.

Rules:
- Consider the complete response, including both the visible answer letter and
  the rationale.
- Do not answer the scenario yourself.
- If the response clearly supports one option, return that option letter.
- If the visible letter and rationale conflict, choose the option best
  supported by the full response and explain the conflict briefly.
- If the supported option cannot be determined, return UNCLEAR.
- Return only valid JSON with keys: label, explanation.
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

Stored model response:
{raw_response}

Which option does the stored response semantically support?
Return JSON exactly like:
{{"label":"A|B|C|UNCLEAR","explanation":"short reason"}}
"""


@dataclass(frozen=True)
class EvaluationRow:
    model: str
    scenario_code: str
    original_raw_response: str
    original_parser_label: str
    scenario_text: str
    option_a: str
    option_b: str
    option_c: str


def original_extract_choice(answer: Any) -> str | None:
    """Tables 8/9 leading-letter extractor."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    s = re.sub(r"^\s*Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
    m = re.match(r"^\s*([ABC])\b", s, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABC])\b", s[:80], flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_scenarios() -> pd.DataFrame:
    with DATASET_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

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
    scenarios = pd.DataFrame(rows)
    if scenarios["scenario_code"].duplicated().any():
        dupes = scenarios.loc[scenarios["scenario_code"].duplicated(), "scenario_code"].tolist()
        raise ValueError(f"Duplicate scenario codes in dataset: {dupes}")
    if len(scenarios) != 120:
        raise ValueError(f"Expected 120 scenarios, found {len(scenarios)} in {DATASET_FILE}")
    return scenarios


def build_evaluation_rows() -> list[EvaluationRow]:
    scenarios = load_scenarios()
    rows: list[EvaluationRow] = []

    for model, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = {"scenario_code", "answer"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if len(df) != 120:
            raise ValueError(f"{path} expected 120 rows, found {len(df)}")
        if df["scenario_code"].nunique() != 120:
            dupes = df.loc[df["scenario_code"].duplicated(), "scenario_code"].tolist()
            raise ValueError(f"{path} duplicate scenario codes: {dupes}")

        df = df.copy()
        df["original_parser_label"] = df["answer"].apply(original_extract_choice)
        if df["original_parser_label"].isna().any():
            bad = df.loc[df["original_parser_label"].isna(), "scenario_code"].tolist()
            raise ValueError(f"{path} has unparsed rows: {bad}")

        joined = df.merge(scenarios, on="scenario_code", how="left", indicator=True)
        missing_join = joined.loc[joined["_merge"] != "both", "scenario_code"].tolist()
        if missing_join:
            raise ValueError(f"{path} rows missing from scenario dataset: {missing_join}")

        for rec in joined.to_dict("records"):
            rows.append(
                EvaluationRow(
                    model=model,
                    scenario_code=str(rec["scenario_code"]),
                    original_raw_response=str(rec["answer"]),
                    original_parser_label=str(rec["original_parser_label"]),
                    scenario_text=str(rec["scenario_text"]),
                    option_a=str(rec["option_a"]),
                    option_b=str(rec["option_b"]),
                    option_c=str(rec["option_c"]),
                )
            )

    if len(rows) != 480:
        raise ValueError(f"Expected 480 evaluation rows, built {len(rows)}")
    return rows


def detect_xfinder() -> tuple[bool, str]:
    try:
        import importlib.util

        spec = importlib.util.find_spec("xfinder") or importlib.util.find_spec("xFinder")
        return (spec is not None, str(spec.origin) if spec else "")
    except Exception as exc:  # pragma: no cover - defensive diagnostic
        return False, f"error checking xFinder: {exc}"


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def format_user_prompt(row: EvaluationRow) -> str:
    return USER_PROMPT_TEMPLATE.format(
        scenario_code=row.scenario_code,
        scenario_text=row.scenario_text,
        option_a=row.option_a,
        option_b=row.option_b,
        option_c=row.option_c,
        raw_response=row.original_raw_response,
    )


def estimate_cost(rows: list[EvaluationRow], output_tokens_per_call: int = 60) -> dict[str, float]:
    input_tokens = 0
    for row in rows:
        input_tokens += estimate_tokens(SYSTEM_PROMPT)
        input_tokens += estimate_tokens(format_user_prompt(row))
    output_tokens = len(rows) * output_tokens_per_call
    # OpenRouter listing for openai/gpt-4o: $2.50/M input, $10/M output.
    input_cost = input_tokens / 1_000_000 * 2.50
    output_cost = output_tokens / 1_000_000 * 10.00
    return {
        "calls": float(len(rows)),
        "estimated_input_tokens": float(input_tokens),
        "estimated_output_tokens": float(output_tokens),
        "estimated_cost_usd": input_cost + output_cost,
    }


def parse_evaluator_json(text: str) -> tuple[str, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return "UNCLEAR", f"Invalid JSON from evaluator: {text[:200]}"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "UNCLEAR", f"Invalid JSON from evaluator: {text[:200]}"
    label = str(data.get("label", "UNCLEAR")).strip().upper()
    if label not in {"A", "B", "C", "UNCLEAR"}:
        label = "UNCLEAR"
    explanation = str(data.get("explanation", "")).strip()
    return label, explanation


def evaluate_with_openrouter(rows: list[EvaluationRow], evaluator_model: str) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        response = client.chat.completions.create(
            model=evaluator_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_prompt(row)},
            ],
            temperature=0,
            max_tokens=120,
        )
        content = (response.choices[0].message.content or "").strip()
        label, explanation = parse_evaluator_json(content)
        agreement = label == row.original_parser_label
        results.append(
            {
                "model": row.model,
                "scenario_code": row.scenario_code,
                "original_raw_response": row.original_raw_response,
                "original_parser_label": row.original_parser_label,
                "independent_evaluator_label": label,
                "agreement": agreement,
                "evaluator_status": "ok",
                "evaluator_explanation": explanation,
            }
        )
        print(f"[{index:03d}/{len(rows)}] {row.model} {row.scenario_code}: {label}")

    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in results.groupby("model", sort=False):
        original_counts = group["original_parser_label"].value_counts()
        eval_counts = group["independent_evaluator_label"].value_counts()
        agreements = int(group["agreement"].sum())
        total = int(len(group))
        rows.append(
            {
                "model": model,
                "evaluated_responses": total,
                "original_A": int(original_counts.get("A", 0)),
                "original_B": int(original_counts.get("B", 0)),
                "original_C": int(original_counts.get("C", 0)),
                "independent_A": int(eval_counts.get("A", 0)),
                "independent_B": int(eval_counts.get("B", 0)),
                "independent_C": int(eval_counts.get("C", 0)),
                "independent_UNCLEAR": int(eval_counts.get("UNCLEAR", 0)),
                "agreement_count": agreements,
                "agreement_pct": round(100 * agreements / total, 2) if total else 0.0,
                "disagreement_count": total - agreements,
                "invalid_or_unresolved": int((group["evaluator_status"] != "ok").sum())
                + int((group["independent_evaluator_label"] == "UNCLEAR").sum()),
            }
        )
    total_agree = int(results["agreement"].sum())
    total = int(len(results))
    rows.append(
        {
            "model": "ALL",
            "evaluated_responses": total,
            "original_A": int((results["original_parser_label"] == "A").sum()),
            "original_B": int((results["original_parser_label"] == "B").sum()),
            "original_C": int((results["original_parser_label"] == "C").sum()),
            "independent_A": int((results["independent_evaluator_label"] == "A").sum()),
            "independent_B": int((results["independent_evaluator_label"] == "B").sum()),
            "independent_C": int((results["independent_evaluator_label"] == "C").sum()),
            "independent_UNCLEAR": int((results["independent_evaluator_label"] == "UNCLEAR").sum()),
            "agreement_count": total_agree,
            "agreement_pct": round(100 * total_agree / total, 2) if total else 0.0,
            "disagreement_count": total - total_agree,
            "invalid_or_unresolved": int((results["evaluator_status"] != "ok").sum())
            + int((results["independent_evaluator_label"] == "UNCLEAR").sum()),
        }
    )
    return pd.DataFrame(rows)


def write_outputs(rows: list[EvaluationRow], results: list[dict[str, Any]], evaluator_model: str) -> None:
    results_df = pd.DataFrame(results)
    summary_df = summarize(results_df)
    disagreements = results_df[~results_df["agreement"]].copy()

    scenario_cols = {
        (row.model, row.scenario_code): {
            "scenario_text": row.scenario_text,
            "option_a": row.option_a,
            "option_b": row.option_b,
            "option_c": row.option_c,
        }
        for row in rows
    }
    if not disagreements.empty:
        extras = disagreements.apply(
            lambda rec: scenario_cols[(rec["model"], rec["scenario_code"])],
            axis=1,
            result_type="expand",
        )
        disagreements = pd.concat([disagreements, extras], axis=1)

    results_df.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    disagreements.to_csv(DISAGREEMENTS_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_latex(TABLE_TEX, index=False, escape=True, float_format="%.2f")

    update_readme(evaluator_model=evaluator_model, executed=True)


def update_readme(evaluator_model: str, executed: bool) -> None:
    xfinder_found, xfinder_detail = detect_xfinder()
    timestamp = datetime.now(timezone.utc).isoformat() if executed else "not executed yet"
    content = f"""# Independent Extraction Robustness

This isolated analysis compares the original leading-letter parser with an
independent semantic evaluator on the same stored baseline responses.

## Inputs

- `LLM Normal Test/gpt4o_test.csv`
- `LLM Normal Test/gemma3-27b_test.csv`
- `LLM Normal Test/llama3.1_test.csv`
- `LLM Normal Test/qwen2.5-32b_test.csv`
- `Dataset/Autonomy_Dataset.json`

Each input response CSV is expected to contain exactly 120 unique
`scenario_code` values and an `answer` column. The scenario dataset is expected
to contain the matching scenario text and exact Option A/B/C text.

## Evaluator

- xFinder available: `{xfinder_found}`
- xFinder detail: `{xfinder_detail}`
- Selected fallback evaluator: OpenRouter chat completion
- Evaluator model/checkpoint: `{evaluator_model}`
- Decoding: `temperature=0`, `max_tokens=120`
- Conversational history: none; each response is evaluated independently.
- Date of execution: `{timestamp}`

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

## Limitations

- This does not rerun the evaluated LLMs; it evaluates only stored responses.
- If OpenRouter/GPT-4o is used, it is an LLM-based evaluator rather than
  xFinder. It may still make semantic judgment errors.
- GPT-4o is also one of the evaluated models, so the evaluator is independent
  from the response-generation call but not independent by model family for the
  GPT-4o subset.
- Costs and provider behavior depend on OpenRouter routing and pricing at run
  time.
"""
    (OUT_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-evaluator", action="store_true", help="make external evaluator calls")
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR)
    args = parser.parse_args()

    rows = build_evaluation_rows()
    xfinder_found, xfinder_detail = detect_xfinder()
    cost = estimate_cost(rows)

    print("Input validation: OK")
    print(f"Evaluation rows: {len(rows)}")
    print(f"xFinder available: {xfinder_found} {xfinder_detail}")
    print(f"Selected evaluator: OpenRouter {args.evaluator_model}")
    print(f"OPENROUTER_API_KEY present: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    print(
        "Estimated calls/tokens/cost: "
        f"{int(cost['calls'])} calls, "
        f"{int(cost['estimated_input_tokens'])} input tokens, "
        f"{int(cost['estimated_output_tokens'])} output tokens, "
        f"${cost['estimated_cost_usd']:.2f}"
    )

    if not args.run_evaluator:
        update_readme(evaluator_model=args.evaluator_model, executed=False)
        print("Dry run only. Re-run with --run-evaluator after approval.")
        return 0

    results = evaluate_with_openrouter(rows, evaluator_model=args.evaluator_model)
    write_outputs(rows=rows, results=results, evaluator_model=args.evaluator_model)
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DISAGREEMENTS_CSV}")
    print(f"Wrote {TABLE_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
