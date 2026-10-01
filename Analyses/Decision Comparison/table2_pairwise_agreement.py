"""Reproduce the pairwise agreement and Cohen's kappa results in Table 2.

The analysis reads the original T=0 baseline response CSVs, extracts the
leading A/B/C answer, pairs responses by scenario_code, and computes exact
agreement and unweighted Cohen's kappa. Source CSVs are never modified.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd


MODEL_FILES = {
    "Aya Expanse 32B": "aya-expanse-32b_test.csv",
    "Gemma 3 12B": "gemma3-12b_test.csv",
    "Gemma 3 27B": "gemma3-27b_test.csv",
    "GPT-4o": "gpt4o_test.csv",
    "Llama 3.1 70B": "llama3.1_test.csv",
    "Llama 3.2 3B": "llama3.2-latest_test.csv",
    "Llama 3.3 70B": "llama3.3-latest_test.csv",
    "Mistral Small 3.2 24B": "mistral-small-3.2-24b_test.csv",
    "Mistral Small 3.1 24B": "mistral-small3.1-latest_test.csv",
    "Qwen2.5 14B": "qwen2.5-14b_test.csv",
    "Qwen2.5 32B": "qwen2.5-32b_test.csv",
    "Qwen2.5 72B": "qwen2.5-72b_test.csv",
}

AYA_MISSING_SCENARIO = "LAW01_EPI02"

# The final six rows in the accepted Qwen2.5 72B CSV have copied RND codes.
# Their positions are the six TRN positions in the canonical dataset. Correct
# only the analysis copy after verifying the complete known mismatch pattern.
QWEN_SCENARIO_CODE_CORRECTIONS = {
    114: ("RND01_EPI01", "TRN01_EPI01"),
    115: ("RND01_EPI02", "TRN01_EPI02"),
    116: ("RND02_MOR01", "TRN02_MOR01"),
    117: ("RND02_MOR02", "TRN02_MOR02"),
    118: ("RND03_REL01", "TRN03_REL01"),
    119: ("RND03_REL02", "TRN03_REL02"),
}

ANSWER_PATTERN = re.compile(
    r"^\s*(?:Answer\s*:\s*)?([ABC])(?:\b|(?=\s*[-:.)]))",
    flags=re.IGNORECASE,
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from the current or notebook directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "LLM Normal Test").is_dir() and (
            candidate / "Dataset" / "Autonomy_Dataset.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")


def _collect_scenario_codes(value: object, codes: list[str]) -> None:
    if isinstance(value, dict):
        if "scenario_code" in value:
            codes.append(str(value["scenario_code"]))
            return
        for nested in value.values():
            _collect_scenario_codes(nested, codes)
    elif isinstance(value, list):
        for nested in value:
            _collect_scenario_codes(nested, codes)


def load_canonical_scenario_codes(repo_root: Path) -> list[str]:
    """Load the 120 scenario codes from the canonical root dataset."""
    dataset_path = repo_root / "Dataset" / "Autonomy_Dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    codes: list[str] = []
    _collect_scenario_codes(dataset, codes)
    if len(codes) != 120 or len(set(codes)) != 120:
        raise ValueError(
            f"Expected 120 unique canonical scenario codes, found "
            f"{len(codes)} rows and {len(set(codes))} unique codes."
        )
    return codes


def extract_choice(response: object) -> str | None:
    """Extract the accepted baseline leading answer letter."""
    match = ANSWER_PATTERN.match(str(response))
    return match.group(1).upper() if match else None


def _correct_qwen_scenario_metadata(df: pd.DataFrame) -> pd.DataFrame:
    corrected = df.copy()
    observed = {
        row_index: (str(corrected.at[row_index, "scenario_code"]), replacement)
        for row_index, (_, replacement) in QWEN_SCENARIO_CODE_CORRECTIONS.items()
    }
    expected = {
        row_index: (source, replacement)
        for row_index, (source, replacement) in QWEN_SCENARIO_CODE_CORRECTIONS.items()
    }
    if observed != expected:
        raise ValueError(
            "Qwen2.5 72B scenario-code metadata no longer matches the known "
            "six-row correction pattern; refusing to apply an implicit repair."
        )
    for row_index, (_, replacement) in QWEN_SCENARIO_CODE_CORRECTIONS.items():
        corrected.at[row_index, "scenario_code"] = replacement
    return corrected


def load_baseline_choices(
    repo_root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load and validate all baseline choices without changing source files."""
    canonical_codes = set(load_canonical_scenario_codes(repo_root))
    input_dir = repo_root / "LLM Normal Test"
    model_frames: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict[str, object]] = []

    for model, filename in MODEL_FILES.items():
        path = input_dir / filename
        df = pd.read_csv(path, dtype=str)
        required = {"scenario_code", "answer"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} is missing columns: {sorted(required - set(df.columns))}")

        expected_rows = 119 if model == "Aya Expanse 32B" else 120
        if len(df) != expected_rows:
            raise ValueError(f"{path} has {len(df)} rows; expected {expected_rows}.")

        source_unique_codes = int(df["scenario_code"].nunique())
        analysis_df = _correct_qwen_scenario_metadata(df) if model == "Qwen2.5 72B" else df.copy()
        analysis_df["choice"] = analysis_df["answer"].map(extract_choice)

        unparsed = int(analysis_df["choice"].isna().sum())
        if unparsed:
            bad_rows = analysis_df.index[analysis_df["choice"].isna()].tolist()
            raise ValueError(f"{path} has unparsed answer rows: {bad_rows}")
        if analysis_df["scenario_code"].duplicated().any():
            duplicates = sorted(
                analysis_df.loc[
                    analysis_df["scenario_code"].duplicated(keep=False), "scenario_code"
                ].unique()
            )
            raise ValueError(f"{path} has duplicate analysis scenario codes: {duplicates}")

        observed_codes = set(analysis_df["scenario_code"])
        expected_missing = {AYA_MISSING_SCENARIO} if model == "Aya Expanse 32B" else set()
        missing_codes = canonical_codes - observed_codes
        extra_codes = observed_codes - canonical_codes
        if missing_codes != expected_missing or extra_codes:
            raise ValueError(
                f"{path} scenario mismatch: missing={sorted(missing_codes)}, "
                f"extra={sorted(extra_codes)}"
            )

        model_frames[model] = analysis_df[["scenario_code", "choice"]].copy()
        validation_rows.append(
            {
                "model": model,
                "source_file": path.relative_to(repo_root).as_posix(),
                "source_rows": len(df),
                "source_unique_scenario_codes": source_unique_codes,
                "analysis_unique_scenario_codes": analysis_df["scenario_code"].nunique(),
                "unparsed_answers": unparsed,
                "analysis_only_metadata_corrections": (
                    len(QWEN_SCENARIO_CODE_CORRECTIONS) if model == "Qwen2.5 72B" else 0
                ),
            }
        )

    return model_frames, pd.DataFrame(validation_rows)


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    observed = float((first == second).mean())
    first_props = first.value_counts(normalize=True)
    second_props = second.value_counts(normalize=True)
    expected = sum(
        float(first_props.get(label, 0.0) * second_props.get(label, 0.0))
        for label in ("A", "B", "C")
    )
    return (observed - expected) / (1.0 - expected)


def calculate_pairwise_agreement(
    model_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate all 66 scenario-matched model comparisons."""
    rows: list[dict[str, object]] = []
    for model_1, model_2 in combinations(MODEL_FILES, 2):
        paired = model_frames[model_1].merge(
            model_frames[model_2],
            on="scenario_code",
            how="inner",
            suffixes=("_1", "_2"),
            validate="one_to_one",
        )
        expected_pairs = 119 if "Aya Expanse 32B" in (model_1, model_2) else 120
        if len(paired) != expected_pairs:
            raise ValueError(
                f"{model_1} / {model_2} produced {len(paired)} pairs; "
                f"expected {expected_pairs}."
            )

        agreement_count = int((paired["choice_1"] == paired["choice_2"]).sum())
        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "n_compared": len(paired),
                "agreement_count": agreement_count,
                "agreement": agreement_count / len(paired),
                "cohen_kappa": _cohen_kappa(paired["choice_1"], paired["choice_2"]),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != 66:
        raise ValueError(f"Expected 66 pairwise comparisons, found {len(result)}.")
    return result


def select_table2_rows(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Select the five highest and five lowest kappa rows shown in Table 2."""
    top = pairwise.nlargest(5, "cohen_kappa").copy()
    bottom = pairwise.nsmallest(5, "cohen_kappa").copy()
    top.insert(0, "rank_group", "Highest agreement")
    bottom.insert(0, "rank_group", "Lowest agreement")
    top.insert(1, "rank", range(1, 6))
    bottom.insert(1, "rank", range(1, 6))
    table = pd.concat([top, bottom], ignore_index=True)
    table.insert(2, "model_pair", table["model_1"] + " -- " + table["model_2"])
    return table[
        [
            "rank_group",
            "rank",
            "model_pair",
            "model_1",
            "model_2",
            "n_compared",
            "agreement_count",
            "agreement",
            "cohen_kappa",
        ]
    ]


def run_analysis(repo_root: Path | None = None, write_outputs: bool = True) -> dict[str, object]:
    """Run the complete Table 2 analysis and optionally write derived CSVs."""
    root = (repo_root or find_repo_root()).resolve()
    model_frames, validation = load_baseline_choices(root)
    pairwise = calculate_pairwise_agreement(model_frames)
    table2 = select_table2_rows(pairwise)
    output_paths: list[Path] = []

    if write_outputs:
        output_dir = root / "Analyses" / "Decision Comparison"
        pairwise_path = output_dir / "table2_pairwise_agreement.csv"
        table2_path = output_dir / "table2_top_bottom.csv"
        pairwise.to_csv(pairwise_path, index=False, float_format="%.6f")
        table2.to_csv(table2_path, index=False, float_format="%.6f")
        output_paths = [pairwise_path, table2_path]

    return {
        "repo_root": root,
        "validation": validation,
        "pairwise": pairwise,
        "table2": table2,
        "output_paths": output_paths,
    }


def main() -> None:
    results = run_analysis()
    print(results["validation"].to_string(index=False))
    print("\nTable 2 (displayed to three decimals):")
    display = results["table2"].copy()
    display["agreement"] = display["agreement"].map(lambda value: f"{value:.3f}")
    display["cohen_kappa"] = display["cohen_kappa"].map(lambda value: f"{value:.3f}")
    print(
        display[
            ["rank_group", "rank", "model_pair", "n_compared", "agreement", "cohen_kappa"]
        ].to_string(index=False)
    )
    for path in results["output_paths"]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
