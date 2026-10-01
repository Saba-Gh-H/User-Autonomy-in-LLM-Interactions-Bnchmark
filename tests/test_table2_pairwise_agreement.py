import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "Analyses"
    / "Decision Comparison"
    / "table2_pairwise_agreement.py"
)
SPEC = importlib.util.spec_from_file_location("table2_pairwise_agreement", SCRIPT_PATH)
TABLE2_ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLE2_ANALYSIS)


class Table2PairwiseAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = TABLE2_ANALYSIS.run_analysis(
            repo_root=REPO_ROOT,
            write_outputs=False,
        )

    def test_all_inputs_and_pairwise_comparisons_are_complete(self):
        validation = self.results["validation"]
        self.assertEqual(len(validation), 12)
        self.assertEqual(int(validation["unparsed_answers"].sum()), 0)
        self.assertEqual(len(self.results["pairwise"]), 66)

    def test_table2_top_and_bottom_pairs_match_manuscript(self):
        table = self.results["table2"]
        observed = list(
            table[["model_pair", "n_compared", "agreement", "cohen_kappa"]]
            .itertuples(index=False, name=None)
        )
        expected = [
            ("Llama 3.1 70B -- Llama 3.3 70B", 120, 0.925, 0.8723404255),
            ("Gemma 3 12B -- Gemma 3 27B", 120, 0.8833333333, 0.7469879518),
            (
                "Mistral Small 3.2 24B -- Mistral Small 3.1 24B",
                120,
                0.85,
                0.7275479314,
            ),
            ("Gemma 3 27B -- Qwen2.5 32B", 120, 0.85, 0.6983240223),
            ("GPT-4o -- Qwen2.5 72B", 120, 0.85, 0.6874095514),
            ("Llama 3.2 3B -- Qwen2.5 14B", 120, 0.6416666667, 0.3162846164),
            ("Aya Expanse 32B -- Llama 3.2 3B", 119, 0.6218487395, 0.3544303797),
            ("Gemma 3 27B -- Llama 3.2 3B", 120, 0.65, 0.3647592639),
            ("Gemma 3 12B -- Llama 3.2 3B", 120, 0.6666666667, 0.3864246453),
            ("Llama 3.2 3B -- Qwen2.5 72B", 120, 0.6666666667, 0.3885350318),
        ]

        self.assertEqual([row[:2] for row in observed], [row[:2] for row in expected])
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual[2], target[2], places=9)
            self.assertAlmostEqual(actual[3], target[3], places=9)


if __name__ == "__main__":
    unittest.main()
