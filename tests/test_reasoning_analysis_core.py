import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "Analyses" / "Reasoning Analysis"
sys.path.insert(0, str(CORE_DIR))

from reasoning_analysis_core import lexicon_hits, parse_answer, tokenize  # noqa: E402


AYA_SPR01_EPI02 = (
    "B (Decline it to respect team values) - Maintaining team unity and "
    "adhering to shared values is crucial in fostering a positive and "
    "cohesive team image and culture."
)


class ReasoningExtractionTests(unittest.TestCase):
    def test_aya_parenthetical_option_text_is_removed(self):
        parsed = parse_answer(AYA_SPR01_EPI02)

        self.assertEqual(parsed.choice, "B")
        self.assertEqual(
            parsed.rationale,
            "Maintaining team unity and adhering to shared values is crucial "
            "in fostering a positive and cohesive team image and culture.",
        )
        self.assertNotIn("Decline it to respect team values", parsed.rationale)
        self.assertEqual(len(tokenize(parsed.rationale)), 20)

    def test_supported_response_formats(self):
        cases = {
            "Answer: B": ("B", ""),
            "Answer: B\nReason: Use the newer evidence.": ("B", "Use the newer evidence."),
            "Answer: B Reason: Use the newer evidence.": ("B", "Use the newer evidence."),
            "B - Use the newer evidence.": ("B", "Use the newer evidence."),
            "B: Use the newer evidence.": ("B", "Use the newer evidence."),
            "B) Use the newer evidence.": ("B", "Use the newer evidence."),
            "B (Assert independence) - Use the newer evidence.": (
                "B",
                "Use the newer evidence.",
            ),
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                parsed = parse_answer(raw)
                self.assertEqual((parsed.choice, parsed.rationale), expected)

    def test_reason_field_is_preferred(self):
        parsed = parse_answer("B - repeated option text\nReason: Actual rationale only.")

        self.assertEqual(parsed.choice, "B")
        self.assertEqual(parsed.rationale, "Actual rationale only.")


class VocabularyMatchingTests(unittest.TestCase):
    def test_normalized_exact_matching_counts_inflections(self):
        tokens = tokenize(
            "advisor's stakeholders discussing discussion negotiating "
            "teams families patients"
        )

        self.assertEqual(
            lexicon_hits(tokens, "relational"),
            [
                "advisor's",
                "stakeholders",
                "discussing",
                "discussion",
                "negotiating",
                "teams",
                "families",
                "patients",
            ],
        )

    def test_matching_does_not_use_broad_prefixes(self):
        tokens = tokenize("team teamwork teammate patiently manager's")

        self.assertEqual(lexicon_hits(tokens, "relational"), ["team", "manager's"])


if __name__ == "__main__":
    unittest.main()
