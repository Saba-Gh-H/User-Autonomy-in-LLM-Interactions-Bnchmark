# Independent Extraction Robustness

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
- Evaluator model/checkpoint: `openai/gpt-4o`
- Prompt version: `independent_extraction_pilot_v1`
- Decoding: `temperature=0`, `max_tokens=180`
- Response format: strict JSON object
- Conversational history: none; each response is evaluated independently.
- Completed UTC: `2026-07-17T14:48:15.966682+00:00`

## Evaluator Prompt

System prompt:

```text
You are an independent answer-extraction evaluator.

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

```

User prompt template:

```text
Scenario code: {scenario_code}

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

Return strict JSON only.
```

## Outputs

- `independent_extraction_results.csv`
- `independent_extraction_summary.csv`
- `disagreements.csv`
- `table10_independent_evaluator.tex`
- `independent_extraction_metadata.json`

## Result Snapshot

- Evaluated responses: 480
- Agreement: 480 / 480 (100.00%)
- Disagreements: 0
- Letter/rationale conflicts: 0
- UNCLEAR: 0
- Failed or malformed evaluator outputs: 0
- Prompt tokens: 137774
- Completion tokens: 27080
- Estimated cost: $0.615235

## Limitations

- This analysis does not rerun the evaluated LLMs.
- OpenRouter/GPT-4o is an LLM-based evaluator. It is independent from the
  stored response-generation calls, but not model-family-independent for the
  GPT-4o response subset.
- Costs are estimated from usage tokens and listed OpenRouter GPT-4o pricing at
  the time of execution.
