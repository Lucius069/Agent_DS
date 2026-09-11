"""
insight_generator.py — Step 2: LLM Insight Layer

Takes the structured stats profile from profiler.py and turns it into
prioritized, natural-language insights using an LLM call.

Design principle: the LLM never sees raw data or does arithmetic. It only
reasons over the JSON profile we already computed exactly with pandas/numpy.
This keeps the numbers trustworthy and uses the LLM for what it's actually
good at — prioritizing and explaining, not calculating.
"""

import json
import os
from anthropic import Anthropic


# The prompt is the most important part of this file. Read the comments below —
# they explain *why* each instruction is there, not just what it says.
SYSTEM_PROMPT = """You are a senior data analyst reviewing a statistical profile \
of a dataset. You will be given exact, pre-computed statistics as JSON — never \
question or recompute these numbers, they are ground truth.

Your job: identify the 3-5 MOST IMPORTANT things a data scientist should know \
before working with this dataset. Prioritize by how much each fact would change \
an analysis or modeling decision — not by how interesting the number looks.

Rules:
- Never just restate a number ("the mean is 45.2" is USELESS on its own).
  Instead, interpret it: what does it imply, what should the analyst do about it.
- Prefer fewer, higher-value insights over many shallow ones.
- If something looks like a data quality problem (extreme outliers, high missingness,
  suspicious duplicates), say so plainly and suggest a concrete next step.
- Respond ONLY with valid JSON matching this exact schema, no other text:

{
  "insights": [
    {
      "finding": "one sentence describing what you found",
      "why_it_matters": "one sentence on the practical implication",
      "suggested_action": "one concrete next step the analyst should take",
      "confidence": "high | medium | low"
    }
  ]
}
"""


def generate_insights(profile: dict, model: str = "claude-sonnet-4-6") -> dict:
    """
    Takes a profile dict (from profiler.profile_dataset) and returns a dict:
    {"insights": [...]}  — same shape the LLM was asked to produce.

    Requires the ANTHROPIC_API_KEY environment variable to be set.
    """
    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically

    user_message = (
        "Here is the statistical profile of a dataset:\n\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        "Produce your insights now, following the schema exactly."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text

    return _parse_llm_json(raw_text)


def _parse_llm_json(raw_text: str) -> dict:
    """
    LLMs occasionally wrap JSON in markdown code fences even when told not to.
    This strips those defensively so parsing doesn't break the whole pipeline.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fail loudly with the raw text visible — silent failures are the
        # hardest bugs to debug in an agent pipeline.
        raise ValueError(
            f"LLM did not return valid JSON. Raw response:\n{raw_text}"
        ) from e


if __name__ == "__main__":
    # End-to-end manual test: profile a sample dataset, then generate insights.
    import pandas as pd
    import numpy as np
    from profiler import profile_dataset

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set the ANTHROPIC_API_KEY environment variable before running this.")
        raise SystemExit(1)

    sample_df = pd.DataFrame({
        "age": [25, 32, 47, 29, np.nan, 200, 31, 28, 45, 33],
        "salary": [50000, 62000, 85000, 58000, 60000, 61000, 59000, 300000, 82000, 63000],
        "department": ["Sales", "Eng", "Eng", "Sales", "HR", "Eng", "Sales", "Eng", "HR", "Eng"],
    })

    profile = profile_dataset(sample_df)
    insights = generate_insights(profile)

    print(json.dumps(insights, indent=2))
