"""
Tests for insight_generator.py.

We deliberately do NOT call the real Anthropic API in these tests — that would
cost money, be slow, and be non-deterministic. Instead we test the JSON-parsing
logic directly, since that's the part most likely to break silently in production
(LLMs sometimes wrap output in markdown fences despite instructions not to).

Run with: python -m pytest tests/test_insight_generator.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
from insight_generator import _parse_llm_json


def test_parses_clean_json():
    raw = '{"insights": [{"finding": "test", "why_it_matters": "x", "suggested_action": "y", "confidence": "high"}]}'
    result = _parse_llm_json(raw)
    assert result["insights"][0]["finding"] == "test"


def test_strips_markdown_json_fence():
    raw = '```json\n{"insights": []}\n```'
    result = _parse_llm_json(raw)
    assert result == {"insights": []}


def test_strips_plain_markdown_fence():
    raw = '```\n{"insights": []}\n```'
    result = _parse_llm_json(raw)
    assert result == {"insights": []}


def test_raises_clear_error_on_invalid_json():
    raw = "Sorry, I can't help with that."
    with pytest.raises(ValueError, match="did not return valid JSON"):
        _parse_llm_json(raw)


def test_handles_leading_trailing_whitespace():
    raw = '   \n  {"insights": []}  \n  '
    result = _parse_llm_json(raw)
    assert result == {"insights": []}
