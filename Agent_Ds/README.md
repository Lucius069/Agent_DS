# Self-Learning Data Analysis Agent

An AI agent that analyzes datasets, generates insights, critiques its own output,
and remembers what worked — improving its analysis strategy over time.

## Progress

- [x] **Step 1: Static Analysis MVP** (`agent/profiler.py`) — pure pandas/numpy/scipy
      EDA profiler. No AI yet. Outputs a structured JSON profile: dtypes, missing
      values, duplicates, numeric stats + skew + outliers, categorical cardinality,
      and notable correlations.
- [ ] Step 2: LLM insight layer
- [ ] Step 3: Self-critique / scoring
- [ ] Step 4: Persistent memory (ChromaDB)
- [ ] Step 5: Metrics dashboard
- [ ] Step 6: Streamlit UI
- [ ] Step 7: Deploy to Streamlit Community Cloud
- [ ] Step 8: Write-up

## Setup

```bash
pip install -r requirements.txt
```

## Usage (current)

```bash
python agent/profiler.py        # runs on a built-in sample dataset
python -m pytest tests/ -v      # run the test suite
```

## Why this project

Most "AI agent" portfolio projects are thin wrappers around an LLM API call.
This one is different: it has a genuine self-improvement loop. The agent scores
its own output quality, stores what worked in persistent memory, and retrieves
relevant past lessons before analyzing a new dataset — producing a measurable
quality-over-time trend, which is the core artifact of the whole project.
