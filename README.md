# 🧠 Self-Learning Data Analysis Agent

An AI agent that analyzes tabular datasets, generates natural-language insights,
**critiques its own output**, and stores what worked in persistent memory —
so it gets measurably better at analysis with every dataset it sees.

> Most "AI agent" projects are a thin wrapper around an LLM call. This one has
> a real feedback loop: **Act → Self-critique → Store lesson → Retrieve lesson
> next time** — and it tracks its own improvement over time as a chart, not
> just a claim.

🔗 **Live demo:** _coming soon_
📊 **Portfolio write-up:** _coming soon_

---

## Why this project exists

I'm a Computer Engineering student aiming for data science / ML roles. Instead
of another notebook full of `df.describe()`, I wanted a project that shows:

- solid EDA/statistics fundamentals (not just prompt engineering)
- an understanding of agentic loops and memory, not just single API calls
- an actual measurable "learning" signal — a metric that improves over time
- something deployed and clickable, not just code sitting in a repo

## How it works

```
 ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
 │  Upload CSV │ --> │   Profiler   │ --> │ LLM Insight    │
 │             │     │ (pandas/numpy│     │ Generator      │
 └─────────────┘     │  stats only) │     └───────┬───────┘
                      └──────────────┘             │
                                                    v
 ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
 │   Metrics   │ <-- │   Memory     │ <-- │ Self-Critique  │
 │  Dashboard  │     │  (ChromaDB)  │     │  (LLM judge)   │
 └─────────────┘     └──────────────┘     └───────────────┘
```

1. **Profiler** — pure pandas/numpy/scipy pass over the dataset: dtypes, missing
   values, duplicates, distribution shape, outliers (IQR method), and notable
   correlations. No AI involved yet — this is the statistical ground truth.
2. **Insight Generator** — an LLM turns the raw stats profile into plain-English
   insights a human analyst would actually say out loud.
3. **Self-Critique** — a second LLM pass scores each insight for relevance,
   correctness, and actionability, with a justification.
4. **Memory** — every (dataset fingerprint → strategy → score) triple is stored
   in a vector DB. Before analyzing a new dataset, the agent retrieves similar
   past attempts and adjusts its approach.
5. **Metrics Dashboard** — logs every run and plots insight-quality over time —
   the core proof-of-learning artifact of the whole project.

## Project status

| Step | Status |
|---|---|
| Static analysis / EDA profiler | ✅ Done |
| LLM insight generation layer | 🔲 Planned |
| Self-critique / scoring | 🔲 Planned |
| Persistent memory (ChromaDB) | 🔲 Planned |
| Metrics dashboard | 🔲 Planned |
| Streamlit UI | 🔲 Planned |
| Deployment (Streamlit Community Cloud) | 🔲 Planned |

## Tech stack

- **Analysis:** Python, Pandas, NumPy, SciPy
- **Agent reasoning:** LLM API (Claude/OpenAI)
- **Memory:** ChromaDB
- **Metrics storage:** SQLite
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Testing:** Pytest

## Getting started

```bash
git clone https://github.com/<your-username>/selflearning-agent.git
cd selflearning-agent
pip install -r requirements.txt
```

Run the profiler on the built-in sample dataset:

```bash
python agent/profiler.py
```

Run it on your own CSV (from a Python shell or script):

```python
import pandas as pd
from agent.profiler import profile_dataset

df = pd.read_csv("your_dataset.csv")
profile = profile_dataset(df)
print(profile)
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Project structure

```
selflearning-agent/
├── agent/
│   └── profiler.py       # static EDA analysis (no AI) — done
├── tests/
│   └── test_profiler.py  # unit tests for the profiler
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Wire in an LLM to turn stats into narrative insights
- [ ] Add a self-scoring rubric (relevance / correctness / actionability)
- [ ] Add ChromaDB memory so lessons persist across datasets
- [ ] Build the quality-over-time metrics dashboard
- [ ] Ship a Streamlit UI
- [ ] Deploy publicly and link the live demo here

## License

MIT
