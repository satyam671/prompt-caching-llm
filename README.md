# Prompt Caching: Cut Your LLM API Spend by 70%

Companion code for the article **"Prompt Caching Is the Most Underrated Cost Optimization in LLM Systems"** published on Medium.

> *I cut my API spend by 70% without changing a single model call. This repo shows you exactly how.*

---

## What's in Here

This repository contains working, runnable implementations of every pattern discussed in the article. Not toy examples — the same structure I use in production document analysis workloads.

```
prompt-caching/
├── examples/
│   ├── 01_baseline_no_caching.py        # The expensive before pattern
│   ├── 02_anthropic_caching.py          # Single cache breakpoint
│   ├── 03_anthropic_multi_breakpoint.py # Two breakpoints, layered stability
│   ├── 04_openai_caching.py             # OpenAI automatic caching
│   ├── 05_conversation_caching.py       # Caching multi-turn conversation history
│   └── 06_cache_invalidation_debug.py   # Diagnosing cache misses
├── utils/
│   ├── cost_calculator.py               # Compute real savings from usage objects
│   └── cache_monitor.py                 # Log and track cache hit rate over time
├── notebooks/
│   └── cost_comparison.ipynb            # Before/after cost analysis with charts
├── diagrams/
│   └── architecture.png                 # Prompt structure diagram from the article
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/satyam671/prompt-caching-llm
cd prompt-caching-llm
pip install -r requirements.txt
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or for OpenAI examples:
export OPENAI_API_KEY="your-key-here"
```

Run the baseline first, then the cached version, and compare:

```bash
python examples/01_baseline_no_caching.py
python examples/02_anthropic_caching.py
```

The output prints token usage and estimated cost side-by-side so you can see the difference immediately.

---

## The Core Pattern (30-second version)

```python
# BEFORE: static content mixed with dynamic, nothing cached
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system="[your full system prompt]",   # recomputed every call
    messages=[{"role": "user", "content": user_query}]
)

# AFTER: static content first, marked for caching
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[
        {"type": "text", "text": STATIC_INSTRUCTIONS},
        {"type": "text", "text": STATIC_CONTEXT, "cache_control": {"type": "ephemeral"}}
    ],
    messages=[{"role": "user", "content": user_query}]  # dynamic stays here
)
```

Cache reads on Anthropic cost 0.1x the normal input price. On a 4,000-token system prompt at 90% hit rate, that's roughly a 72% reduction in input token costs.

---

## Pricing Reference (as of May 2025)

| Provider | Model | Regular Input | Cache Write | Cache Read | Cache TTL |
|----------|-------|---------------|-------------|------------|-----------|
| Anthropic | Claude Sonnet | $3.00/M | $3.75/M | $0.30/M | 5 min |
| Anthropic | Claude Haiku | $0.25/M | $0.30/M | $0.03/M | 5 min |
| OpenAI | GPT-4o | $2.50/M | (automatic) | $1.25/M | 5-10 min |
| OpenAI | GPT-4o mini | $0.15/M | (automatic) | $0.075/M | 5-10 min |

*Always verify against official documentation — pricing changes.*

---

## When Caching Pays Back

| Workload | Static Token % | Expected Savings |
|----------|---------------|-----------------|
| Document analysis tool | 60-80% | 50-70% |
| Customer service bot | 70-90% | 60-80% |
| RAG pipeline (shared KB) | 50-70% | 40-65% |
| Coding assistant | 60-75% | 50-68% |
| Single-use script | <20% | Minimal |

---

## Monitoring Cache Performance

```python
from utils.cache_monitor import CacheMonitor

monitor = CacheMonitor()

# Wrap your existing calls
response = client.messages.create(...)
monitor.record(response.usage)

# Print a summary after N calls
monitor.summary()
# Output:
# Calls recorded: 1000
# Cache hit rate: 91.2%
# Tokens saved from cache: 3,812,400
# Estimated cost saved: $1.14 (at Claude Sonnet pricing)
```

---

## Common Pitfalls (from the article)

1. **Timestamp in the prefix** — any runtime value in your static content busts the cache every call
2. **Session IDs in system prompts** — move these to the user turn
3. **Non-deterministic template assembly** — different whitespace on different code paths = different cache keys
4. **Testing across model versions** — cache keys don't transfer between model strings

See `examples/06_cache_invalidation_debug.py` for a diagnostic script that identifies which part of your prefix is changing.

---

## References

- [Anthropic Prompt Caching Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI Prompt Caching Docs](https://platform.openai.com/docs/guides/prompt-caching)
- Original article on Medium *(link added after publication)*

---

## License

MIT — use it, adapt it, ship it.
