"""
02_anthropic_caching.py
-----------------------
The "after" pattern. Static content is placed first in the system array
and marked with cache_control. Dynamic content stays in the user turn.

On the first call, the static prefix is written to cache (1.25x price).
On subsequent calls within the 5-minute TTL, it's read from cache (0.10x price).

Compare this output to 01_baseline_no_caching.py.
"""

import os
import time
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Static content blocks (ordered: most stable → least stable) ───────────────

ROLE_INSTRUCTIONS = """
You are a document analysis assistant. Your job is to analyze documents
according to a structured framework and return well-organized findings.

You are precise, concise, and always ground your analysis in the document text.
You do not speculate beyond what the document contains. If the document does not
contain information relevant to a criterion, say so explicitly rather than inferring.
"""

ANALYSIS_FRAMEWORK = """
Apply the following analysis criteria to every document:

1. SUMMARY — A 2-3 sentence summary of the document's main purpose and content.

2. KEY CLAIMS — List the 3-5 most important factual claims or assertions made.
   For each claim, note whether it is supported by evidence in the document.

3. GAPS — What questions does the document raise but not answer?
   What information is notably absent?

4. AUDIENCE — Who is this document written for, based on its language and assumptions?

5. RELIABILITY SIGNALS — Does the document cite sources? Are claims qualified or absolute?
   Are there obvious biases in framing?
"""

OUTPUT_FORMAT = """
Return your analysis in this exact structure:

## Summary
[2-3 sentences]

## Key Claims
- [Claim 1] — [Supported / Unsupported / Partially supported]
- [Claim 2] — [Supported / Unsupported / Partially supported]
(continue for all key claims)

## Gaps
[Paragraph or bullet list of missing information]

## Audience
[1-2 sentences on intended readership]

## Reliability Signals
[Brief assessment of source quality and framing]
"""

FEW_SHOT_EXAMPLES = """
Here is an example of a correctly formatted analysis:

INPUT DOCUMENT:
"The company reported record Q3 revenues of $2.1B, up 34% year-over-year.
CEO Jane Smith attributed the growth to expanded enterprise sales. The company
did not provide Q4 guidance."

CORRECT OUTPUT:

## Summary
This is a brief corporate earnings announcement reporting strong quarterly revenue growth.
The document attributes the performance to enterprise sales expansion without providing
forward-looking projections.

## Key Claims
- Q3 revenue was $2.1B — Supported (stated as reported figure)
- Revenue grew 34% YoY — Supported (stated explicitly)
- Growth driven by enterprise sales — Partially supported (attributed to CEO, no breakdown given)

## Gaps
The document provides no Q4 guidance, no breakdown of revenue by segment,
and no explanation of how enterprise sales specifically drove the growth.

## Audience
Financial analysts, investors, and business journalists familiar with corporate
earnings terminology.

## Reliability Signals
The document is a corporate announcement with no independent citation.
Claims are presented as facts without qualification. The absence of Q4
guidance may indicate uncertainty the company is not disclosing.
"""

# ── Sample documents (dynamic — different per call) ───────────────────────────

SAMPLE_DOCUMENTS = [
    "The city council voted 7-2 to approve the new transit expansion, citing projected ridership growth of 40% over five years. Critics argued the cost estimates were based on pre-pandemic data.",
    "Our internal study shows that remote workers report 23% higher job satisfaction. Methodology note: survey conducted online, voluntary participation, N=340.",
    "The drug trial was halted after 12 of 200 participants in the treatment group reported elevated liver enzymes. The control group showed no such effects.",
]


def analyze_document_with_cache(document: str) -> dict:
    """
    Makes a single API call WITH prompt caching enabled.

    Key structural decisions:
    - System prompt is split into content blocks, not a single string
    - The final system block carries cache_control — this caches everything before it
    - Dynamic content (the actual document) stays in the user turn, never in system
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=[
            # Block 1: Role instructions (most stable — rarely changes)
            {
                "type": "text",
                "text": ROLE_INSTRUCTIONS
            },
            # Block 2: Framework + format + examples (stable but might update monthly)
            # cache_control here means: cache blocks 1 AND 2 together as the prefix
            {
                "type": "text",
                "text": ANALYSIS_FRAMEWORK + OUTPUT_FORMAT + FEW_SHOT_EXAMPLES,
                "cache_control": {"type": "ephemeral"}   # <-- this is the only required change
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Please analyze the following document:\n\n{document}"
                # ↑ Dynamic content goes HERE — not in the system blocks
            }
        ]
    )

    return {
        "output": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }


def estimate_cost(usage: dict, model: str = "haiku") -> float:
    """Cost estimate using Anthropic's Haiku pricing (May 2025)."""
    prices = {
        "haiku": {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03}
    }
    p = prices.get(model, prices["haiku"])
    cost = (
        usage["input_tokens"] * p["input"] / 1_000_000
        + usage["output_tokens"] * p["output"] / 1_000_000
        + usage["cache_creation_tokens"] * p["cache_write"] / 1_000_000
        + usage["cache_read_tokens"] * p["cache_read"] / 1_000_000
    )
    return cost


if __name__ == "__main__":
    print("=" * 60)
    print("WITH CACHING: Anthropic prompt caching enabled")
    print("Call 1: cache miss (prefix written). Calls 2+: cache hits.")
    print("=" * 60)

    total_input = 0
    total_cache_write = 0
    total_cache_read = 0
    total_cost = 0.0

    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n--- Document {i} ---")
        result = analyze_document_with_cache(doc)

        cost = estimate_cost(result)
        total_input += result["input_tokens"]
        total_cache_write += result["cache_creation_tokens"]
        total_cache_read += result["cache_read_tokens"]
        total_cost += cost

        hit_or_miss = "CACHE HIT" if result["cache_read_tokens"] > 0 else "CACHE MISS (write)"
        print(f"Status:              {hit_or_miss}")
        print(f"Input tokens:        {result['input_tokens']}")
        print(f"Cache creation:      {result['cache_creation_tokens']}")
        print(f"Cache reads:         {result['cache_read_tokens']}")
        print(f"Output tokens:       {result['output_tokens']}")
        print(f"Estimated cost:      ${cost:.6f}")

        # Small delay between calls is not required but mimics a realistic call pattern
        if i < len(SAMPLE_DOCUMENTS):
            time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"TOTAL input tokens:         {total_input}")
    print(f"TOTAL cache write tokens:   {total_cache_write}")
    print(f"TOTAL cache read tokens:    {total_cache_read}")
    print(f"TOTAL estimated cost:       ${total_cost:.6f}")
    print("=" * 60)
    print("\nCompare this cost to the output of 01_baseline_no_caching.py")
    print("The difference is the static prefix recomputation cost you were paying.")
