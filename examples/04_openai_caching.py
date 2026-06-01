"""
04_openai_caching.py
---------------------
OpenAI's prompt caching is automatic — no cache_control markers needed.
The API caches prompts of 1,024+ tokens automatically and charges
50% of the normal input price on cache hits.

The structural principle is identical: static content first, dynamic content
at the end. The difference is you have no control over breakpoints.
The cache key is the exact byte sequence of the prompt prefix.

Requires: openai >= 1.0.0
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Static system content (put this first, keep it stable) ────────────────────
# Must be 1,024+ tokens total for caching to activate on OpenAI.
# Pad with detail if needed — richer instructions improve output quality anyway.

STATIC_SYSTEM = """
You are a document analysis assistant. Your job is to analyze documents
according to a structured framework and return well-organized findings.

You are precise, concise, and always ground your analysis in the document text.
You do not speculate beyond what the document contains. If the document does not
contain information relevant to a criterion, say so explicitly rather than inferring.

ANALYSIS CRITERIA — apply all five to every document:

1. SUMMARY
   Write 2-3 sentences summarizing the document's main purpose and key content.
   This should be informative enough that someone who hasn't read the document
   understands what it covers.

2. KEY CLAIMS
   Identify the 3-5 most important factual claims or assertions in the document.
   For each claim, assess whether it is:
   - Supported: the document provides evidence or data for this claim
   - Partially supported: some evidence exists but it is incomplete or indirect
   - Unsupported: the claim is made without supporting evidence

3. GAPS
   What important questions does the document raise but not answer?
   What information is conspicuously absent given the document's subject matter?
   Be specific — "lacks methodology detail" is more useful than "could be more thorough."

4. AUDIENCE
   Based on vocabulary, assumed knowledge, and framing, who is this document
   written for? Be specific: "financial analysts" is better than "business readers."

5. RELIABILITY SIGNALS
   Evaluate the document's credibility indicators:
   - Does it cite external sources?
   - Are claims qualified (probably, approximately) or stated as absolute fact?
   - Is there obvious bias in framing or selective presentation?
   - What is the author's apparent relationship to the subject?

OUTPUT FORMAT — use this exact structure:

## Summary
[2-3 sentences]

## Key Claims
- [Claim text] — [Supported / Partially supported / Unsupported]
(list all key claims)

## Gaps
[Specific gaps, as paragraph or bullet list]

## Audience
[1-2 sentences identifying intended readership]

## Reliability Signals
[Assessment of source credibility and framing]

EXAMPLE:

Input: "The company reported Q3 revenues of $2.1B, up 34% YoY.
The CEO attributed growth to enterprise sales. No Q4 guidance was provided."

Output:

## Summary
This corporate earnings announcement reports strong Q3 revenue growth of 34% year-over-year.
The document attributes the performance to enterprise sales but provides no forward guidance.

## Key Claims
- Q3 revenue was $2.1B — Supported
- Revenue grew 34% YoY — Supported
- Growth was driven by enterprise sales — Partially supported (CEO attribution, no data breakdown)

## Gaps
No Q4 guidance, no revenue breakdown by segment, no explanation of the enterprise sales mechanism.

## Audience
Financial analysts and investors familiar with earnings report conventions.

## Reliability Signals
Self-reported corporate data with no independent verification. Claims are stated as facts
without qualification. The omission of Q4 guidance may indicate forward uncertainty
the company is not disclosing.
"""

SAMPLE_DOCUMENTS = [
    "The city council voted 7-2 to approve the new transit expansion, citing projected ridership growth of 40% over five years. Critics argued the cost estimates were based on pre-pandemic data.",
    "Our internal study shows that remote workers report 23% higher job satisfaction. Methodology: survey conducted online, voluntary participation, N=340.",
    "The drug trial was halted after 12 of 200 participants in the treatment group reported elevated liver enzymes. The control group showed no such effects.",
]


def analyze_document_openai(document: str) -> dict:
    """
    OpenAI caches the system prompt automatically when it's 1,024+ tokens.
    No code change required — just keep the static content first and stable.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # Using mini to keep demo costs low
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": STATIC_SYSTEM    # Static content: cached automatically
            },
            {
                "role": "user",
                "content": f"Analyze this document:\n\n{document}"   # Dynamic: not cached
            }
        ]
    )

    usage = response.usage
    cached_tokens = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

    return {
        "output": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": cached_tokens,
    }


def estimate_cost_openai(usage: dict) -> float:
    """GPT-4o-mini pricing (May 2025). Verify against current OpenAI pricing."""
    # Non-cached input: $0.15/M, cached input: $0.075/M, output: $0.60/M
    non_cached_input = usage["prompt_tokens"] - usage["cached_tokens"]
    cost = (
        non_cached_input * 0.15 / 1_000_000
        + usage["cached_tokens"] * 0.075 / 1_000_000
        + usage["completion_tokens"] * 0.60 / 1_000_000
    )
    return cost


if __name__ == "__main__":
    print("=" * 60)
    print("OPENAI: Automatic prompt caching")
    print("No cache_control needed. Static content first = cache hits.")
    print("=" * 60)

    total_cost = 0.0
    total_cached = 0

    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n--- Document {i} ---")
        result = analyze_document_openai(doc)
        cost = estimate_cost_openai(result)
        total_cost += cost
        total_cached += result["cached_tokens"]

        status = "CACHE HIT" if result["cached_tokens"] > 0 else "CACHE MISS"
        print(f"Status:              {status}")
        print(f"Prompt tokens:       {result['prompt_tokens']}")
        print(f"Cached tokens:       {result['cached_tokens']}  (50% off these)")
        print(f"Completion tokens:   {result['completion_tokens']}")
        print(f"Estimated cost:      ${cost:.6f}")

    print(f"\nTOTAL estimated cost: ${total_cost:.6f}")
    print(f"TOTAL cached tokens:  {total_cached}")
    print("\nNote: OpenAI cache TTL is 5-10 minutes. Run calls within this window")
    print("to maintain hit rates in production.")
