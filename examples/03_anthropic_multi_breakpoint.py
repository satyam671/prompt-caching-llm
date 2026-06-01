"""
03_anthropic_multi_breakpoint.py
---------------------------------
Two cache breakpoints for systems where different parts of the static
content have different update frequencies.

Use case: Your core role instructions never change (breakpoint 1),
but your analysis framework and examples update monthly (breakpoint 2).

With two breakpoints, a framework update only invalidates breakpoint 2.
Breakpoint 1 stays warm. You still save on the stable core.

Anthropic supports up to 4 cache breakpoints per request.
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Tier 1: Never changes (or changes quarterly at most) ─────────────────────

CORE_ROLE = """
You are a document analysis assistant. Your job is to analyze documents
according to a structured framework and return well-organized findings.

You are precise, concise, and always ground your analysis in the document text.
You do not speculate beyond what the document contains. If the document does not
contain information relevant to a criterion, say so explicitly rather than inferring.

This role definition is version-controlled and does not change between sessions.
"""

# ── Tier 2: Changes occasionally (monthly updates, new examples added) ────────

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
(continue for all key claims)

## Gaps
[Paragraph or bullet list]

## Audience
[1-2 sentences]

## Reliability Signals
[Brief assessment]
"""

FEW_SHOT_EXAMPLES = """
EXAMPLE:

INPUT: "The company reported Q3 revenues of $2.1B, up 34% YoY.
CEO attributed growth to enterprise sales. No Q4 guidance provided."

OUTPUT:
## Summary
Corporate earnings announcement with strong revenue growth. No forward guidance given.

## Key Claims
- Q3 revenue $2.1B — Supported
- 34% YoY growth — Supported
- Growth from enterprise sales — Partially supported (attribution only)

## Gaps
No Q4 guidance, no segment breakdown, no explanation of enterprise sales driver.

## Audience
Financial analysts and investors.

## Reliability Signals
Corporate self-report, no independent citation, claims unqualified.
"""

SAMPLE_DOCUMENTS = [
    "The city council voted 7-2 to approve the new transit expansion, citing projected ridership growth of 40% over five years. Critics argued the cost estimates were based on pre-pandemic data.",
    "Our internal study shows that remote workers report 23% higher job satisfaction. Methodology: survey conducted online, voluntary participation, N=340.",
    "The drug trial was halted after 12 of 200 participants in the treatment group reported elevated liver enzymes. Control group showed no such effects.",
]


def analyze_document_two_breakpoints(document: str) -> dict:
    """
    Two cache_control markers create two cache breakpoints.

    Breakpoint 1 caches CORE_ROLE.
    Breakpoint 2 caches CORE_ROLE + ANALYSIS_FRAMEWORK + OUTPUT_FORMAT + FEW_SHOT_EXAMPLES.

    If the framework updates, only breakpoint 2 is invalidated.
    The stable core at breakpoint 1 keeps saving.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=[
            # Tier 1 block — most stable, first breakpoint
            {
                "type": "text",
                "text": CORE_ROLE,
                "cache_control": {"type": "ephemeral"}   # Breakpoint 1
            },
            # Tier 2 block — changes occasionally, second breakpoint
            {
                "type": "text",
                "text": ANALYSIS_FRAMEWORK + OUTPUT_FORMAT + FEW_SHOT_EXAMPLES,
                "cache_control": {"type": "ephemeral"}   # Breakpoint 2
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Analyze this document:\n\n{document}"
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


def estimate_cost(usage: dict) -> float:
    prices = {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03}
    return (
        usage["input_tokens"] * prices["input"] / 1_000_000
        + usage["output_tokens"] * prices["output"] / 1_000_000
        + usage["cache_creation_tokens"] * prices["cache_write"] / 1_000_000
        + usage["cache_read_tokens"] * prices["cache_read"] / 1_000_000
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TWO BREAKPOINTS: Layered stability")
    print("Tier 1 (role): most stable — survives framework updates")
    print("Tier 2 (framework + examples): updated monthly")
    print("=" * 60)

    total_cost = 0.0

    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n--- Document {i} ---")
        result = analyze_document_two_breakpoints(doc)
        cost = estimate_cost(result)
        total_cost += cost

        status = "CACHE HIT" if result["cache_read_tokens"] > 0 else "CACHE MISS"
        print(f"Status:              {status}")
        print(f"Input tokens:        {result['input_tokens']}")
        print(f"Cache creation:      {result['cache_creation_tokens']}")
        print(f"Cache reads:         {result['cache_read_tokens']}")
        print(f"Estimated cost:      ${cost:.6f}")

    print(f"\nTOTAL estimated cost: ${total_cost:.6f}")
    print("\nWhen the framework updates, only re-run calls after the update.")
    print("Breakpoint 1 (core role) stays warm and keeps saving.")
