"""
06_cache_invalidation_debug.py
--------------------------------
Diagnostic tool for when your cache hit rate is zero (or lower than expected).

The most common causes of unexpected cache misses:
1. A timestamp, UUID, or session ID is embedded in the static prefix
2. Template assembly is non-deterministic (different whitespace paths)
3. The prompt is being modified at runtime in a way you didn't notice
4. You're testing across different model version strings

This script checks a prompt for common invalidation patterns and then
runs two sequential calls and compares their cache usage to confirm
whether your prefix is actually being cached.

Usage:
    python examples/06_cache_invalidation_debug.py
"""

import os
import re
import time
import hashlib
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ── Pattern checks ─────────────────────────────────────────────────────────────

INVALIDATION_PATTERNS = [
    {
        "name": "Timestamp or datetime",
        "pattern": r"\b\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|datetime\.now|time\.time\(\)",
        "description": "Runtime timestamps change every call and bust the cache every time."
    },
    {
        "name": "UUID or session ID",
        "pattern": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "description": "UUIDs are unique per call. Move these to the user turn."
    },
    {
        "name": "Environment variable interpolation",
        "pattern": r"os\.environ|os\.getenv|\$\{.*?\}",
        "description": "Environment values can differ between processes or deployments."
    },
    {
        "name": "Random value",
        "pattern": r"random\.|uuid\.|secrets\.",
        "description": "Any randomness in the prefix guarantees a cache miss every time."
    },
    {
        "name": "Version string that changes often",
        "pattern": r"v\d+\.\d+\.\d+",
        "description": "If a version number in the prompt increments with deploys, cache is busted on every deploy."
    },
]


def check_prompt_for_invalidation_risks(prompt_text: str) -> list[dict]:
    """
    Scan a prompt string for common patterns that cause cache invalidation.
    Returns a list of detected risks.
    """
    risks = []
    for check in INVALIDATION_PATTERNS:
        if re.search(check["pattern"], prompt_text, re.IGNORECASE):
            risks.append({
                "name": check["name"],
                "description": check["description"]
            })
    return risks


def compute_prefix_hash(prompt_text: str) -> str:
    """Compute a stable hash of the prompt prefix for comparison across calls."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def run_cache_confirmation_test(
    static_content: str,
    dynamic_content: str = "What is the main topic of this content?",
    delay_seconds: float = 1.0
) -> dict:
    """
    Makes two sequential calls with the same static prefix.
    Call 1 should be a cache miss (write). Call 2 should be a cache hit (read).
    If call 2 is also a miss, something in your prefix is changing.
    """

    def make_call(label: str) -> dict:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            system=[
                {
                    "type": "text",
                    "text": static_content,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": dynamic_content}
            ]
        )
        usage = response.usage
        return {
            "label": label,
            "input_tokens": usage.input_tokens,
            "cache_creation": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        }

    call1 = make_call("Call 1 (expect: cache MISS)")
    time.sleep(delay_seconds)   # Small gap to ensure sequential processing
    call2 = make_call("Call 2 (expect: cache HIT)")

    return {"call1": call1, "call2": call2}


# ── Demo prompts (one clean, one with an invalidation problem) ─────────────────

CLEAN_STATIC_PROMPT = """
You are a document analysis assistant. Your role is stable and version-controlled.

Analysis framework version: v2.1 (static — does not change at runtime)

Apply the following criteria: summary, key claims, gaps, audience, reliability signals.
Return findings in structured markdown format.
""" * 10   # Repeat to hit a realistic token count for caching

BROKEN_STATIC_PROMPT_TEMPLATE = """
You are a document analysis assistant.

Session started: {timestamp}
Session ID: {session_id}

Apply the following criteria: summary, key claims, gaps, audience, reliability signals.
Return findings in structured markdown format.
""" * 10   # Same content, but timestamp and session_id will bust the cache


if __name__ == "__main__":
    import uuid
    from datetime import datetime

    print("=" * 60)
    print("CACHE INVALIDATION DEBUGGER")
    print("=" * 60)

    # ── Test 1: Pattern scan on a broken prompt ────────────────────────────────
    print("\n[1] Pattern scan: broken prompt template")
    broken_filled = BROKEN_STATIC_PROMPT_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        session_id=str(uuid.uuid4())
    )

    risks = check_prompt_for_invalidation_risks(broken_filled)
    if risks:
        print(f"  Found {len(risks)} invalidation risk(s):")
        for r in risks:
            print(f"  ❌ {r['name']}: {r['description']}")
    else:
        print("  No risks detected.")

    print("\n[2] Pattern scan: clean prompt")
    clean_risks = check_prompt_for_invalidation_risks(CLEAN_STATIC_PROMPT)
    if clean_risks:
        print(f"  Found {len(clean_risks)} risk(s): {[r['name'] for r in clean_risks]}")
    else:
        print("  ✓ No invalidation patterns detected.")

    # ── Test 2: Live confirmation test (requires API key) ─────────────────────
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[3] Live confirmation test: clean prompt")
        print("  Making two sequential calls. Call 2 should be a cache hit.")
        results = run_cache_confirmation_test(CLEAN_STATIC_PROMPT)

        for call in [results["call1"], results["call2"]]:
            hit = "HIT" if call["cache_read"] > 0 else "MISS"
            print(f"  {call['label']}: {hit}")
            print(f"    cache_creation: {call['cache_creation']}  |  cache_read: {call['cache_read']}")

        if results["call2"]["cache_read"] > 0:
            print("\n  ✓ Cache is working. Prefix is stable.")
        else:
            print("\n  ❌ Cache miss on call 2. Prefix is not stable.")
            print("  Check: is the static_content string built deterministically?")
            print("  Compute the hash of your prefix before and after each call")
            print("  to find where it's changing:")
            h1 = compute_prefix_hash(CLEAN_STATIC_PROMPT)
            print(f"  Prefix hash: {h1}")
    else:
        print("\n[3] Skipping live test — ANTHROPIC_API_KEY not set.")
        print("  Set the key and re-run to confirm cache behaviour with real API calls.")

    print("\n" + "=" * 60)
    print("SUMMARY: Common fixes for cache misses")
    print("  - Remove timestamps, UUIDs, session IDs from static content")
    print("  - Move any runtime-variable values to the user turn")
    print("  - Version-control your prompt as a file, not a template string")
    print("  - Pin the model version string — cache doesn't transfer across models")
    print("=" * 60)
