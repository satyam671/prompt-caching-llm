"""
utils/cost_calculator.py
-------------------------
Compute exact costs from Anthropic and OpenAI usage objects.
Includes a savings report comparing cached vs uncached cost.

Usage:
    from utils.cost_calculator import CostCalculator

    calc = CostCalculator(provider="anthropic", model="haiku")
    cost = calc.compute(usage_dict)
    report = calc.savings_report(usage_dict)
"""

from dataclasses import dataclass


# ── Pricing tables ─────────────────────────────────────────────────────────────
# Prices per 1M tokens. Verify against official docs — pricing changes.
# Sources:
#   Anthropic: docs.anthropic.com/en/docs/about-claude/models
#   OpenAI:    platform.openai.com/docs/models

ANTHROPIC_PRICING = {
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,   # 1.25x input
        "cache_read": 0.30,    # 0.10x input
    },
    "haiku": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
    },
    "opus": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
}

OPENAI_PRICING = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "cache_read": 1.25,   # 50% off
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.075,
    },
    "o3": {
        "input": 10.00,
        "output": 40.00,
        "cache_read": 2.50,
    },
}


@dataclass
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class CostCalculator:
    """
    Compute exact API costs from usage objects returned by the API.

    Example:
        calc = CostCalculator(provider="anthropic", model="sonnet")
        cost = calc.compute({
            "input_tokens": 150,
            "output_tokens": 800,
            "cache_creation_tokens": 4000,
            "cache_read_tokens": 0,
        })
    """

    def __init__(self, provider: str = "anthropic", model: str = "sonnet"):
        self.provider = provider.lower()
        self.model = model.lower()

        if self.provider == "anthropic":
            if self.model not in ANTHROPIC_PRICING:
                raise ValueError(f"Unknown Anthropic model: {model}. Options: {list(ANTHROPIC_PRICING.keys())}")
            self.prices = ANTHROPIC_PRICING[self.model]
        elif self.provider == "openai":
            if self.model not in OPENAI_PRICING:
                raise ValueError(f"Unknown OpenAI model: {model}. Options: {list(OPENAI_PRICING.keys())}")
            self.prices = OPENAI_PRICING[self.model]
        else:
            raise ValueError(f"Unknown provider: {provider}. Options: anthropic, openai")

    def compute(self, usage: dict) -> float:
        """Return exact cost in USD from a usage dict."""
        p = self.prices
        if self.provider == "anthropic":
            return (
                usage.get("input_tokens", 0) * p["input"] / 1_000_000
                + usage.get("output_tokens", 0) * p["output"] / 1_000_000
                + usage.get("cache_creation_tokens", 0) * p["cache_write"] / 1_000_000
                + usage.get("cache_read_tokens", 0) * p["cache_read"] / 1_000_000
            )
        else:  # openai
            non_cached = usage.get("input_tokens", 0) - usage.get("cache_read_tokens", 0)
            return (
                non_cached * p["input"] / 1_000_000
                + usage.get("cache_read_tokens", 0) * p["cache_read"] / 1_000_000
                + usage.get("output_tokens", 0) * p["output"] / 1_000_000
            )

    def hypothetical_no_cache_cost(self, usage: dict) -> float:
        """What this call would have cost with no caching at all."""
        total_input = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_tokens", 0)
            + usage.get("cache_read_tokens", 0)
        )
        return (
            total_input * self.prices["input"] / 1_000_000
            + usage.get("output_tokens", 0) * self.prices["output"] / 1_000_000
        )

    def savings_report(self, usage: dict) -> str:
        """Return a formatted savings report string."""
        actual = self.compute(usage)
        without_cache = self.hypothetical_no_cache_cost(usage)
        saved = without_cache - actual
        pct = (saved / without_cache * 100) if without_cache > 0 else 0

        cache_read = usage.get("cache_read_tokens", 0)
        cache_write = usage.get("cache_creation_tokens", 0)
        status = "HIT" if cache_read > 0 else ("WRITE" if cache_write > 0 else "NO CACHE")

        return (
            f"Cache status:       {status}\n"
            f"Actual cost:        ${actual:.6f}\n"
            f"Without caching:    ${without_cache:.6f}\n"
            f"Saved:              ${saved:.6f}  ({pct:.1f}%)"
        )


def aggregate_savings(usage_records: list[dict], provider: str, model: str) -> dict:
    """
    Aggregate savings across multiple calls.
    Returns total actual cost, total hypothetical cost, and total saved.

    Args:
        usage_records: list of usage dicts from multiple API calls
        provider: "anthropic" or "openai"
        model: model name string

    Returns:
        dict with total_actual, total_without_cache, total_saved, savings_pct, call_count
    """
    calc = CostCalculator(provider=provider, model=model)
    total_actual = 0.0
    total_without = 0.0

    for record in usage_records:
        total_actual += calc.compute(record)
        total_without += calc.hypothetical_no_cache_cost(record)

    saved = total_without - total_actual
    pct = (saved / total_without * 100) if total_without > 0 else 0

    return {
        "call_count": len(usage_records),
        "total_actual": total_actual,
        "total_without_cache": total_without,
        "total_saved": saved,
        "savings_pct": pct,
    }


if __name__ == "__main__":
    # Example: compare single-call savings
    example_usage = {
        "input_tokens": 150,
        "output_tokens": 800,
        "cache_creation_tokens": 4000,
        "cache_read_tokens": 0,
    }

    example_hit = {
        "input_tokens": 150,
        "output_tokens": 800,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 4000,
    }

    calc = CostCalculator(provider="anthropic", model="sonnet")

    print("=== Cache MISS (first call, prefix written) ===")
    print(calc.savings_report(example_usage))

    print("\n=== Cache HIT (subsequent calls) ===")
    print(calc.savings_report(example_hit))

    # Aggregate: 1 write + 9 hits (realistic 10-call batch)
    print("\n=== Aggregate: 1 write + 9 hits (10-call batch) ===")
    records = [example_usage] + [example_hit] * 9
    summary = aggregate_savings(records, provider="anthropic", model="sonnet")
    print(f"Calls:              {summary['call_count']}")
    print(f"Total actual:       ${summary['total_actual']:.4f}")
    print(f"Without caching:    ${summary['total_without_cache']:.4f}")
    print(f"Total saved:        ${summary['total_saved']:.4f}  ({summary['savings_pct']:.1f}%)")
