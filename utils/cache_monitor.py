"""
utils/cache_monitor.py
-----------------------
Production-grade cache hit rate monitoring.

Wrap your existing API calls with CacheMonitor.record() to track
hit rates, savings, and trends over time. Prints periodic summaries
and alerts when hit rate drops below a threshold.

Usage:
    from utils.cache_monitor import CacheMonitor

    monitor = CacheMonitor(provider="anthropic", model="sonnet", alert_threshold=0.7)

    # In your API call loop:
    response = client.messages.create(...)
    monitor.record(response.usage)

    # Print a summary at any point:
    monitor.summary()
"""

import time
from collections import deque
from dataclasses import dataclass, field
from utils.cost_calculator import CostCalculator


@dataclass
class CallRecord:
    timestamp: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    actual_cost: float
    hypothetical_cost: float

    @property
    def was_cache_hit(self) -> bool:
        return self.cache_read_tokens > 0

    @property
    def saved(self) -> float:
        return self.hypothetical_cost - self.actual_cost


class CacheMonitor:
    """
    Tracks prompt cache performance across API calls.

    Args:
        provider: "anthropic" or "openai"
        model: model name string (e.g., "sonnet", "haiku", "gpt-4o")
        alert_threshold: log a warning when hit rate drops below this (0.0 to 1.0)
        rolling_window: number of recent calls to use for rolling hit rate
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "sonnet",
        alert_threshold: float = 0.7,
        rolling_window: int = 100,
    ):
        self.calc = CostCalculator(provider=provider, model=model)
        self.alert_threshold = alert_threshold
        self.records: list[CallRecord] = []
        self.recent: deque = deque(maxlen=rolling_window)

    def record(self, usage) -> CallRecord:
        """
        Record a usage object from an API response.
        Accepts both Anthropic and OpenAI usage objects, or a plain dict.
        """
        # Normalize to dict regardless of input type
        if hasattr(usage, "__dict__"):
            raw = usage.__dict__
        elif isinstance(usage, dict):
            raw = usage
        else:
            raw = {}

        # Handle OpenAI's nested prompt_tokens_details structure
        cache_read = raw.get("cache_read_input_tokens", 0)
        if cache_read == 0 and hasattr(usage, "prompt_tokens_details"):
            ptd = usage.prompt_tokens_details
            if ptd:
                cache_read = getattr(ptd, "cached_tokens", 0)

        usage_dict = {
            "input_tokens": raw.get("input_tokens", raw.get("prompt_tokens", 0)),
            "output_tokens": raw.get("output_tokens", raw.get("completion_tokens", 0)),
            "cache_creation_tokens": raw.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": cache_read,
        }

        actual = self.calc.compute(usage_dict)
        hypothetical = self.calc.hypothetical_no_cache_cost(usage_dict)

        record = CallRecord(
            timestamp=time.time(),
            input_tokens=usage_dict["input_tokens"],
            output_tokens=usage_dict["output_tokens"],
            cache_creation_tokens=usage_dict["cache_creation_tokens"],
            cache_read_tokens=usage_dict["cache_read_tokens"],
            actual_cost=actual,
            hypothetical_cost=hypothetical,
        )

        self.records.append(record)
        self.recent.append(record)

        # Alert if rolling hit rate drops below threshold
        rolling_hit_rate = self._rolling_hit_rate()
        if len(self.recent) >= 10 and rolling_hit_rate < self.alert_threshold:
            print(
                f"[CacheMonitor ALERT] Rolling hit rate dropped to "
                f"{rolling_hit_rate:.1%} (threshold: {self.alert_threshold:.1%}). "
                f"Check prompt prefix stability."
            )

        return record

    def _rolling_hit_rate(self) -> float:
        if not self.recent:
            return 0.0
        return sum(1 for r in self.recent if r.was_cache_hit) / len(self.recent)

    def summary(self, last_n: int = None) -> str:
        """Print a formatted summary of cache performance."""
        records = self.records[-last_n:] if last_n else self.records

        if not records:
            print("No calls recorded yet.")
            return

        total_calls = len(records)
        cache_hits = sum(1 for r in records if r.was_cache_hit)
        cache_writes = sum(1 for r in records if r.cache_creation_tokens > 0)
        no_cache = total_calls - cache_hits - cache_writes

        total_actual = sum(r.actual_cost for r in records)
        total_hypothetical = sum(r.hypothetical_cost for r in records)
        total_saved = total_hypothetical - total_actual
        savings_pct = (total_saved / total_hypothetical * 100) if total_hypothetical > 0 else 0

        total_input = sum(r.input_tokens for r in records)
        total_cache_read = sum(r.cache_read_tokens for r in records)
        total_cache_write = sum(r.cache_creation_tokens for r in records)

        separator = "=" * 50
        print(f"\n{separator}")
        print("CACHE MONITOR SUMMARY")
        print(separator)
        print(f"Calls recorded:       {total_calls}")
        print(f"Cache hits:           {cache_hits}  ({cache_hits/total_calls:.1%})")
        print(f"Cache writes:         {cache_writes}")
        print(f"No cache activity:    {no_cache}")
        print(f"Rolling hit rate:     {self._rolling_hit_rate():.1%} (last {len(self.recent)} calls)")
        print(separator)
        print(f"Total input tokens:   {total_input:,}")
        print(f"Total cache writes:   {total_cache_write:,}")
        print(f"Total cache reads:    {total_cache_read:,}")
        print(separator)
        print(f"Total actual cost:    ${total_actual:.4f}")
        print(f"Without caching:      ${total_hypothetical:.4f}")
        print(f"Total saved:          ${total_saved:.4f}  ({savings_pct:.1f}%)")
        print(separator)

        if savings_pct < 30 and total_calls > 5:
            print("NOTE: Savings below 30%. Consider:")
            print("  - Increasing static prefix size (more content = more cacheable tokens)")
            print("  - Checking call frequency (5-min TTL requires consistent traffic)")
            print("  - Running 06_cache_invalidation_debug.py to check prefix stability")

        return {
            "hit_rate": cache_hits / total_calls,
            "total_saved": total_saved,
            "savings_pct": savings_pct,
        }

    def reset(self):
        """Clear all recorded calls."""
        self.records = []
        self.recent.clear()


if __name__ == "__main__":
    # Demo with simulated usage records
    monitor = CacheMonitor(provider="anthropic", model="sonnet", alert_threshold=0.6)

    # Simulate 10 calls: 1 write + 9 hits
    class FakeUsage:
        def __init__(self, write, read):
            self.input_tokens = 150
            self.output_tokens = 800
            self.cache_creation_input_tokens = write
            self.cache_read_input_tokens = read

    monitor.record(FakeUsage(write=4000, read=0))    # Cache write
    for _ in range(9):
        monitor.record(FakeUsage(write=0, read=4000))  # Cache hits

    monitor.summary()
