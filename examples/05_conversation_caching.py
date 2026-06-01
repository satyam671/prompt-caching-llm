"""
05_conversation_caching.py
---------------------------
Caching conversation history — not just the system prompt.

In multi-turn conversations, the full message history grows with each turn.
On long conversations this becomes expensive fast: by turn 10, you're
paying for 9 turns of history on every single call.

Anthropic lets you cache the conversation history prefix too.
Mark the last message in the history with cache_control, and the entire
history up to that point is cached for subsequent turns.

This is the most underused form of prompt caching.
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": """You are a research assistant helping analyze a collection of documents.
You maintain context across the conversation and build on previous analysis.
Be concise. Reference earlier findings when relevant.""",
        "cache_control": {"type": "ephemeral"}   # Cache the system prompt
    }
]


def multi_turn_with_history_caching():
    """
    Simulates a multi-turn conversation where the growing history is cached.

    Pattern:
    - Turn 1: system cached, no history yet
    - Turn 2: system cached, turn 1 history cached, only turn 2 is new
    - Turn N: system cached, turns 1..N-1 cached, only turn N is new

    The key: mark the second-to-last message in history with cache_control.
    Everything up to and including it gets cached.
    """

    conversation_history = []

    user_turns = [
        "I have three documents to analyze. Let's start with this one: 'The city council voted 7-2 to approve the transit expansion. Ridership growth projected at 40% over 5 years. Critics cite pre-pandemic cost estimates.' What are the key reliability concerns?",
        "Good. Second document: 'Internal survey shows remote workers report 23% higher satisfaction. Online survey, voluntary, N=340.' How does the methodology affect how much we should trust this?",
        "Third document: 'Drug trial halted. 12 of 200 treatment participants showed elevated liver enzymes. Control group unaffected.' Compared to the other two documents, how does the strength of evidence compare?",
        "Across all three, which document makes the strongest factual claims relative to the quality of its evidence?"
    ]

    print("=" * 60)
    print("CONVERSATION CACHING: History grows, costs stay controlled")
    print("=" * 60)

    for turn_num, user_message in enumerate(user_turns, 1):
        print(f"\n--- Turn {turn_num} ---")
        print(f"User: {user_message[:80]}...")

        # Build the messages array with cache_control on the history prefix.
        # The pattern: mark the last EXISTING message (not the new one) for caching.
        # This tells Anthropic: "cache everything up to here; the new message is dynamic."

        messages_to_send = []

        if conversation_history:
            # Add all history except the last message without cache_control
            for msg in conversation_history[:-1]:
                messages_to_send.append(msg)

            # Mark the last history message for caching
            last_hist = conversation_history[-1].copy()
            if isinstance(last_hist["content"], str):
                last_hist["content"] = [
                    {
                        "type": "text",
                        "text": last_hist["content"],
                        "cache_control": {"type": "ephemeral"}   # Cache the history prefix
                    }
                ]
            messages_to_send.append(last_hist)

        # Add the new (dynamic) user message — NOT cached
        messages_to_send.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages_to_send
        )

        assistant_reply = response.content[0].text
        usage = response.usage

        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_write = getattr(usage, "cache_creation_input_tokens", 0)

        print(f"Input tokens:    {usage.input_tokens}")
        print(f"Cache write:     {cache_write}")
        print(f"Cache read:      {cache_read}  {'(history cached)' if cache_read > 0 else '(first turn)'}")
        print(f"Output tokens:   {usage.output_tokens}")
        print(f"Reply preview:   {assistant_reply[:120]}...")

        # Add this turn to history for the next call
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

    print("\n" + "=" * 60)
    print("Without history caching, turn 4 would pay for all 3 prior")
    print("turns of history at full input price on every call.")
    print("With caching, only the new turn is priced at full rate.")
    print("=" * 60)


if __name__ == "__main__":
    multi_turn_with_history_caching()
