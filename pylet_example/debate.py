#!/usr/bin/env python3
"""
🐍 vs 🦀  Python vs Rust Debate

Two LLM instances argue about programming languages.
Requires two SGLang endpoints deployed via PyLet.

Usage:
    # Deploy both models first (see README.md), then:
    python debate.py <python-fan-endpoint> <rust-fan-endpoint>

    # Example:
    python debate.py 192.168.1.10:15600 192.168.1.10:15601
"""

import sys

from openai import OpenAI

# -- Config --
MODEL = "Qwen/Qwen3.5-2B"
ROUNDS = 3
TOPIC = "Which language is better for building distributed systems?"

PYTHON_SYSTEM = "You are a passionate Python developer. Defend Python in 2-3 sentences. Be witty."
RUST_SYSTEM = "You are a passionate Rust developer. Defend Rust in 2-3 sentences. Be witty."


def main():
    if len(sys.argv) != 3:
        print("Usage: python debate.py <python-fan-endpoint> <rust-fan-endpoint>")
        print("Example: python debate.py 192.168.1.10:15600 192.168.1.10:15601")
        sys.exit(1)

    python_llm = OpenAI(base_url=f"http://{sys.argv[1]}/v1", api_key="na")
    rust_llm = OpenAI(base_url=f"http://{sys.argv[2]}/v1", api_key="na")

    print(f"🎤 Topic: {TOPIC}\n")

    history = []
    msg = TOPIC

    for _ in range(ROUNDS):
        # Python fan responds
        py_resp = python_llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PYTHON_SYSTEM},
                *history,
                {"role": "user", "content": msg},
            ],
        ).choices[0].message.content
        print(f"🐍 Python Fan: {py_resp}\n")
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": py_resp})

        # Rust fan responds to Python fan
        rs_resp = rust_llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RUST_SYSTEM},
                {"role": "user", "content": py_resp},
            ],
        ).choices[0].message.content
        print(f"🦀 Rust Fan: {rs_resp}\n")

        msg = rs_resp

    print("🏁 Debate over!")


if __name__ == "__main__":
    main()
