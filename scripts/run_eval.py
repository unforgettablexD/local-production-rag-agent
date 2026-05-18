from __future__ import annotations

import argparse
import json

import requests

BACKEND_URL = "http://localhost:8000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    payload = {
        "top_k": args.top_k,
        "max_questions": args.max_questions,
        "fast_mode": args.fast,
    }
    response = requests.post(f"{BACKEND_URL}/evaluate", json=payload, timeout=args.timeout)
    if not response.ok:
        print(response.text)
        response.raise_for_status()
    payload = response.json()
    print("Evaluation Metrics")
    print(json.dumps(payload["metrics"], indent=2))
    print("\nFirst 5 Results")
    print(json.dumps(payload["results"][:5], indent=2))


if __name__ == "__main__":
    main()
