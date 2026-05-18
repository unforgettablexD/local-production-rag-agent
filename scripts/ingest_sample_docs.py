from __future__ import annotations

from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://localhost:8000"
SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"


def main() -> None:
    files = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.is_file():
            files.append(("files", (path.name, path.read_bytes(), "text/plain")))
    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=180)
    if not response.ok:
        print(response.text)
        response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
