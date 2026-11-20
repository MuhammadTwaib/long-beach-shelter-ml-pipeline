# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Download the raw CSV (only needed if data/raw/ is empty) and verify its checksum.

    python -m src.download_data

The file is pinned to one Git commit and checked with SHA-256, so everyone gets
byte-for-byte the same raw data.
"""

import hashlib
import sys
import urllib.request

from src import config


def sha256(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Download if missing, then verify. Exit code 1 if the checksum does not match."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not config.RAW_FILE.exists():
        print(f"Downloading {config.RAW_URL}")
        urllib.request.urlretrieve(config.RAW_URL, config.RAW_FILE)
    actual = sha256(config.RAW_FILE)
    if actual != config.RAW_SHA256:
        print(f"Checksum mismatch!\n expected {config.RAW_SHA256}\n got      {actual}", file=sys.stderr)
        return 1
    print(f"Raw data OK: {config.RAW_FILE} (sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
