"""On-disk response cache for LLM calls.

The meta-loop re-evaluates genomes (re-sampling parents, re-running the seed for the
noise band), so identical (model, system, user) triples recur constantly. Caching makes
those re-evals ~free and fully deterministic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from medusa import config

CACHE_DIR = config.REPO_ROOT / ".cache" / "responses"


def _key(sig: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    for part in (sig, system, user):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class ResponseCache:
    """Keyed by sha256(request-signature \\n system \\n user), where the signature
    covers everything that changes the reply (model, temperature, thinking)."""

    def __init__(self, enabled: bool = True, cache_dir: Path | None = None) -> None:
        self.enabled = enabled and os.environ.get("MEDUSA_NO_CACHE") != "1"
        self.dir = cache_dir or CACHE_DIR
        self.hits = 0
        self.misses = 0

    def get(self, sig: str, system: str, user: str) -> dict | None:
        if not self.enabled:
            return None
        p = self.dir / f"{_key(sig, system, user)}.json"
        if p.exists():
            self.hits += 1
            return json.loads(p.read_text())
        self.misses += 1
        return None

    def put(self, sig: str, system: str, user: str, completion) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{_key(sig, system, user)}.json"
        p.write_text(json.dumps(dataclasses.asdict(completion)))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


# process-wide default; meta-loop passes nocache=True to force fresh calls for the
# seed noise-band measurement.
SHARED = ResponseCache()
