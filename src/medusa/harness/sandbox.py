"""Run `evaluate_source` in a subprocess with a hard wall-clock timeout.

Isolation is best-effort (separate process, scrubbed env, no inherited proxies, cwd in a
temp dir). It is NOT a security sandbox -- candidate code is model-generated and only
lightly vetted by the AST checker. Acceptable for local research use; documented in the
README.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from medusa import config
from medusa.config import LoopConfig
from medusa.data import build
from medusa.harness.evaluate import EvalResult, evaluate_source


def run_candidate(
    source: str, cfg: LoopConfig, *, processed_dir: Path | None = None
) -> EvalResult:
    processed_dir = processed_dir or config.PROCESSED_DIR

    try:
        spatial = build.load(processed_dir=processed_dir).task.mode == "spatial"
    except Exception:
        spatial = False
    timeout_s = cfg.spatial_candidate_timeout_s if spatial else cfg.candidate_timeout_s

    with tempfile.TemporaryDirectory(prefix="medusa-cand-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "twin.py").write_text(source)
        (tmp_path / "config.json").write_text(cfg.to_json())

        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
        }
        env["MEDUSA_SANDBOX"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "medusa.harness.sandbox",
                    str(tmp_path / "twin.py"),
                    str(processed_dir),
                    str(tmp_path / "config.json"),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(
                crashed=True,
                error=f"candidate exceeded {timeout_s:.0f}s wall-clock timeout",
            )

        payload = _extract_json(proc.stdout)
        if payload is None:
            return EvalResult(
                crashed=True,
                error="sandbox produced no result JSON\n"
                f"stdout tail: {proc.stdout[-800:]!r}\nstderr tail: {proc.stderr[-800:]!r}",
            )
        return _result_from_dict(payload)


def _extract_json(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _result_from_dict(d: dict) -> EvalResult:
    fields = {f for f in EvalResult.__slots__}  # type: ignore[attr-defined]
    return EvalResult(**{k: v for k, v in d.items() if k in fields})


def _runner_main(argv: list[str]) -> int:
    twin_path, processed_dir, config_path = argv
    source = Path(twin_path).read_text()
    cfg = LoopConfig.from_dict(json.loads(Path(config_path).read_text()))
    dataset = build.load(processed_dir=Path(processed_dir))
    try:
        result = evaluate_source(source, dataset, cfg)
        print(json.dumps(result.to_dict(), default=str))
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"crashed": True, "error": f"sandbox runner crashed: {exc!r}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_runner_main(sys.argv[1:]))
