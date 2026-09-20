"""The in-run critic: soft natural-language feedback that steers the inner loop.

The within-run analogue of `meta.digest.reflect()`. Each Nth scored iteration the loop
hands the developing twin -- its source, its holdout error, its plausibility, the
families tried so far -- to a separate LLM with a *stance* (a ships coach that suggests
the next mechanism, or a skeptic that red-teams the twin's stated assumptions). The
reply is appended verbatim to the next iteration's prompt. The critic has NO reward
signal and NO veto: the deterministic domain constraints + the plausibility floor stay
the only hard gate, so it cannot be reward-hacked.

Dependency-free leaf: stdlib only (plus a function-local import of `Completion` in the
dry-run stand-in). Must not import `medusa.agent.*` / `medusa.harness.*` /
`medusa.domains.*` -- `config` imports this module, so anything heavier would cycle.
The loop pulls the primitives (strings, dicts, lists) out of its own objects and passes
them in.
"""

from __future__ import annotations

import string

# --- stance system prompts (the evolvable half; stable across a run) -----------

COACH_PROMPT = """\
You are the ships-coach critic embedded in a loop that is building a mechanistic
digital twin of a real system. Each round you see the developing twin, its latest
holdout error, its plausibility, and the model families tried so far. Steer the search:
in at most 120 words of plain prose -- no code, no bullet lists -- give the modeller ONE
concrete mechanism or structural change to try next that is most likely to cut the
forecast error, and say briefly why the current attempt falls short. Be specific about
the mechanism ("add an explicit lag phase governed by a physiological adjustment rate",
not "improve the model"). You have no veto and cannot change any setting; your note is
advice appended to the next prompt.
"""

SKEPTIC_PROMPT = """\
You are the skeptic critic embedded in a loop that is building a mechanistic digital
twin of a real system. Each round you see the developing twin, the assumptions it states
in METADATA, its latest holdout error and plausibility, and the families tried so far.
Red-team it: in at most 120 words of plain prose -- no code, no bullet lists -- name the
ONE stated assumption or structural choice most likely to be wrong or unsupported by the
data, explain what in the observations contradicts it, and say what to check or relax.
Prefer the assumption whose failure would most distort the forecast. You have no veto and
cannot change any setting; your note is advice appended to the next prompt.
"""

_STANCES: dict[str, str] = {"coach": COACH_PROMPT, "skeptic": SKEPTIC_PROMPT}
STANCES = tuple(_STANCES)


def prompt_for_stance(name: str) -> str:
    """The seed system prompt for a stance name. Falls back to the coach."""
    return _STANCES.get(name, COACH_PROMPT)


# --- domain-conditional default -------------------------------------------------

# A live A/B (2026-09-19, `medusa bench --iters 8 --temperature 0.35`) found the critic
# is not a uniform win: on real E. coli data it roughly halved holdout sMAPE (it catches
# real-data-specific failure modes -- Monod-term unidentifiability, segmentation jitter
# read as biology, missing lag phase -- that the base agent doesn't self-correct); on
# synthetic bacterial data it made no measurable difference; on the synthetic SaaS domain
# it regressed the (multi-dataset) result. A follow-up multi-replicate saas-seed-only A/B
# (5 reps/condition, response cache disabled) found that SaaS regression doesn't
# replicate at the single-dataset level: critic-on vs critic-off differ by well under 1x
# the within-condition run-to-run stdev there -- see the note by `domains.saas._register`
# for the numbers. So the default stays per-domain, not global: "real" is a proven win;
# "business" has no proven win (a wash, not a regression), so it stays off as the
# conservative no-extra-cost-for-no-benefit default.
DEFAULT_ON_KINDS = frozenset({"real"})


def resolve_enabled(critic_enabled: bool | None, domain_kind: str) -> bool:
    """Effective on/off state for one dataset. `critic_enabled=None` (the `LoopConfig`
    default, and a genome that never touched the knob) resolves from `domain_kind`; an
    explicit True/False -- from the CLI or a genome that did touch the knob -- always
    wins, so the meta-loop and `--critic`/`--no-critic` keep meaning "force it"."""
    if critic_enabled is not None:
        return critic_enabled
    return domain_kind in DEFAULT_ON_KINDS


# --- the critic's user-prompt scaffold (rendered fresh each iteration) ---------

CRITIC_CONTEXT_TEMPLATE = """\
You are reviewing iteration {iteration} of an in-progress digital-twin search.
Your stance: {stance}.

## Dataset

{datasheet}

## Observations (fit window only, downsampled)

{obs_table}

## Hard constraints every candidate must satisfy

{constraint_names}

## The latest candidate

family: {family}
status: {status}
combined holdout sMAPE: {holdout_smape}
plausibility: {plausibility}

per-observable holdout error:

{per_observable_table}

## Model families tried so far

{archive_summary}

## Recent iterations

{recent_trace}

## Latest candidate source

```python
{candidate_source}
```

Write your single most useful note now, following your stance instructions
(at most 120 words, plain prose, no code, one main suggestion).
"""


# --- rendering helpers --------------------------------------------------------


def _fmt_num(x: object, fmt: str = "{:.4f}") -> str:
    try:
        xf = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if xf != xf or xf in (float("inf"), float("-inf")):
        return "n/a"
    return fmt.format(xf)


def _per_observable_table(per_observable: dict | None) -> str:
    if not per_observable:
        return "(no scored trajectory -- the candidate was invalid or crashed)"
    rows = ["| observable | holdout sMAPE |", "|---|---|"]
    for k, v in per_observable.items():
        rows.append(f"| {k} | {_fmt_num(v, '{:.3f}')} |")
    return "\n".join(rows)


def _recent_trace_text(rows: list[dict] | None) -> str:
    if not rows:
        return "(no earlier iterations)"
    out = []
    for r in rows:
        out.append(
            f"- iter {r.get('iter', '?')}: family={r.get('family') or '-'}, "
            f"status={r.get('status', '?')}, "
            f"holdout sMAPE={_fmt_num(r.get('holdout_smape'), '{:.3f}')}"
        )
    return "\n".join(out)


def render_critic_context(
    template: str,
    *,
    iteration: int,
    datasheet: str,
    obs_table: str,
    constraint_names: list[str] | tuple[str, ...],
    family: str | None,
    status: str,
    holdout_smape: object,
    plausibility: object,
    per_observable: dict | None,
    archive_summary: str,
    candidate_source: str | None,
    recent_trace: list[dict] | None,
    stance: str,
) -> str:
    """Deterministic assembler. Tolerates a template that omits slots (same trick as
    `prompts.render_iteration`) and degrades gracefully when the candidate produced no
    scored trajectory."""
    fields = {
        "iteration": str(iteration),
        "datasheet": (datasheet or "").strip() or "(no datasheet)",
        "obs_table": (obs_table or "").strip() or "(none)",
        "constraint_names": ", ".join(constraint_names) if constraint_names else "(none)",
        "family": family or "(unnamed / invalid)",
        "status": status or "(unknown)",
        "holdout_smape": _fmt_num(holdout_smape),
        "plausibility": _fmt_num(plausibility, "{:.2f}"),
        "per_observable_table": _per_observable_table(per_observable),
        "archive_summary": (archive_summary or "").strip() or "(nothing tried yet)",
        "candidate_source": (candidate_source or "").strip() or "(no code produced)",
        "recent_trace": _recent_trace_text(recent_trace),
        "stance": stance or "coach",
    }
    present = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    return template.format(**{k: v for k, v in fields.items() if k in present})


def format_note(text: str, stance: str = "coach") -> str:
    """Wrap the critic's raw reply for injection into the next prompt. Empty -> ""."""
    body = (text or "").strip()
    if not body:
        return ""
    return f"## Critic feedback ({stance})\n\n{body}"


# --- dry-run stand-in (mirrors meta.prompts.CannedMetaClient) -----------------


class CannedCritic:
    """Cycles canned notes. No network, no cost. For --dry-run and tests."""

    model = "dry-run-critic"

    _NOTES = (
        "The current attempt treats growth as unconstrained exponential, but the fit "
        "window already shows the per-step rate falling as the count rises. Next, add an "
        "explicit carrying-capacity term so the rate declines toward zero as N approaches "
        "K, and fit K jointly rather than fixing it.",
        "Your model has no lag phase, yet the first few observations are nearly flat "
        "before growth accelerates. Introduce a physiological-adjustment state that "
        "delays the onset of exponential growth, parameterised by a single adjustment "
        "rate, and check whether the early-window residuals shrink.",
        "The single-pool assumption looks shaky: late fit-window points bend more sharply "
        "than one rate constant allows. Try a two-stage structure -- an actively dividing "
        "sub-population feeding a non-dividing one -- and see if the holdout curvature is "
        "captured without hurting the early fit.",
    )

    def __init__(self) -> None:
        self._i = 0

    def complete(self, system: str, user: str):
        from medusa.agent.deepseek import Completion

        text = self._NOTES[self._i % len(self._NOTES)]
        self._i += 1
        return Completion(
            text=text,
            prompt_tokens=len(system) // 4 + len(user) // 4,
            response_tokens=len(text) // 4,
            model=self.model,
        )
