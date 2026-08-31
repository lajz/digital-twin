"""Conformance checks for a generated `twin.py` -- static (AST) and runtime (object)."""

from __future__ import annotations

import ast
import dataclasses

from medusa.contract.interface import (
    ALLOWED_IMPORT_ROOTS,
    FORBIDDEN_IMPORT_ROOTS,
    PARAM_TRIPLE_LEN,
)

_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "globals"}


@dataclasses.dataclass(slots=True)
class CheckResult:
    ok: bool
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)

    def merge(self, other: "CheckResult") -> "CheckResult":
        return CheckResult(
            ok=self.ok and other.ok,
            errors=[*self.errors, *other.errors],
            warnings=[*self.warnings, *other.warnings],
        )

    def as_report(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARN:  {w}")
        return "\n".join(lines) if lines else "OK"


def _import_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level and node.level > 0:  # relative import
        return ["<relative>"]
    return [(node.module or "").split(".")[0]]


def check_twin_source(
    source: str, required_methods: tuple[str, ...] = ("fit", "predict")
) -> CheckResult:
    """Static checks: parseable, import allowlist, no dangerous builtins, has `class Twin`."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CheckResult(ok=False, errors=[f"twin.py does not parse: {exc}"])

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in _import_roots(node):
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"forbidden import: {root!r}")
                elif root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(
                        f"import {root!r} is not on the allowlist "
                        f"(stdlib subset + numpy + scipy only)"
                    )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                errors.append(f"forbidden call: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "fork"}:
            errors.append(f"forbidden attribute access: .{node.attr}")

    twin_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Twin"),
        None,
    )
    if twin_cls is None:
        errors.append("no top-level `class Twin` found")
    else:
        assigned = {
            t.id
            for stmt in twin_cls.body
            if isinstance(stmt, ast.Assign)
            for t in stmt.targets
            if isinstance(t, ast.Name)
        }
        methods = {s.name for s in twin_cls.body if isinstance(s, ast.FunctionDef)}
        for attr in ("FAMILY", "PARAMS"):
            if attr not in assigned:
                errors.append(f"Twin is missing class attribute {attr!r}")
        for meth in required_methods:
            if meth not in methods:
                errors.append(f"Twin is missing method {meth!r}")

    return CheckResult(ok=not errors, errors=errors, warnings=warnings)


def check_twin_object(
    twin: object,
    params: dict | None = None,
    required_methods: tuple[str, ...] = ("fit", "predict"),
) -> CheckResult:
    """Runtime checks on an instantiated Twin (and optionally a fitted param dict)."""
    errors: list[str] = []
    warnings: list[str] = []

    for attr in ("FAMILY", "PARAMS", *required_methods):
        if not hasattr(twin, attr):
            errors.append(f"Twin instance has no attribute {attr!r}")
    if errors:
        return CheckResult(ok=False, errors=errors)

    family = getattr(twin, "FAMILY")
    if not isinstance(family, str) or not family.strip():
        errors.append("FAMILY must be a non-empty string")

    prm = getattr(twin, "PARAMS")
    if not isinstance(prm, dict) or not prm:
        errors.append("PARAMS must be a non-empty dict")
        return CheckResult(ok=False, errors=errors)

    for name, triple in prm.items():
        if (
            not isinstance(triple, (tuple, list))
            or len(triple) != PARAM_TRIPLE_LEN
        ):
            errors.append(f"PARAMS[{name!r}] must be (low, high, unit)")
            continue
        low, high, unit = triple
        if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            errors.append(f"PARAMS[{name!r}] bounds must be numeric")
        elif low >= high:
            errors.append(f"PARAMS[{name!r}] requires low < high (got {low}, {high})")
        if not isinstance(unit, str) or not unit:
            warnings.append(f"PARAMS[{name!r}] unit should be a non-empty string")

    for meth in required_methods:
        if not callable(getattr(twin, meth, None)):
            errors.append(f"Twin.{meth} is not callable")

    if params is not None and not errors:
        missing = set(prm) - set(params)
        if missing:
            errors.append(f"fit() did not return required params: {sorted(missing)}")
        else:
            # extra keys (e.g. stashed internal state) are ignored; only PARAMS are checked
            for name in prm:
                low, high, _ = prm[name]
                value = params[name]
                if not isinstance(value, (int, float)):
                    errors.append(f"fitted {name!r} is not a number: {value!r}")
                elif not (low <= value <= high):
                    errors.append(
                        f"fitted {name!r}={value:g} is outside prior [{low:g}, {high:g}]"
                    )

    return CheckResult(ok=not errors, errors=errors, warnings=warnings)
