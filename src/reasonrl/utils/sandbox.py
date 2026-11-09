"""Minimal sandbox for executing model-written Python against unit tests.

Layers of defense (none of them alone is a security boundary):

1. **Static AST screen** - rejects imports outside a whitelist, dangerous
   builtins (``eval``, ``open``, ``getattr`` ...), any dunder attribute access
   (``().__class__.__subclasses__()``), frame/code introspection attributes and
   string constants containing ``__``.
2. **Separate interpreter** - code runs in ``python -I -S`` (isolated mode, no
   site-packages, no user env vars) in a temp working directory with a
   wall-clock timeout. On POSIX, CPU-time and address-space rlimits are set.
3. **Restricted builtins** - ``exec`` with a whitelisted ``__builtins__`` dict
   and a guarded ``__import__``.
4. **Unforgeable result channel** - user stdout/stderr are captured; the runner
   writes its verdict on a line prefixed by a random per-call nonce, so a
   completion that prints ``{"passed": 10}`` cannot spoof the result.
5. **Type-strict comparison** for input/output tests, so an object overriding
   ``__eq__`` to always return True ("AlwaysEqual" hack) fails.

For large-scale RL on untrusted code, run this inside a container / gVisor /
Firecracker with no network; the interface stays the same.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

ALLOWED_MODULES = frozenset(
    {
        "math", "cmath", "itertools", "functools", "collections", "heapq", "bisect",
        "re", "string", "typing", "dataclasses", "fractions", "decimal", "statistics",
        "operator", "copy", "random", "array", "enum", "numbers",
    }
)

BANNED_NAMES = frozenset(
    {
        "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
        "memoryview", "__builtins__", "__loader__", "__spec__",
    }
)

BANNED_ATTRS = frozenset(
    {
        "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "f_globals",
        "f_locals", "f_back", "f_builtins", "f_code", "tb_frame", "tb_next", "co_code",
        "func_globals", "mro",
    }
)

SAFE_BUILTINS = (
    "abs all any ascii bin bool bytearray bytes callable chr classmethod complex dict "
    "divmod enumerate filter float format frozenset hash hex id int isinstance issubclass "
    "iter len list map max min next object oct ord pow print property range repr reversed "
    "round set slice sorted staticmethod str sum super tuple type zip "
    "ArithmeticError AssertionError AttributeError Exception IndexError KeyError "
    "LookupError NameError NotImplementedError OverflowError RecursionError RuntimeError "
    "StopIteration TypeError ValueError ZeroDivisionError NotImplemented True False None"
).split()


def static_check(code: str) -> list[str]:
    """Return a list of violations (empty == passes the screen)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax_error: {exc.msg}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    violations.append(f"import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level != 0 or mod not in ALLOWED_MODULES:
                violations.append(f"import_from:{node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            violations.append(f"name:{node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BANNED_ATTRS:
                violations.append(f"attr:{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                violations.append("dunder_string")
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and any(
            n.startswith("__") for n in node.names
        ):
            violations.append("dunder_global")
    return sorted(set(violations))


@dataclass
class SandboxResult:
    passed: int
    total: int
    timed_out: bool = False
    error: str | None = None
    violations: list = field(default_factory=list)
    per_test: list = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


# The runner executes inside a fresh, isolated interpreter. It must only use
# the standard library and must not trust anything the user code does.
_RUNNER = r"""
import builtins, contextlib, copy, io, json, sys
data = json.loads(sys.stdin.read())
nonce = data["nonce"]
real_stdout = sys.stdout
try:
    import resource
    mem = int(data["memory_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    cpu = int(data["cpu_s"]) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
except Exception:
    pass
sys.setrecursionlimit(2000)
allowed = set(data["allowed_modules"])
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in allowed:
        raise ImportError("import of %r is not allowed" % name)
    return real_import(name, globals, locals, fromlist, level)
safe = {n: getattr(builtins, n) for n in data["safe_builtins"] if hasattr(builtins, n)}
safe["__import__"] = guarded_import
safe["__build_class__"] = builtins.__build_class__
ns = {"__builtins__": safe, "__name__": "__sandbox__"}

def strict_eq(a, b):
    if isinstance(b, bool) or isinstance(a, bool):
        return type(a) is type(b) and a == b
    if type(b) in (int, float) and type(a) in (int, float):
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if type(b) in (list, tuple) and type(a) in (list, tuple):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    if type(b) is dict and type(a) is dict:
        return set(a.keys()) == set(b.keys()) and all(strict_eq(a[k], b[k]) for k in b)
    if b is None:
        return a is None
    if type(b) is str and type(a) is str:
        return a == b
    return False

per_test, error = [], None
sink = io.StringIO()
with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    try:
        exec(compile(data["code"], "<solution>", "exec"), ns)
    except BaseException as e:
        error = "load:" + type(e).__name__
    if error is None:
        for t in data["tests"]:
            try:
                if isinstance(t, str):
                    exec(compile(t, "<test>", "exec"), ns)
                    ok = True
                else:
                    fn = ns[data["entry_point"]]
                    out = fn(*copy.deepcopy(t["input"]))
                    ok = strict_eq(out, t["output"])
            except BaseException:
                ok = False
            per_test.append(bool(ok))
real_stdout.write("\n" + nonce + json.dumps({"per_test": per_test, "error": error}) + "\n")
real_stdout.flush()
"""


def _child_env() -> dict:
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("SYSTEMROOT", "PATH", "TEMP", "TMP"):  # required on Windows
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_tests(
    code: str,
    tests: list,
    entry_point: str | None = None,
    timeout: float = 3.0,
    memory_mb: int = 512,
) -> SandboxResult:
    """Run ``tests`` against ``code``.

    ``tests`` items are either assert strings (``"assert f(1) == 2"``) or
    I/O dicts (``{"input": [1], "output": 2}``) requiring ``entry_point``.
    I/O tests use type-strict comparison and are preferred for RL rewards.
    """
    total = len(tests)
    t0 = time.time()
    violations = static_check(code)
    if violations:
        return SandboxResult(0, total, error="static_check", violations=violations)
    nonce = "@@" + secrets.token_hex(16) + "@@"
    payload = json.dumps(
        {
            "nonce": nonce,
            "code": code,
            "tests": tests,
            "entry_point": entry_point,
            "allowed_modules": sorted(ALLOWED_MODULES),
            "safe_builtins": SAFE_BUILTINS,
            "memory_mb": memory_mb,
            "cpu_s": timeout,
        }
    )
    with tempfile.TemporaryDirectory(prefix="reasonrl_sbx_") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=_child_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(0, total, timed_out=True, error="timeout", duration_s=time.time() - t0)
    verdict = None
    for line in proc.stdout.splitlines():
        if line.startswith(nonce):
            verdict = json.loads(line[len(nonce):])
    if verdict is None:
        err = (proc.stderr or "").strip().splitlines()
        return SandboxResult(
            0, total, error="crash:" + (err[-1] if err else "no_verdict"), duration_s=time.time() - t0
        )
    per_test = [bool(x) for x in verdict["per_test"]]
    return SandboxResult(
        passed=sum(per_test),
        total=total,
        error=verdict["error"],
        per_test=per_test,
        duration_s=time.time() - t0,
    )
