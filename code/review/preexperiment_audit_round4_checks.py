"""Rerun review-3 regressions against 5642dfb without changing historical results.

The only test-interface adaptation is support for the new _ci(target=...) argument.
All experiment functions and expected scientific invariants remain unchanged.
"""
from pathlib import Path


def main():
    original = Path(__file__).with_name("preexperiment_audit_round3_checks.py")
    source = original.read_text(encoding="utf-8")
    replacements = {
        "def capture(boots, point, ci):": "def capture(boots, point, ci, target=None):",
        "return old_ci(boots, point, ci)": "return old_ci(boots, point, ci, target=target)",
        "preexperiment_audit_round3_results.json": "preexperiment_audit_round4_results.json",
        'revision="4081130"': 'revision="5642dfb"',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Historical regression interface changed: {before}")
        source = source.replace(before, after)
    namespace = {"__name__": "audit_round4_regressions", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(original), "exec"), namespace)
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
