"""Code creation and analysis system with repository ingestion, static analysis, bug detection, API testing, and auto-fix suggestions."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class StaticIssue:
    file_path: str
    line: int
    severity: str
    message: str
    suggestion: str


@dataclass(slots=True)
class APITestCase:
    name: str
    method: str
    url: str
    expected_status: int


@dataclass(slots=True)
class APITestResult:
    name: str
    passed: bool
    status_code: int
    response_excerpt: str


class CodeIntelligenceEngine:
    def generate_stable_code(self, spec: str, language: str = "python") -> str:
        if language.lower() != "python":
            return f"// Production-ready template generation for {language} is not configured yet."
        return (
            '"""Production-ready module."""\n\n'
            "from __future__ import annotations\n\n"
            f"# Spec: {spec}\n\n"
            "def main() -> None:\n"
            "    \"\"\"Entrypoint with explicit typing and deterministic behavior.\"\"\"\n"
            "    pass\n"
        )

    def ingest_github_repo(self, repo_url: str, target_dir: str) -> str:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
        return str(target)

    def static_analysis(self, repo_dir: str) -> list[StaticIssue]:
        issues: list[StaticIssue] = []
        for py_file in Path(repo_dir).rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.returns is None:
                    issues.append(
                        StaticIssue(
                            file_path=str(py_file),
                            line=node.lineno,
                            severity="medium",
                            message=f"Function '{node.name}' has no return type hint.",
                            suggestion="Add explicit return type annotation for production stability.",
                        )
                    )
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    issues.append(
                        StaticIssue(
                            file_path=str(py_file),
                            line=node.lineno,
                            severity="low",
                            message="print() found in code path.",
                            suggestion="Use structured logging instead of print() in production code.",
                        )
                    )
        return issues

    def bug_detection(self, repo_dir: str) -> list[StaticIssue]:
        bugs: list[StaticIssue] = []
        for py_file in Path(repo_dir).rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            if "TODO" in source or "FIXME" in source:
                bugs.append(
                    StaticIssue(
                        file_path=str(py_file),
                        line=1,
                        severity="medium",
                        message="Potential unfinished implementation markers found (TODO/FIXME).",
                        suggestion="Resolve TODO/FIXME with concrete implementation and tests.",
                    )
                )
        return bugs

    def api_test(self, test_cases: list[APITestCase], timeout_s: int = 10) -> list[APITestResult]:
        import urllib.request
        import urllib.error

        results: list[APITestResult] = []
        for case in test_cases:
            req = urllib.request.Request(case.url, method=case.method.upper())
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read(250).decode("utf-8", errors="ignore")
                    status = resp.status
            except urllib.error.HTTPError as err:
                status = err.code
                body = err.read(250).decode("utf-8", errors="ignore") if err.fp else ""
            passed = status == case.expected_status
            results.append(APITestResult(case.name, passed, status, body))
        return results

    def auto_fix_suggestions(self, issues: list[StaticIssue]) -> list[dict]:
        fixes = []
        for issue in issues:
            fixes.append(
                {
                    "file": issue.file_path,
                    "line": issue.line,
                    "issue": issue.message,
                    "suggested_fix": issue.suggestion,
                    "priority": issue.severity,
                }
            )
        return fixes

    def analyze_repo(self, repo_dir: str) -> dict:
        static_issues = self.static_analysis(repo_dir)
        bug_issues = self.bug_detection(repo_dir)
        all_issues = static_issues + bug_issues
        return {
            "static_analysis": [asdict(issue) for issue in static_issues],
            "bug_detection": [asdict(issue) for issue in bug_issues],
            "auto_fix_suggestions": self.auto_fix_suggestions(all_issues),
            "summary": {
                "static_issue_count": len(static_issues),
                "bug_count": len(bug_issues),
                "total_issues": len(all_issues),
            },
        }

    def export_report(self, report: dict, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return str(path)
