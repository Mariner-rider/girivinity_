"""CodeIntelligenceEngine — static analysis and lightweight code understanding."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeAnalysisResult:
    language: str
    loc: int
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    security_findings: list[dict[str, Any]] = field(default_factory=list)
    complexity_score: float = 0.0
    vulnerable_dependencies: list[dict[str, str]] = field(default_factory=list)
    llm_explanation: str = ""
    llm_suggestions: list[str] = field(default_factory=list)


class CodeIntelligenceEngine:
    LANGUAGE_PATTERNS = {
        "python": re.compile(r"^\s*(import |from |def |class |if __name__)", re.M),
        "javascript": re.compile(r"\b(const|let|var|function)\b|=>|require\("),
        "java": re.compile(r"\b(public class|private |protected |import java)"),
        "go": re.compile(r"\b(package |func )|:=|import \("),
        "rust": re.compile(r"\b(fn |let mut |use |impl |pub struct)"),
        "c": re.compile(r"#include|\bint main\b|\bvoid\b|printf\("),
    }
    KNOWN_VULNERABLE = {
        "python": {"requests<2.25.0", "urllib3<1.26.0", "cryptography<41.0.0"},
        "javascript": {"lodash<4.17.21", "axios<0.21.1", "express<4.17.3"},
    }

    def __init__(self, llm_engine: Any = None) -> None:
        self.llm = llm_engine

    def detect_language(self, code: str) -> str:
        for language, pattern in self.LANGUAGE_PATTERNS.items():
            if pattern.search(code or ""):
                return language
        return "unknown"

    def extract_functions(self, code: str, language: str) -> list[str]:
        if language == "python":
            try:
                tree = ast.parse(code)
                return [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            except SyntaxError:
                return re.findall(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", code, re.M)
        if language in ("javascript", "typescript"):
            matches = re.findall(r"function\s+(\w+)\s*\(|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>", code)
            return [left or right for left, right in matches]
        if language == "go":
            return re.findall(r"func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(", code)
        return []

    def extract_imports(self, code: str, language: str) -> list[str]:
        if language == "python":
            imports: list[str] = []
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module.split(".")[0])
                return sorted(set(imports))
            except SyntaxError:
                return re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, re.M)
        if language == "javascript":
            requires = re.findall(r"require\(['\"]([^'\"]+)['\"]\)", code)
            es_imports = re.findall(r"import\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]", code)
            return sorted(set(requires + es_imports))
        return []

    def run_semgrep(self, code: str, language: str) -> list[dict[str, Any]]:
        suffix = {"python": ".py", "javascript": ".js", "java": ".java", "go": ".go", "rust": ".rs", "c": ".c"}.get(language, ".txt")
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as handle:
                handle.write(code)
                tmp_path = handle.name
            result = subprocess.run(["semgrep", "--config", "p/security-audit", "--json", tmp_path], capture_output=True, text=True, timeout=30)
            if result.returncode in (0, 1) and result.stdout:
                data = json.loads(result.stdout)
                return [
                    {"rule": item.get("check_id"), "message": item.get("extra", {}).get("message", ""), "severity": item.get("extra", {}).get("severity", "INFO"), "line": item.get("start", {}).get("line")}
                    for item in data.get("results", [])
                ]
        except Exception:
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return []

    def run_bandit(self, code: str) -> list[dict[str, Any]]:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
                handle.write(code)
                tmp_path = handle.name
            result = subprocess.run(["bandit", "-f", "json", tmp_path], capture_output=True, text=True, timeout=30)
            if not result.stdout:
                return []
            data = json.loads(result.stdout)
            return [{"severity": item.get("issue_severity"), "confidence": item.get("issue_confidence"), "message": item.get("issue_text"), "line": item.get("line_number")} for item in data.get("results", [])]
        except Exception:
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def cyclomatic_complexity(self, code: str, language: str) -> float:
        if language == "python":
            try:
                tree = ast.parse(code)
                branches = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert, ast.BoolOp, ast.comprehension)) for node in ast.walk(tree))
                return max(1.0, float(branches + 1))
            except SyntaxError:
                pass
        branches = len(re.findall(r"\b(if|elif|else|for|while|try|except|case|catch|switch)\b", code))
        return max(1.0, float(branches + 1))

    def analyse(self, code: str, include_llm: bool = True) -> CodeAnalysisResult:
        code = code or ""
        language = self.detect_language(code)
        functions = self.extract_functions(code, language)
        imports = self.extract_imports(code, language)
        security = self.run_semgrep(code, language)
        if language == "python":
            security.extend(self.run_bandit(code))
        vulnerable_dependencies = self._find_vulnerable_dependencies(imports, language)
        explanation, suggestions = self._llm_review(code, language) if include_llm and self.llm else ("", [])
        return CodeAnalysisResult(
            language=language,
            loc=len([line for line in code.splitlines() if line.strip()]),
            functions=functions,
            imports=imports,
            security_findings=security,
            complexity_score=self.cyclomatic_complexity(code, language),
            vulnerable_dependencies=vulnerable_dependencies,
            llm_explanation=explanation,
            llm_suggestions=suggestions,
        )

    analyze = analyse

    def _find_vulnerable_dependencies(self, imports: list[str], language: str) -> list[dict[str, str]]:
        findings = []
        for package in imports:
            root = package.split("/")[0].split(".")[0]
            for issue in self.KNOWN_VULNERABLE.get(language, set()):
                if issue.startswith(f"{root}<"):
                    findings.append({"package": root, "issue": issue})
        return findings

    def _llm_review(self, code: str, language: str) -> tuple[str, list[str]]:
        prompt = f'Explain this {language} code briefly and list two improvements as JSON.\nCode:\n{code[:1500]}'
        try:
            result = self.llm.generate(prompt, max_new_tokens=300, temperature=0.1)
            text = getattr(result, "text", str(result))
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                parsed = json.loads(match.group(0))
                return parsed.get("explanation", ""), list(parsed.get("suggestions", []))
        except Exception:
            pass
        return "", []
