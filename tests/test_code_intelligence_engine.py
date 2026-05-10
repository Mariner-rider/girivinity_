from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from code_intelligence_engine import APITestCase, CodeIntelligenceEngine


def test_generate_and_analyze_repo(tmp_path: Path):
    engine = CodeIntelligenceEngine()
    code = engine.generate_stable_code("Build deterministic service")
    assert "Production-ready module" in code

    repo = tmp_path / "repo"
    repo.mkdir()
    sample = repo / "sample.py"
    sample.write_text("def f(x):\n    print(x)\n    # TODO fix\n", encoding="utf-8")

    report = engine.analyze_repo(str(repo))
    assert report["summary"]["total_issues"] >= 2
    assert len(report["auto_fix_suggestions"]) >= 2


def test_api_testing_and_report_export(tmp_path: Path):
    engine = CodeIntelligenceEngine()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    tests = [APITestCase(name="local", method="GET", url=f"http://127.0.0.1:{server.server_port}", expected_status=200)]
    results = engine.api_test(tests)
    server.shutdown()
    assert len(results) == 1
    assert isinstance(results[0].passed, bool)

    report_path = engine.export_report({"ok": True}, str(tmp_path / "report.json"))
    assert Path(report_path).exists()
