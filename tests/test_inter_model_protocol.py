import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.engines.inter_model_protocol import DelegatedTask, InterModelCommunicationProtocol, ModelEndpoint


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        response = {
            "success": True,
            "output": f"done:{payload['task_id']}",
            "confidence": 0.9,
        }
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_delegate_and_aggregate_with_trust_scoring():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    protocol = InterModelCommunicationProtocol()
    endpoint_a = ModelEndpoint(model_id="m1", base_url=f"http://127.0.0.1:{server.server_port}", metadata={"reliability": 0.9})
    endpoint_b = ModelEndpoint(model_id="m2", base_url=f"http://127.0.0.1:{server.server_port}", metadata={"reliability": 0.6})

    task = DelegatedTask(task_id="t-1", instruction="summarize")
    resp_a = protocol.delegate_task(endpoint_a, task)
    resp_b = protocol.delegate_task(endpoint_b, task)

    aggregate = protocol.aggregate_results({"m1": endpoint_a, "m2": endpoint_b}, [resp_a, resp_b])
    server.shutdown()

    assert aggregate["selected_model"] == "m1"
    assert aggregate["all_success"] is True
    assert aggregate["trust_scores"]["m1"] >= aggregate["trust_scores"]["m2"]
