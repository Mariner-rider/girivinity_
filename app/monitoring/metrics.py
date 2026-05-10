from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["path", "method"])
MODEL_LOAD_SECONDS = Histogram("model_load_seconds", "Model load latency in seconds")
INFERENCE_COUNTER = Counter("inference_requests_total", "Total inference requests")
