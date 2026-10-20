"""Dependency-free HTTP API and dashboard for a LiveController."""

from __future__ import annotations

import argparse
import json
import threading
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MAX_REQUEST_BYTES = 1_048_576


def create_handler(controller: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: object, content_type: str = "application/json") -> None:
            body = (
                json.dumps(payload).encode("utf-8")
                if content_type == "application/json"
                else str(payload).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in {"/", "/dashboard"}:
                html = """<!doctype html>
                <html lang="en"><head><meta charset="utf-8">
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <title>Adaptive QoS</title>
                <style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:72rem}
                pre{background:#f4f4f4;padding:1rem;overflow:auto}</style></head>
                <body><h1>Adaptive QoS controller</h1>
                <p id="message">Connecting to controller...</p>
                <pre id="status">loading...</pre>
                <script>
                const statusElement = document.getElementById('status');
                const messageElement = document.getElementById('message');
                async function refreshStatus() {
                  try {
                    const response = await fetch('/api/status', {cache: 'no-store'});
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    statusElement.textContent = JSON.stringify(await response.json(), null, 2);
                    messageElement.textContent = 'Controller connected';
                  } catch (error) {
                    messageElement.textContent = `Controller unavailable: ${error.message}`;
                  }
                }
                refreshStatus();
                setInterval(refreshStatus, 1000);
                </script></body></html>"""
                self._write(200, html, "text/html; charset=utf-8")
            elif path == "/health":
                self._write(200, {"ok": True, **controller.status()})
            elif path == "/api/status":
                self._write(200, controller.status())
            else:
                self._write(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in {"/api/feedback", "/api/classify"}:
                self._write(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
                if self.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if path == "/api/classify":
                    self._write(200, controller.classify(payload))
                    return
                drifted = controller.feedback(str(payload["predicted"]), str(payload["actual"]))
                self._write(200, {"drift_detected": drifted})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._write(400, {"error": str(exc)})
            except Exception:
                self._write(500, {"error": "internal server error"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(
    controller: Any,
    host: str = "127.0.0.1",
    port: int = 8080,
    monitor_interval: float | None = None,
) -> None:
    requested_port = port
    for candidate_port in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate_port), create_handler(controller))
            port = candidate_port
            break
        except OSError as exc:
            if candidate_port == port + 19:
                raise OSError(
                    f"Could not bind ports {requested_port}-{port + 19}. "
                    "Choose an unused port with --port."
                ) from exc
    else:
        raise RuntimeError("Unable to create the API server.")

    if port != requested_port:
        print(
            f"Port {requested_port} is busy; using available port {port}.",
            flush=True,
        )
    print(f"Dashboard/API listening on http://{host}:{port}")
    stop = threading.Event()
    if monitor_interval is not None:
        threading.Thread(
            target=controller.run_monitor,
            kwargs={"interval": monitor_interval, "stop": stop},
            daemon=True,
        ).start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--dataset",
        help="training dataset used for drift-triggered retraining",
    )
    parser.add_argument(
        "--output-dir",
        help="directory for retrained models (defaults to the model directory)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--monitor-interval", type=float, default=None)
    args = parser.parse_args()
    from qos_ml.controller import LiveController
    serve(
        LiveController(
            args.model,
            dataset_path=args.dataset,
            retrain_output=args.output_dir,
        ),
        args.host,
        args.port,
        args.monitor_interval,
    )


if __name__ == "__main__":
    main()
