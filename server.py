"""Serve the frontend and trained correction pipeline on one local origin."""

import json
import mimetypes
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.correction.service import CorrectionService


PROJECT_ROOT = Path(__file__).resolve().parent
# The React + Cult UI build (web/dist). When it hasn't been built, serve the original page.
WEB_DIST = PROJECT_ROOT / "web" / "dist"
LEGACY_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 16_384
MAX_TEXT_LENGTH = 2_000
service = CorrectionService()
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")


def resolve_static(request_path: str) -> tuple[Path, str] | None:
    """Map a URL path to a file inside web/dist (or the legacy page), refusing traversal."""
    path = request_path.split("?", 1)[0].split("#", 1)[0]
    if not (WEB_DIST / "index.html").is_file():
        legacy = LEGACY_FILES.get(path)
        return (PROJECT_ROOT / legacy[0], legacy[1]) if legacy else None

    relative = path.lstrip("/") or "index.html"
    candidate = (WEB_DIST / relative).resolve()
    if not candidate.is_relative_to(WEB_DIST.resolve()):
        return None
    if not candidate.is_file():
        if "." in Path(relative).name:
            return None  # a missing asset, not an app route
        candidate = WEB_DIST / "index.html"  # single-page-app fallback
    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type in ("application/json", "image/svg+xml"):
        content_type += "; charset=utf-8"
    return candidate, content_type


class RequestHandler(BaseHTTPRequestHandler):
    """Handle the app's static assets and same-origin JSON endpoints."""

    def do_HEAD(self) -> None:
        if self.path == "/api/status":
            content = json.dumps(service.get_status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return

        static_file = resolve_static(self.path)
        if static_file is None:
            self.send_error(404, "Not found")
            return
        file_path, content_type = static_file
        content_length = file_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._send_json(200, service.get_status())
            return
        if self.path == "/api/emotion/metrics":
            metrics = service.get_emotion_metrics()
            if metrics is None:
                self._send_json(503, {"error": "The emotion classifier is still training."})
            else:
                self._send_json(200, metrics)
            return
        if self.path.startswith("/api/"):
            self._send_json(404, {"error": "Unknown API endpoint."})
            return

        static_file = resolve_static(self.path)
        if static_file is None:
            self.send_error(404, "Not found")
            return

        file_path, content_type = static_file
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/correct":
            self.send_error(404, "Not found")
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Send the request as application/json."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "Request body must be between 1 and 16 KB."})
            return

        try:
            request_data = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must contain valid JSON."})
            return
        if not isinstance(request_data, dict):
            self._send_json(400, {"error": "Request body must be a JSON object."})
            return

        text = request_data.get("text")
        if not isinstance(text, str):
            self._send_json(400, {"error": 'The "text" field must be a string.'})
            return
        if len(text) > MAX_TEXT_LENGTH:
            self._send_json(
                413,
                {"error": f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer."},
            )
            return
        if not text.strip():
            self._send_json(400, {"error": "Enter some text before running correction."})
            return

        try:
            self._send_json(200, service.correct(text))
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except RuntimeError as error:
            status = service.get_status()
            code = 500 if status["status"] == "error" else 503
            self._send_json(code, {"error": str(error), "status": status})
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"error": "Correction failed. Check the server log."})

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), RequestHandler)
    server.daemon_threads = True
    print(f"Translit Lab is available at http://{host}:{port}", flush=True)
    print("Training the emotion classifier and Phase 2/3 models in the background…", flush=True)
    threading.Thread(target=service.warm_up, name="model-training", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Translit Lab.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
