#!/usr/bin/env python3
"""Open mesh_input_designer.html in the browser, served from a local web server.

Why: the page's map uses OpenStreetMap tiles, and OpenStreetMap's tile servers
refuse requests that carry no Referer header (403 "Referer is required").
A page opened straight from disk (file://, double-click, `open file.html`)
never sends one, so the map stays blank or shows "Access blocked" tiles.
Served over http://localhost the browser sends a Referer and the map loads.

Usage:
    python3 tools/open_mesh_designer.py

A free port is picked automatically, so it never clashes with anything else.

Press Ctrl+C in the terminal to stop the server.
"""
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

PAGE = "mesh_input_designer.html"
HERE = Path(__file__).resolve().parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass


def main() -> None:
    handler = functools.partial(QuietHandler, directory=str(HERE))
    # Port 0: the operating system hands out a port that is free right now.
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]

    url = f"http://localhost:{port}/{PAGE}"
    print(f"Mesh input designer: {url}\nPress Ctrl+C to stop.")
    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
