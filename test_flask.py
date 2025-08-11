#!/usr/bin/env python3

import os
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def hello():
    return f"""
    <h1>Hello from Flask!</h1>
    <p>Running on port: {os.getenv('PORT', 'NOT SET')}</p>
    <p>If you can see this, Flask is working on Railway!</p>
    <p>Headers: {dict(request.headers)}</p>
    """


@app.route("/health")
def health():
    return {"status": "ok", "port": os.getenv("PORT")}


@app.route("/healthz")
def healthz():
    return "OK"


@app.route("/_health")
def _health():
    return "OK"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
