"""Optional public tunnels. Default is 'none' (local only). Nothing here runs unless config.tunnel says so.

UNTESTED in this repo snapshot: ngrok/cloudflared need network access and your own account/binary.
Secrets (NGROK_AUTHTOKEN, optional NGROK_DOMAIN) are read from the environment; the notebook fills them
from Kaggle Secrets. They are never written to disk or printed.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Callable, Optional

logger = logging.getLogger("http_worker")


def start_tunnel(kind: str, port: int) -> Optional[Callable[[], None]]:
    kind = (kind or "none").lower()
    if kind == "none":
        logger.info("[TUNNEL] none: server is reachable only inside this machine (127.0.0.1:%d)", port)
        return None
    if kind == "ngrok":
        token = os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            raise RuntimeError("tunnel=ngrok needs NGROK_AUTHTOKEN (Kaggle Secret)")
        from pyngrok import conf, ngrok  # pip install pyngrok
        conf.get_default().auth_token = token
        options = {}
        if os.environ.get("NGROK_DOMAIN"):
            options["domain"] = os.environ["NGROK_DOMAIN"]
        tunnel = ngrok.connect(port, "http", **options)
        logger.info("[TUNNEL] ngrok public URL: %s  (free plan: 1 GB/month out, 20k requests/month)", tunnel.public_url)
        return lambda: ngrok.disconnect(tunnel.public_url)
    if kind == "cloudflared":
        exe = shutil.which("cloudflared")
        if not exe:
            raise RuntimeError("tunnel=cloudflared needs the cloudflared binary on PATH")
        proc = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # type: ignore[union-attr]
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                logger.info("[TUNNEL] cloudflared public URL (changes every run): %s", m.group(0))
                break
        return proc.terminate
    raise ValueError(f"unknown tunnel '{kind}' (use none, ngrok or cloudflared)")
