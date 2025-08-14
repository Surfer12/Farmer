import subprocess
import sys
import os

# Default ports; can be overridden via env vars
HR_PORT = int(os.environ.get("HR_PORT", 8001))
IT_PORT = int(os.environ.get("IT_PORT", 8002))

# HTTPS configuration
USE_HTTPS = str(os.environ.get("USE_HTTPS", "0")).lower() in {"1", "true", "yes"}
SSL_CERT_PATH = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
SSL_KEY_PATH = os.environ.get("SSL_KEY_PATH", "certs/key.pem")

# Host binding: default to localhost unless ALLOW_REMOTE=1/true/yes
ALLOW_REMOTE = str(os.environ.get("ALLOW_REMOTE", "0")).lower() in {"1", "true", "yes"}
HOST = "0.0.0.0" if ALLOW_REMOTE else "127.0.0.1"

# Compute repository root (two levels up from this launcher file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Build uvicorn commands
hr_cmd = [
    "/Users/ryan_david_oates/Library/Python/3.9/bin/uvicorn",
    "assistants.hr_assistant.main:app",
    "--host",
    HOST,
    "--port",
    str(HR_PORT),
]

it_cmd = [
    "/Users/ryan_david_oates/Library/Python/3.9/bin/uvicorn",
    "assistants.it_assistant.main:app",
    "--host",
    HOST,
    "--port",
    str(IT_PORT),
]

# Add SSL options if HTTPS is enabled
if USE_HTTPS:
    ssl_cert_full_path = os.path.join(ROOT, SSL_CERT_PATH)
    ssl_key_full_path = os.path.join(ROOT, SSL_KEY_PATH)
    
    if not os.path.exists(ssl_cert_full_path) or not os.path.exists(ssl_key_full_path):
        print(f"ERROR: SSL certificate files not found!")
        print(f"  Certificate: {ssl_cert_full_path}")
        print(f"  Key: {ssl_key_full_path}")
        sys.exit(1)
    
    hr_cmd.extend(["--ssl-certfile", ssl_cert_full_path, "--ssl-keyfile", ssl_key_full_path])
    it_cmd.extend(["--ssl-certfile", ssl_cert_full_path, "--ssl-keyfile", ssl_key_full_path])

if __name__ == '__main__':
    protocol = "https" if USE_HTTPS else "http"
    print(
        f"Starting HR on {protocol}://{HOST}:{HR_PORT} and IT on {protocol}://{HOST}:{IT_PORT}"
    )
    print(f"Root: {ROOT}, ALLOW_REMOTE: {ALLOW_REMOTE}, HTTPS: {USE_HTTPS}")
    
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = ROOT
    
    try:
        p_hr = subprocess.Popen(hr_cmd, cwd=ROOT, env=env)
        p_it = subprocess.Popen(it_cmd, cwd=ROOT, env=env)
        p_hr.wait()
        p_it.wait()
    except KeyboardInterrupt:
        for p in [p_hr, p_it]:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)
