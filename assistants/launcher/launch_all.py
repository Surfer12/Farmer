import subprocess
import sys
import os

# Default ports; can be overridden via env vars
HR_PORT = int(os.environ.get("HR_PORT", 8001))
IT_PORT = int(os.environ.get("IT_PORT", 8002))

# Compute repository root (three levels up from this launcher file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

hr_cmd = ["uvicorn", "assistants.hr_assistant.main:app", "--host", "0.0.0.0", "--port", str(HR_PORT)]
it_cmd = ["uvicorn", "assistants.it_assistant.main:app", "--host", "0.0.0.0", "--port", str(IT_PORT)]

if __name__ == '__main__':
    print(f"Starting HR on port {HR_PORT} and IT on port {IT_PORT} (root={ROOT})")
    try:
        p_hr = subprocess.Popen(hr_cmd, cwd=ROOT)
        p_it = subprocess.Popen(it_cmd, cwd=ROOT)
        p_hr.wait()
        p_it.wait()
    except KeyboardInterrupt:
        for p in [p_hr, p_it]:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)