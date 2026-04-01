"""
main.py — entry point for the Retail Inventory OpenEnv server.

Run from the Hackathon folder:
    python main.py

Or via uvicorn:
    uvicorn main:app --host 0.0.0.0 --port 7860
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # flat structure — app.py is in the same folder

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
