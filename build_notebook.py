"""Builds harness_colab.ipynb from harness.py, so the notebook and the local script never drift apart.

Shared logic (prompt, checks, runner, grader, feedback, loop, selftest) is copied out of harness.py by name;
only the Colab-specific cells (setup, Ollama, Secrets, paths, download) are written here.

Usage:  python build_notebook.py        -> writes harness_colab.ipynb
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = (ROOT / "harness.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def src(*names):
    """Source of top-level functions / assignments in harness.py, in the order given."""
    found = {}
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = node
    missing = [n for n in names if n not in found]
    assert not missing, f"not in harness.py: {missing}"
    return "\n\n".join(ast.get_source_segment(SRC, found[n]) for n in names)


cells = []
md = lambda s: cells.append(("markdown", s.strip("\n")))
code = lambda s: cells.append(("code", s.strip("\n")))

md(r"""
# Harness Engine: a small LLM writes a CIFAR-10 CNN, the harness runs it, grades it and loops

**How it works**
```
prompt ─► Qwen2.5-Coder-7B (local, Ollama) ─► attempt_N.py ─► static checks ─► run (subprocess, timeout)
   ▲                                                                                │
   └── feedback: root cause + epoch curve + lessons ◄── grader scores the saved model itself
```
The LLM runs **on this Colab GPU**, no API key needed. The harness never trusts the accuracy the script prints:
it loads `models/cifar10_cnn.keras` itself and grades it.

**Setup**
1. Runtime → Change runtime type → **T4 GPU** (required: the 7B model needs ~5 GB of GPU memory)
2. Runtime → **Run all**

Optional: set `LLM_BACKEND = "nim"` in Step 1 to use NVIDIA NIM instead (needs a Colab secret `NVIDIA_KEY`,
🔑 icon in the left sidebar, with notebook access on).

**Log tags:** `[SETUP] [DATA] [ATTEMPT] [PROMPT] [THINK] [LLM] [EXTRACT] [CODE] [CHECK] [EXEC] [VERIFY] [METRIC] [DECISION] [REPORT]`

*Generated from `harness.py` by `build_notebook.py`: edit those, not this notebook.*
""")

md(r"""
## Step 1 — Configuration
`qwen2.5-coder:7b` is Qwen2.5-Coder-7B-Instruct, 4-bit quantized (~4.7 GB download).
Other tags to try: `qwen2.5:7b` (general Qwen 7B), `qwen2.5-coder:3b` (weaker), `qwen2.5-coder:14b` (stronger, tight on a T4).
""")
code(r"""
LLM_BACKEND = "local"            # "local" = Ollama on this GPU, "nim" = NVIDIA NIM API
LOCAL_MODEL = "qwen2.5-coder:7b"
NIM_MODEL = "openai/gpt-oss-20b"
LLM_CONTEXT = 16384              # tokens; the feedback conversation grows every attempt
MODEL = LOCAL_MODEL if LLM_BACKEND == "local" else NIM_MODEL

""" + src("TARGET", "MAX_GAP", "MAX_ATTEMPTS", "MAX_LLM_CALLS", "MAX_CONSECUTIVE_FAILS", "MAX_TEMPERATURE",
          "TOTAL_BUDGET_MIN", "TRAIN_BUDGET_MIN", "RUN_TIMEOUT") + r"""
LLM_TIMEOUT = 900        # per LLM call (seconds)

import os
os.makedirs("/content/harness", exist_ok=True)
os.chdir("/content/harness")

from pathlib import Path
ROOT = Path("/content/harness")
LOG_DIR, GEN_DIR = ROOT / "logs", ROOT / "generated"
""" + src("MODEL_FILE", "BEST_MODEL", "DATA_FILE", "HF_BASE") + r"""
for d in (LOG_DIR, GEN_DIR, MODEL_FILE.parent, DATA_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)
""")

md("## Step 2 — Logging (console + `logs/run_<timestamp>.log`)")
code(r"""
import ast, builtins, io, json, logging, re, shutil, subprocess, symtable, sys, threading, time, urllib.request

log_file = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, force=True,  # force: Colab pre-installs root handlers
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("harness")
log.info("[SETUP] logging to %s", log_file)
""")

md(r"""
## Step 3 — Start the LLM
**local:** installs Ollama, starts its server (OpenAI-compatible API on `localhost:11434`), downloads Qwen 7B.
Takes ~2-4 min the first time. **nim:** just reads `NVIDIA_KEY` from Colab Secrets.
""")
code(r"""
!pip install -q openai pyarrow

def wait_for(url, secs=120):
    for _ in range(secs):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False

if LLM_BACKEND == "local":
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if gpu.returncode != 0:
        raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU, then run again.")
    log.info("[SETUP] GPU: %s", gpu.stdout.strip())

    !apt-get -qq install -y zstd > /dev/null
    !curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    # keep_alive=-1: model stays in GPU memory between attempts (~5 GB; the CNN gets the rest)
    ollama_env = {**os.environ, "OLLAMA_CONTEXT_LENGTH": str(LLM_CONTEXT), "OLLAMA_KEEP_ALIVE": "-1"}
    ollama_proc = subprocess.Popen(["ollama", "serve"], env=ollama_env,
                                   stdout=open(LOG_DIR / "ollama.log", "w"), stderr=subprocess.STDOUT)
    if not wait_for("http://localhost:11434"):
        raise SystemExit("Ollama server did not start, see logs/ollama.log")
    log.info("[SETUP] ollama server up, pulling %s ...", LOCAL_MODEL)
    t0 = time.time()
    !ollama pull {LOCAL_MODEL} 2>&1 | tail -n 1
    log.info("[SETUP] model ready in %.0fs", time.time() - t0)
    LLM_BASE_URL, LLM_KEY = "http://localhost:11434/v1", "ollama"  # Ollama ignores the key
else:
    from google.colab import userdata
    try:
        LLM_KEY = userdata.get("NVIDIA_KEY")
    except Exception as e:
        raise SystemExit("Add a Colab secret named NVIDIA_KEY (🔑 icon, left sidebar) and enable notebook access.") from e
    LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
    log.info("[SETUP] NVIDIA_KEY loaded: %s", bool(LLM_KEY))  # never log the key itself
""")

md(r"""
## Step 4 — The task prompt
Requirements only, no architecture hints: we want to see what the model comes up with.
""")
code(src("SYSTEM", "TASK") + "\nprint(TASK)")

md(r"""
## Step 5 — LLM client
Same OpenAI-compatible client for both backends. Logs the model's reasoning as `[THINK]`: the native reasoning
stream when the model has one (gpt-oss on NIM), otherwise the PLAN text Qwen writes before its code block.
""")
code(r"""
from openai import OpenAI

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_KEY, timeout=LLM_TIMEOUT)

""" + src("ask_llm") + r"""

# connectivity check
ask_llm([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=300)
""")

md("## Step 6 — Extract the code + cheap static checks (before spending minutes running it)")
code(src("extract_code", "undefined_names", "_loads", "_binds", "use_before_definition",
         "check_code", "normalize", "generate"))

md(r"""
## Step 7 — Dataset: build `data/cifar10_split.npz` once
`tf.keras.datasets.cifar10` downloads from cs.toronto.edu, which can throttle to ~40 KB/s,
so we build the same 60k images from the Hugging Face mirror (~30 s), pre-split into train / val / test.
""")
code(src("prepare_data") + "\n\nprepare_data()")

md(r"""
## Step 8 — Run the generated script (subprocess, live output, hard timeout)
`TF_FORCE_GPU_ALLOW_GROWTH`: TensorFlow otherwise grabs ALL GPU memory and crashes next to the loaded LLM.
""")
code(src("TF_ENV", "run_script", "parse_accuracy"))

md(r"""
## Step 9 — Grader: the harness scores the saved model itself
Test accuracy, train accuracy (overfitting gap) and a /255 probe, measured by the harness, not read from the
script's logs. Runs in a subprocess so TensorFlow never grabs GPU memory inside this notebook.
""")
code(src("GRADER", "verify_model"))

md(r"""
## Step 10 — Diagnosis + feedback: turn each result into a message the LLM can act on
- **Bugs a low score hides:** not learning (train accuracy at chance), pre-scaled input, claimed ≠ verified.
- **Underfit vs overfit:** a small gap with low train accuracy means the network is too small: the fix is *more*
  capacity, not more Dropout. Overfit advice leads with data augmentation and never says "shrink" while test
  accuracy is still below target.
- **Budget:** says how much of the compute budget went unused.
- **Plan ≠ code:** flags a PLAN that promises augmentation the script doesn't contain.
- The model sees the epoch curve (first 3 + last 5 epochs) with warning noise filtered out.
""")
code(src("NOISE", "clean", "epoch_curve", "error_line", "diagnose", "AUG", "TECHNIQUES", "techniques",
         "budget_note", "feedback"))

md("## Step 11 — Self-test (no API calls)")
code(src("selftest") + "\n\nselftest()")

md(r"""
## Step 12 — The loop
- A crashed run is still graded if its checkpoint exists.
- **Identical scripts are never re-trained:** a duplicate is sent straight back with "make a substantive change".
- **Stuck → more exploration:** temperature rises 0.2 → 0.5 → 0.8 → 1.0 with each attempt that doesn't beat the
  best, and after 2 such attempts the prompt opens with a PLATEAU notice asking for a structural change.
- **Best-so-far is kept:** `models/best.keras` + `generated/best.py`, even if the target is never reached.
- Short context: the task + only the last attempt + a one-line lesson (with test/train/params) per attempt.
""")
code(src("attempt_once", "main") + "\n\nresults = main()")

md("## Step 13 — Inspect and download the results")
code(r"""
from IPython.display import Markdown, display
best = GEN_DIR / "best.py"
if best.exists():
    display(Markdown("### Best script\n```python\n" + best.read_text() + "\n```"))

!cd /content/harness && zip -qr harness_results.zip generated logs models
from google.colab import files
files.download("/content/harness/harness_results.zip")
""")

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "cells": [
        {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if k == "code" else {})}
        for k, s in cells
    ],
}
out = ROOT / "harness_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out.name}: {len(cells)} cells")
