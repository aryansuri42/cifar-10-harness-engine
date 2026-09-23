"""Harness engine: an LLM writes a CIFAR-10 CNN, the harness runs it, grades it and loops.
Local twin of harness_colab.ipynb (same logic, same log tags).

Step 1: LLM client (NVIDIA NIM, or a local Ollama server) + logging.
Step 2: prompt -> generated code -> extract -> static checks -> save.
Step 3: run the script in a subprocess, stream its output into the log, hard timeout.
Step 4: VERIFY: the harness loads the saved model itself (in a subprocess) and measures test accuracy, train
        accuracy (overfitting gap) and a /255 probe. The script's own printed accuracy is only cross-checked.
Step 5: diagnose + feedback -> the LLM gets the root cause (crash / timeout / OOM / not learning / scaling bug /
        overfit / low accuracy) plus the epoch curve, and a one-line lesson per earlier attempt. Retry.
Step 6: final report.

Usage:  python harness.py              run the loop
        python harness.py --selftest   check checks/feedback logic, no API calls
"""
import ast
import builtins
import io
import json
import logging
import os
import re
import shutil
import subprocess
import symtable
import sys
import threading
import time
import urllib.request
from pathlib import Path

LLM_BACKEND = "nim"              # "nim" = NVIDIA NIM API (key NVIDIA_KEY in .env), "local" = Ollama on localhost
NIM_MODEL = "openai/gpt-oss-20b"  # weakest NIM chat model that was live on 2026-09-22
LOCAL_MODEL = "qwen2.5-coder:7b"
MODEL = NIM_MODEL if LLM_BACKEND == "nim" else LOCAL_MODEL

TARGET = 0.95             # verified test accuracy needed to pass
MAX_GAP = 0.15            # max verified (train acc - test acc) AT the target; blocks memorised passes
MAX_ATTEMPTS = 25         # GRADED experiments; crashes and rejected scripts do not consume these
MAX_LLM_CALLS = 60        # hard stop on generations, so a crash loop cannot run forever
MAX_CONSECUTIVE_FAILS = 8 # give up if this many attempts in a row never reach the grader
MAX_TEMPERATURE = 0.7     # above this a 7B model writes broken code instead of new ideas
TOTAL_BUDGET_MIN = 360    # stop starting new attempts after this much wall clock
TRAIN_BUDGET_MIN = 45     # what we tell the LLM
RUN_TIMEOUT = 60 * 60     # hard kill per training run (seconds)
LLM_TIMEOUT = 900         # gpt-oss-20b on NIM can take 5-15 min per generation

ROOT = Path(__file__).parent
LOG_DIR, GEN_DIR = ROOT / "logs", ROOT / "generated"
MODEL_FILE = ROOT / "models" / "cifar10_cnn.keras"
BEST_MODEL = ROOT / "models" / "best.keras"        # best verified model across attempts (kept even on failure)
DATA_FILE = ROOT / "data" / "cifar10_split.npz"  # pre-split train / val / test
# tf.keras.datasets.cifar10 downloads from cs.toronto.edu, which throttles to ~40 KB/s (70+ min),
# so we build the same dataset once from the Hugging Face mirror.
HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
for d in (LOG_DIR, GEN_DIR, MODEL_FILE.parent, DATA_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("harness")

# ---------------------------------------------------------------- prompt

SYSTEM = (
    "You are a machine-learning researcher who iterates empirically, not a tutorial writer. Each turn you receive "
    "measurements from your previous script (verified test and train accuracy, the epoch curve, the time used, the "
    "techniques already tried) and you run the next experiment.\n"
    "Answer in exactly this shape:\n"
    "  Root cause: what the numbers say about the last script - one or two sentences, referring to specific values.\n"
    "  Hypothesis: the ONE thing that is limiting accuracy now, and why.\n"
    "  Change: what you are changing, how big the change is, and how many accuracy points you expect from it.\n"
    "  Budget: how many epochs and roughly how many minutes that will take.\n"
    "Then one complete, runnable Python script in a single ```python code block, and nothing after it.\n"
    "Write the code your plan describes: if the plan says augmentation, the script must contain augmentation layers.")

TASK = f"""Train a convolutional neural network on CIFAR-10 and reach {TARGET:.0%} test accuracy.

This is a research loop, not a one-shot exercise: you get up to {MAX_ATTEMPTS} attempts, each one measured by an
independent grader, and each result comes back to you. Treat every script as an experiment.

WHAT SUCCESS MEANS
- The grader loads your saved model and measures accuracy on 10000 held-out test images: it must be >= {TARGET:.2f}.
- It also measures accuracy on training images. At the target, (train - test) must be <= {MAX_GAP:.2f}, so a model
  that memorises the training set does not pass.
- A plain 2-3 conv-layer CNN plateaus around 0.75-0.80 no matter how you tune it. Reaching {TARGET:.0%} needs a
  fundamentally stronger approach, and you are expected to search for one.

HARD CONSTRAINTS (checked before your script is run; violations are rejected without training)
1. TensorFlow 2.x / tf.keras only. No PyTorch, no other frameworks.
2. Data comes ONLY from the local file "data/cifar10_split.npz", already split, used exactly as-is:
     d = np.load("data/cifar10_split.npz")
     x_train, y_train = d["x_train"], d["y_train"]   # 45000 images
     x_val,   y_val   = d["x_val"],   d["y_val"]     # 5000 images, for validation
     x_test,  y_test  = d["x_test"],  d["y_test"]    # 10000 images, final evaluation ONLY
   x_* are uint8 (N, 32, 32, 3) with pixels 0..255; y_* are int64 (N,) labels. Do not download anything.
3. ALL pixel preprocessing happens INSIDE the model, starting with one tf.keras.layers.Rescaling (or a
   Normalization layer). Feed the RAW uint8 arrays to fit/evaluate. Any "/ 255" outside the model is rejected:
   the grader calls model.predict(x_test) on raw 0..255 pixels, so a model expecting pre-scaled input scores 10%.
4. Validate with validation_data=(x_val, y_val). NEVER touch x_test / y_test for training, validation, early
   stopping or checkpoint selection - only for the single final evaluation.
5. Save DURING training so a crash never loses the model, to EXACTLY this path - the grader reads this one file
   and nothing else, whatever your architecture is called:
     tf.keras.callbacks.ModelCheckpoint("models/cifar10_cnn.keras", monitor="val_accuracy", save_best_only=True)
   Do not rename it (not cifar10_resnet.keras, not best_model.keras) and do not overwrite it after training.
   Create the models directory first.
6. Print one line per epoch (model.fit(..., verbose=2)), no per-batch progress bars.
7. The LAST printed line must be exactly: FINAL_TEST_ACCURACY=<float 0..1, 4 decimals>, e.g. FINAL_TEST_ACCURACY=0.9512
8. Guard the entry point with if __name__ == "__main__":
9. The saved model must load with tf.keras.models.load_model and use built-in Keras layers only: no Lambda layers
   and no custom Layer/Model subclasses. (Custom Callbacks are fine - they are not part of the saved model.)
10. Training must finish within {TRAIN_BUDGET_MIN} minutes; the process is killed at {RUN_TIMEOUT // 60} minutes.
    A GPU may or may not be available - the script must work either way.

WHAT YOU ARE FREE TO DO - use it
Everything below is allowed and encouraged. Do not stay with the textbook Conv-Pool-Conv-Pool-Dense model.
- Architecture: any depth and width; residual / skip connections built with the functional API (layers.Add);
  bottleneck or wide-residual blocks; separable or dilated convolutions; squeeze-and-excitation; strided convs
  instead of pooling; GlobalAveragePooling2D instead of Flatten; a small stem followed by 3-4 stages.
- Transfer learning: tf.keras.applications backbones with weights="imagenet" (internet is available). Put a
  tf.keras.layers.Resizing inside the model to bring 32x32 up to the backbone's size, keep Rescaling/Normalization
  inside the model too, and fine-tune the upper blocks. This is usually the fastest route past 0.90.
- Augmentation inside the model: RandomFlip("horizontal"), RandomTranslation, RandomZoom, RandomRotation,
  RandomContrast, RandomCrop after padding. Stronger schemes (cutout, mixup, cutmix) may be done in a tf.data
  pipeline on the training set - but never change what the model itself expects at inference.
- Training recipe: SGD with momentum/nesterov or AdamW; cosine decay, warmup, one-cycle or ReduceLROnPlateau;
  label smoothing; weight decay; gradient clipping; batch sizes from 64 to 512; 50-200 epochs if time allows.
- Speed: tf.data with cache().shuffle().batch().prefetch(), and mixed precision
  (tf.keras.mixed_precision.set_global_policy("mixed_float16")) on GPU - if you use it, give the final Dense
  layer dtype="float32". Faster epochs mean more epochs inside the budget.
- Time control: a small custom Callback that stops training when the budget is nearly spent is a good idea.

HOW TO ITERATE
- Read the numbers you are given before changing anything: train vs test accuracy says overfitting or
  underfitting; the epoch curve says whether learning stalled, diverged, or was still improving when it stopped;
  the time used says how much room is left.
- Make each change big enough to move accuracy by at least a point. Renaming, reformatting or re-tuning one
  hyper-parameter by 10% is a wasted attempt.
- Use the budget: if the last run took 5 of {TRAIN_BUDGET_MIN} minutes, train much longer or much bigger.
- Build on the best script so far. Do not fall back to a smaller model that already scored worse.
- If a family of changes has failed twice, switch family: architecture -> training recipe -> transfer learning.

Return your reasoning in the required shape, then the full script in a single ```python code block."""

# ---------------------------------------------------------------- LLM

client = None


def make_client():
    from openai import OpenAI
    if LLM_BACKEND == "local":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=LLM_TIMEOUT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("NVIDIA_KEY")
    if not api_key:
        sys.exit("NVIDIA_KEY not set in .env")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=LLM_TIMEOUT)


def ask_llm(messages, max_tokens=8192, temperature=0.2):
    """Logs the reasoning as [THINK]: the native stream when the model has one (gpt-oss on NIM),
    otherwise the PLAN text written before the code block (Qwen)."""
    log.info("[PROMPT] (temperature %.1f) %s", temperature, messages[-1]["content"][:1500])
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    msg, finish, u = resp.choices[0].message, resp.choices[0].finish_reason, resp.usage
    log.info("[LLM] %.1fs | tokens in=%s out=%s | finish=%s", time.time() - t0, u.prompt_tokens, u.completion_tokens, finish)
    if finish == "length":
        log.warning("[LLM] response TRUNCATED at token limit - code may be incomplete")
    text = msg.content or ""
    thinking = (msg.model_extra or {}).get("reasoning_content") or re.split(r"```", text)[0].strip()
    if thinking:
        log.info("[THINK] model reasoning:\n%s", thinking)
    log.info("[LLM] response:\n%s", text)
    return text

# ---------------------------------------------------------------- code extraction + static checks


def extract_code(text):
    """Longest ```python block; fall back to the raw text if the model skipped fences."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        log.warning("[EXTRACT] no code fence found, using raw response")
        return text.strip()
    log.info("[EXTRACT] %d code block(s), taking the longest", len(blocks))
    return max(blocks, key=len).strip()


def undefined_names(code):
    """Names the script reads but never defines or imports - a NameError waiting to happen.
    Catches the commonest crash from a small model: calling a helper it forgot to write."""
    try:
        table = symtable.symtable(code, "generated", "exec")
    except SyntaxError:
        return []
    known = set(table.get_identifiers())
    missing = set()

    def visit(t):
        for sym in t.get_symbols():
            name = sym.get_name()
            if name in known or hasattr(builtins, name):
                continue
            if t is table:  # module scope: referenced but never bound anywhere
                if sym.is_referenced() and not (sym.is_assigned() or sym.is_imported()):
                    missing.add(name)
            elif sym.is_global() and not sym.is_assigned():  # inside a function, resolves to module scope
                missing.add(name)
        for child in t.get_children():
            visit(child)

    visit(table)
    return sorted(missing)


def _loads(node, local=frozenset(), root=True):
    """Name reads inside one statement, skipping deferred bodies (functions, lambdas, classes) and
    names bound by an enclosing comprehension (`[layer for layer in ...]` binds its own `layer`)."""
    if not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        return []
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        local = local | {n.id for gen in node.generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
    out = [node] if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local else []
    for child in ast.iter_child_nodes(node):
        out += _loads(child, local, root=False)
    return out


def _binds(node):
    """Names one statement binds: assignments, imports, defs, loop targets, except-as, with-as."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.alias):
            names.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
    return names


def use_before_definition(code):
    """Module-level names read before they are ever bound, e.g.
        model = Sequential([residual_block(model.layers[-1].output, 64), ...])
    Function bodies are skipped: they run later, so order does not matter there."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    bound, problems = set(), []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound |= _binds(stmt)
            continue
        # A compound statement binds its own targets before its body runs (`for layer in ...: layer.trainable`),
        # so only a simple statement's own expression can read a name too early (`model = f(model.layers)`).
        visible = bound if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return)) \
            else bound | _binds(stmt)
        for node in sorted(_loads(stmt), key=lambda n: (n.lineno, n.col_offset)):
            if node.id not in visible and not hasattr(builtins, node.id):
                problems.append(f"'{node.id}' is used on line {node.lineno} before it is defined - the script would "
                                f"crash with NameError before training starts")
                bound.add(node.id)  # report each name once
        bound |= _binds(stmt)
    return problems


def check_code(code):
    """Cheap static checks before we spend minutes running it. Returns a list of problems."""
    problems = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        problems.append(f"SyntaxError line {e.lineno}: {e.msg}")
    if "FINAL_TEST_ACCURACY" not in code:
        problems.append("script never prints FINAL_TEST_ACCURACY")
    if "Conv2D" not in code and "applications" not in code:
        problems.append("no Conv2D layer and no tf.keras.applications backbone - not a CNN")
    if "cifar10_split.npz" not in code:
        problems.append("does not load data/cifar10_split.npz")
    if "ModelCheckpoint" not in code:
        problems.append("no ModelCheckpoint callback - model must be saved during training")
    elif "models/cifar10_cnn.keras" not in code:
        paths = re.findall(r"ModelCheckpoint\(\s*[\"']([^\"']+)[\"']", code) or re.findall(r"[\w/]+\.keras", code)
        problems.append('the checkpoint path must be EXACTLY "models/cifar10_cnn.keras" - the grader reads only that '
                        "file and ignores any other name" + (f" (yours: {', '.join(paths)})" if paths else ""))
    if re.search(r"validation_data\s*=\s*\(\s*x_test", code):
        problems.append("uses the TEST set as validation_data - use validation_data=(x_val, y_val) (test leakage)")
    if "Rescaling(" not in code and "Normalization(" not in code:
        problems.append("no Rescaling / Normalization layer inside the model - it would receive raw 0..255 pixels")
    # "/ 255" anywhere except inside a Rescaling(...) / preprocessing call = pixels scaled outside the model too
    outside = [l.strip() for l in re.sub(r"(Rescaling|Normalization|Resizing)\([^)]*\)", "", code).splitlines()
               if re.search(r"/\s*255", l)]
    if outside:
        problems.append("pixels are divided by 255 OUTSIDE the model, and the Rescaling layer divides again, so the "
                        "model gets near-zero inputs and learns nothing. Delete these lines and feed the raw uint8 "
                        "arrays to fit/evaluate: " + " | ".join(outside))
    for name in undefined_names(code):
        problems.append(f"'{name}' is used but never defined or imported - the script would crash with NameError")
    problems += use_before_definition(code)
    if re.search(r"SparseCategoricalCrossentropy\([^)]*label_smoothing", code):
        problems.append("SparseCategoricalCrossentropy does not accept label_smoothing - either keep sparse labels "
                        "without smoothing, or one-hot the labels and use CategoricalCrossentropy(label_smoothing=...)")
    if "Lambda(" in code:
        problems.append("uses a Lambda layer - saved model must use built-in layers only")
    if re.search(r"^\s*(import|from)\s+torch", code, re.M):
        problems.append("imports PyTorch - must be TensorFlow only")
    return problems


def generate(attempt, messages, temperature=0.2):
    reply = ask_llm(messages, temperature=temperature)
    code = extract_code(reply)
    path = GEN_DIR / f"attempt_{attempt}.py"
    path.write_text(code, encoding="utf-8")
    log.info("[CODE] saved %s (%d lines)", path.relative_to(ROOT).as_posix(), code.count("\n") + 1)
    problems = check_code(code)
    for p in problems:
        log.warning("[CHECK] %s", p)
    if not problems:
        log.info("[CHECK] static checks passed")
    return reply, path, problems

# ---------------------------------------------------------------- dataset


def prepare_data():
    """Build data/cifar10_split.npz once from the HF parquet mirror: train 45k / val 5k / test 10k.
    Pre-split because array slicing is where small models kept breaking things (NameError on y_val,
    45000-vs-50000 shape mismatch, labels shifted against images)."""
    if DATA_FILE.exists():
        log.info("[DATA] using cached %s", DATA_FILE.relative_to(ROOT).as_posix())
        return
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    out = {}
    for split in ("train", "test"):
        url = f"{HF_BASE}{split}-00000-of-00001.parquet"
        log.info("[DATA] downloading %s", url)
        with urllib.request.urlopen(url, timeout=300) as resp:
            table = pq.read_table(io.BytesIO(resp.read())).to_pydict()
        out[f"x_{split}"] = np.stack([np.array(Image.open(io.BytesIO(i["bytes"])).convert("RGB")) for i in table["img"]])
        out[f"y_{split}"] = np.array(table["label"], dtype=np.int64)
        log.info("[DATA] %s: x=%s y=%s", split, out[f"x_{split}"].shape, out[f"y_{split}"].shape)
    assert out["x_train"].shape == (50000, 32, 32, 3) and out["x_test"].shape == (10000, 32, 32, 3)
    assert (np.bincount(out["y_test"]) == 1000).all()
    out["x_val"], out["y_val"] = out["x_train"][45000:], out["y_train"][45000:]
    out["x_train"], out["y_train"] = out["x_train"][:45000], out["y_train"][:45000]
    log.info("[DATA] split: train=%d val=%d test=%d", len(out["x_train"]), len(out["x_val"]), len(out["x_test"]))
    np.savez(DATA_FILE, **out)
    log.info("[DATA] saved %s (%.0f MB)", DATA_FILE.relative_to(ROOT).as_posix(), DATA_FILE.stat().st_size / 2**20)

# ---------------------------------------------------------------- run


# Unbuffered so epoch lines arrive live; TF C++ noise down; allow-growth so TF can share a GPU with a local LLM.
TF_ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
          "PYTHONIOENCODING": "utf-8"}


def run_script(path):
    """Run generated script, stream each output line into the log. Returns (exit_code, lines, seconds, timed_out)."""
    log.info("[EXEC] running %s (hard timeout %d min)", path.relative_to(ROOT).as_posix(), RUN_TIMEOUT // 60)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, str(path)], cwd=ROOT, env=TF_ENV, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timer = threading.Timer(RUN_TIMEOUT, proc.kill)
    timer.start()
    lines = []
    for line in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip()  # strip ANSI colors
        if line and "━" not in line:                          # skip Keras progress-bar redraws
            lines.append(line)
            log.info("[EXEC] | %s", line)
    proc.wait()
    timed_out = not timer.is_alive() and proc.returncode != 0
    timer.cancel()
    secs = time.time() - t0
    if timed_out:
        log.error("[EXEC] KILLED after %d min timeout", RUN_TIMEOUT // 60)
    log.info("[EXEC] exit code %s in %.1f min", proc.returncode, secs / 60)
    return proc.returncode, lines, secs, timed_out


def parse_accuracy(lines):
    """Last FINAL_TEST_ACCURACY=x in the output, as a float in [0, 1], else None."""
    for line in reversed(lines):
        m = re.search(r"FINAL_TEST_ACCURACY=([0-9]*\.?[0-9]+)", line)
        if m:
            acc = float(m.group(1))
            return acc if 0 <= acc <= 1 else None
    return None

# ---------------------------------------------------------------- grader

# Runs in a subprocess: keeps TensorFlow (RAM, GPU memory) out of the harness process.
GRADER = r"""
import json, sys, numpy as np, tensorflow as tf
def score(m, x, y):
    s = m.predict(x.astype("float32"), batch_size=500, verbose=0)
    if s.shape != (len(x), 10):
        raise ValueError(f"model output shape {s.shape}, expected ({len(x)}, 10)")
    return float(np.mean(s.argmax(1) == y))
try:
    m = tf.keras.models.load_model(sys.argv[1])
    d = np.load(sys.argv[2])
    print("GRADE=" + json.dumps({"acc": score(m, d["x_test"], d["y_test"]),
                                 "train_acc": score(m, d["x_train"][:10000], d["y_train"][:10000]),
                                 # diagnostic probe: does the model secretly expect pixels already divided by 255?
                                 "probe_acc": score(m, d["x_test"][:2000] / 255.0, d["y_test"][:2000]),
                                 "params": int(m.count_params())}))
except Exception as e:
    print("GRADE=" + json.dumps({"error": f"{type(e).__name__}: {e}"}))
"""


def verify_model():
    """Returns dict with acc, train_acc, probe_acc, params - or error.
    If the script saved under a different name, grade that file anyway (a whole training run is too expensive to
    throw away over a filename) and report it, so the feedback can tell the model to fix the path."""
    target, wrong_path = MODEL_FILE, None
    if not target.exists():
        others = [f for f in MODEL_FILE.parent.glob("*.keras") if f != BEST_MODEL]
        if len(others) != 1:
            return {"error": "no model file at models/cifar10_cnn.keras"
                             + (f" (found instead: {', '.join(f.name for f in others)})" if others else "")}
        target, wrong_path = others[0], others[0].name
        log.warning("[VERIFY] models/cifar10_cnn.keras is missing; grading %s instead", wrong_path)
    out = subprocess.run([sys.executable, "-c", GRADER, str(target), str(DATA_FILE)], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         env={**TF_ENV, "TF_CPP_MIN_LOG_LEVEL": "3"}, timeout=600).stdout
    m = re.search(r"GRADE=(.*)", out)
    g = json.loads(m.group(1)) if m else {"error": "grader crashed: " + out[-500:]}
    if "error" not in g:
        g["graded_file"] = str(target)  # may not be MODEL_FILE: see wrong_path above
    if wrong_path:
        g["wrong_path"] = wrong_path
    return g

# ---------------------------------------------------------------- diagnosis + feedback

NOISE = re.compile(r"UserWarning|warnings\.warn|absl::|^[IWE]\d{4} |XLA|oneDNN|TensorFlow GPU support|cuda|computation placer")


def clean(lines):
    return [l for l in lines if not NOISE.search(l)]


def epoch_curve(lines, head=3, tail=5):
    """'Epoch k/N' header + its metrics line -> one line each; first `head` and last `tail` epochs."""
    ep = [f"{h}: {m}" for h, m in zip(lines, lines[1:]) if re.match(r"Epoch \d+/\d+$", h) and "loss" in m]
    if len(ep) <= head + tail:
        return "\n".join(ep)
    return "\n".join(ep[:head] + [f"... ({len(ep) - head - tail} epochs omitted) ..."] + ep[-tail:])


def error_line(lines):
    """Last 'SomethingError: message' line of a traceback, trimmed for the lesson list."""
    errs = [l.strip() for l in lines if re.match(r"\s*\w*(Error|Exception)\b", l)]
    return errs[-1][:160] if errs else None


def diagnose(r):
    """(short tag, explanation) for systematic bugs a low score alone hides, or (None, None).
    Order: not learning -> pre-scaled input -> claimed != verified."""
    acc, tr, probe, claimed = r.get("acc"), r.get("train_acc"), r.get("probe_acc"), r.get("claimed")
    if acc is None:
        return None, None
    if tr is not None and tr < 0.20:
        return "NOT LEARNING", (
            f"NOT LEARNING (found by the grader): accuracy on TRAINING images is only {tr:.4f}, i.e. random guessing "
            f"(10 classes = 0.10; a training loss stuck near ln(10) = 2.3026 means the same). The model learned "
            f"NOTHING. This is a BUG in the data pipeline or training setup, NOT a capacity problem: do NOT make the "
            f"network bigger. Check, in this order:\n"
            f"  1. Input scale: raw uint8 pixels go into the model, ONE Rescaling(1./255) layer inside it, no other "
            f"division or normalization anywhere.\n"
            f"  2. Labels match images: fit on (x_train, y_train), validate on (x_val, y_val), exactly as loaded.\n"
            f"  3. Loss matches the output layer: Dense(10) without softmax -> from_logits=True; "
            f"with softmax -> from_logits=False.\n"
            f"  4. Learning rate: Adam around 1e-3 (1e-4 with decay is too small to get started).")
    if probe is not None and probe > acc + 0.15:
        return "preprocessing bug", (
            f"PREPROCESSING BUG (found by the grader): on RAW 0..255 pixels your saved model scores {acc:.4f}, but on "
            f"pixels divided by 255 it scores {probe:.4f}. So your saved model expects inputs that were already scaled "
            f"outside the model, while the grader feeds RAW pixels. Fix: do NOT divide or normalize x_train / x_val / "
            f"x_test in the script. Put tf.keras.layers.Rescaling(1./255) as the first layer INSIDE the model and pass "
            f"the raw uint8 arrays to model.fit and model.evaluate. The network itself may be fine.")
    if claimed is not None and abs(claimed - acc) > 0.05:
        return "claimed/verified mismatch", (
            f"MISMATCH: your script printed FINAL_TEST_ACCURACY={claimed:.4f} but the grader measured {acc:.4f} on the "
            f"saved model with RAW 0..255 test pixels. Your script evaluates on differently preprocessed data than the "
            f"grader. All preprocessing must happen INSIDE the saved model; feed raw pixels to fit and evaluate.")
    return None, None


AUG = re.compile(r"Random(Flip|Translation|Rotation|Crop|Zoom|Contrast)")

# Techniques the harness can recognise in a script. The ledger of what has and has not been tried goes into every
# feedback message: a small model otherwise keeps re-tuning the same two knobs.
TECHNIQUES = {
    "data augmentation": AUG,
    "cutout/mixup/cutmix": re.compile(r"cutout|mixup|cutmix", re.I),
    "batch normalization": re.compile(r"BatchNormalization"),
    "dropout": re.compile(r"Dropout\("),
    "weight decay / L2": re.compile(r"weight_decay|regularizers\.l2|kernel_regularizer"),
    "label smoothing": re.compile(r"label_smoothing"),
    "residual / skip connections": re.compile(r"layers\.Add|Add\(\)|add\(\[|residual|ResNet", re.I),
    "pretrained backbone": re.compile(r"applications\.|weights\s*=\s*[\"']imagenet[\"']"),
    "LR schedule": re.compile(r"CosineDecay|LearningRateScheduler|ReduceLROnPlateau|ExponentialDecay|PiecewiseConstant|warmup|OneCycle", re.I),
    "SGD + momentum": re.compile(r"SGD\("),
    "AdamW": re.compile(r"AdamW"),
    "mixed precision": re.compile(r"mixed_float16|set_global_policy"),
    "tf.data pipeline": re.compile(r"tf\.data|from_tensor_slices"),
    "global average pooling": re.compile(r"GlobalAveragePooling2D"),
    "early stopping": re.compile(r"EarlyStopping"),
    "separable / dilated conv": re.compile(r"SeparableConv2D|dilation_rate"),
}


def techniques(code):
    """Which known techniques a script uses (best-effort, by pattern)."""
    return {name for name, rx in TECHNIQUES.items() if rx.search(code or "")}


def budget_note(r):
    """Tell the model how much of the compute budget it left on the table."""
    used = r.get("secs", 0) / 60
    if 0 < used < TRAIN_BUDGET_MIN / 3:
        return (f"\nCompute: training used only {used:.1f} of the {TRAIN_BUDGET_MIN} minute budget, so roughly "
                f"{TRAIN_BUDGET_MIN / used:.0f}x more compute is available for a bigger network or longer training "
                f"(more epochs, larger EarlyStopping patience).")
    return ""


def feedback(r):
    """Build the message telling the LLM what went wrong. r = result dict of one attempt. Returns (status, message)."""
    lines = clean(r.get("lines", []))
    tail = "\n".join(lines[-25:])
    curve = epoch_curve(lines) or tail
    acc = r.get("acc")
    err = error_line(lines)
    tag, why = diagnose(r)
    if r.get("problems"):
        status = "static check failed: " + r["problems"][0][:120]
        body = "Your script failed static checks, it was NOT run:\n- " + "\n- ".join(r["problems"])
    elif r.get("duplicate_of"):
        k = r["duplicate_of"]
        status = f"duplicate of attempt {k}"
        body = (f"Your script is IDENTICAL to your script from attempt {k} "
                f"({r.get('duplicate_summary', 'see the results list below')}). It was NOT run: running the same code "
                f"again cannot give a different result. Make a SUBSTANTIVE change to the network or the training "
                f"(not comments or formatting).")
    elif r.get("timed_out"):
        status = "timeout"
        body = (f"Your script was KILLED after {RUN_TIMEOUT // 60} minutes, it is too slow. Epochs before the kill:\n"
                f"```\n{curve}\n```\nMake training cheaper so it finishes within {TRAIN_BUDGET_MIN} minutes.")
    elif r.get("exit_code") != 0 and not err:
        status = f"killed externally (exit {r.get('exit_code')})"
        body = (f"Your script died with exit code {r.get('exit_code')} and NO Python traceback - it was most likely killed "
                f"by the operating system for using too much memory (RAM).\nOutput before it died:\n```\n{tail}\n```\n"
                f"Reduce memory use: e.g. feed data with tf.data from uint8 arrays instead of float copies, "
                f"smaller batches, a smaller model.")
    elif r.get("exit_code") != 0 and acc is None:
        status = f"crashed: {err}"
        body = f"Your script crashed with exit code {r.get('exit_code')}. Traceback:\n```\n{tail}\n```"
    elif r.get("eval_error"):
        status = "grader failed: " + r["eval_error"][:120]
        body = (f"The grader could not use the saved model: {r['eval_error']}\nThe saved model must load with "
                f"tf.keras.models.load_model, accept raw 0..255 images of shape (N, 32, 32, 3) and output 10 scores.\n"
                f"Epochs:\n```\n{curve}\n```")
    elif r.get("gap", 0) > MAX_GAP and acc >= TARGET:
        tr = r["train_acc"]
        status = f"overfit at target (train {tr:.3f} / test {acc:.3f})"
        body = (f"Test accuracy {acc:.4f} REACHES the {TARGET} target, but train accuracy is {tr:.4f}: a gap of "
                f"{r['gap']:.4f}, above the allowed {MAX_GAP:.2f}, so it does not pass. Keep this accuracy and close "
                f"the gap: stronger augmentation (cutout / mixup), label smoothing, weight decay, or stopping earlier "
                f"on val_accuracy.\nEpochs:\n```\n{curve}\n```" + budget_note(r))
    else:
        tr, gap = r.get("train_acc"), r.get("gap", 0)
        status = f"below target ({acc:.4f})"
        head = f"The grader measured test accuracy {acc:.4f} on your saved model; the target is {TARGET}. "
        if tr is not None and not tag:
            if gap > MAX_GAP:
                status += " + overfit"
                head += (f"Train accuracy is {tr:.4f}, a gap of {gap:.4f}: the network CAN fit the training data but "
                         f"does not generalise. The lever is stronger regularisation WITHOUT losing capacity: data "
                         f"augmentation inside the model (RandomFlip, RandomTranslation, RandomZoom), cutout or mixup "
                         f"in a tf.data pipeline, label smoothing, weight decay. Do NOT shrink the network.")
            elif tr < TARGET:
                status += " + underfit"
                head += (f"Train accuracy is only {tr:.4f}, so the model cannot fit even the TRAINING data: it is too "
                         f"small or trained too briefly, and this is NOT overfitting. Do NOT add more Dropout. Add "
                         f"capacity (deeper / wider, residual blocks, or an ImageNet-pretrained backbone) and train "
                         f"longer.")
            else:
                head += (f"Train accuracy is {tr:.4f} with a healthy gap of {gap:.4f}: the recipe is sound but not "
                         f"strong enough. A bigger jump is needed - a different architecture family or a pretrained "
                         f"backbone, not another small tweak.")
        body = head + f"\nEpochs:\n```\n{curve}\n```" + budget_note(r)
    if r.get("wrong_path"):
        body = (f"WRONG CHECKPOINT PATH: you saved to models/{r['wrong_path']}, but the grader reads ONLY "
                f"models/cifar10_cnn.keras. It graded your file this time; next time that is a failed attempt. "
                f'Use ModelCheckpoint("models/cifar10_cnn.keras", ...) exactly.\n\n') + body
    if "augment" in r.get("plan", "").lower() and r.get("code") and not AUG.search(r["code"]):
        body = ("NOTE: your PLAN said you would add data augmentation, but the script contains NO augmentation layer "
                "(RandomFlip / RandomTranslation / ...). Write the code your plan describes.\n\n") + body
    if tag:  # the systematic bug goes FIRST: it is the thing to fix
        status, body = f"{status} + {tag}", f"{why}\n\n{body}"
    return status, body + (
        "\n\nStart your PLAN with 'Root cause:' naming the exact line(s) of your previous script that caused this "
        "result and why. Then fix them. Keep all the original hard requirements. "
        "Return the full script in a single ```python code block.")

# ---------------------------------------------------------------- loop


def normalize(code):
    """Code with comments and all whitespace removed: two scripts that differ only in those are duplicates."""
    return re.sub(r"\s+", "", re.sub(r"#[^\n]*", "", code))


def attempt_once(attempt, messages, temperature, seen):
    """One generate -> run -> verify cycle. A crashed run is still graded if its checkpoint exists.
    seen: normalized code -> attempt number, so an identical script is never trained twice."""
    reply, path, problems = generate(attempt, messages, temperature)
    code = path.read_text(encoding="utf-8")
    r = {"problems": problems, "code": code, "plan": reply.split("```")[0]}
    if problems:
        return reply, r
    key = normalize(code)
    if key in seen:
        r["duplicate_of"] = seen[key]
        log.warning("[CHECK] script is IDENTICAL to attempt %d - not running it", seen[key])
        return reply, r
    seen[key] = attempt
    for stale in MODEL_FILE.parent.glob("*.keras"):  # no file from an earlier attempt may be graded
        if stale != BEST_MODEL:
            stale.unlink()
    r["exit_code"], r["lines"], r["secs"], r["timed_out"] = run_script(path)
    claimed = parse_accuracy(r["lines"])
    log.info("[VERIFY] grading %s on 10k test + first 10k train images", MODEL_FILE.relative_to(ROOT).as_posix())
    g = verify_model()
    if g.get("error"):
        r["eval_error"] = g["error"]
        log.warning("[VERIFY] %s", r["eval_error"])
        return reply, r
    r["acc"], r["train_acc"], r["probe_acc"], r["claimed"] = g["acc"], g["train_acc"], g["probe_acc"], claimed
    r["gap"], r["params"], r["wrong_path"] = r["train_acc"] - r["acc"], g["params"], g.get("wrong_path")
    r["graded_file"] = g["graded_file"]
    log.info("[METRIC] test_acc=%.4f | train_acc=%.4f | gap=%.4f (max %.2f) | claimed=%s | probe(/255)=%.4f | "
             "params=%s | train=%.1f min | exit=%s", r["acc"], r["train_acc"], r["gap"], MAX_GAP,
             f"{claimed:.4f}" if claimed is not None else "none", r["probe_acc"], f"{g['params']:,}",
             r["secs"] / 60, r["exit_code"])
    if r["gap"] > MAX_GAP:
        log.warning("[VERIFY] OVERFIT: train %.4f vs test %.4f", r["train_acc"], r["acc"])
    tag, _ = diagnose(r)
    if tag:
        log.warning("[VERIFY] DIAGNOSIS: %s (raw=%.4f, /255=%.4f, claimed=%s)", tag, r["acc"], r["probe_acc"],
                    f"{claimed:.4f}" if claimed is not None else "none")
    return reply, r


def main():
    prepare_data()
    log.info("[SETUP] backend=%s model=%s task=CIFAR-10 target=%.2f max_gap=%.2f max_attempts=%d train_budget=%dmin "
             "kill=%dmin total_budget=%dmin log=%s", LLM_BACKEND, MODEL, TARGET, MAX_GAP, MAX_ATTEMPTS,
             TRAIN_BUDGET_MIN, RUN_TIMEOUT // 60, TOTAL_BUDGET_MIN, log_file.name)
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    messages = base
    results, lessons, seen, t_start = [], [], {}, time.time()
    best_acc, best_attempt, best_reply, stall = -1.0, None, None, 0  # stall = experiments in a row without a new best
    tried, errors = set(), {}   # techniques seen anywhere; how often each crash message has been seen
    experiments = attempt = fails = 0  # experiments = attempts that produced a measured accuracy
    broken_last = False         # last attempt crashed or was rejected: ask for precision, not creativity

    while experiments < MAX_ATTEMPTS and attempt < MAX_LLM_CALLS:
        if (time.time() - t_start) / 60 > TOTAL_BUDGET_MIN:
            log.warning("[DECISION] total budget of %d min spent - stopping", TOTAL_BUDGET_MIN)
            break
        attempt += 1
        # Stuck on a working script -> sample more freely. But after broken code, go back to the most
        # deterministic setting: high temperature is what turns a 7B model's ResNet into a NameError.
        temperature = 0.2 if broken_last else min(0.2 + 0.2 * stall, MAX_TEMPERATURE)
        log.info("=" * 70)
        log.info("[ATTEMPT %d] experiment %d/%d | elapsed %.1f min | temperature %.1f", attempt, experiments + 1,
                 MAX_ATTEMPTS, (time.time() - t_start) / 60, temperature)
        try:
            reply, r = attempt_once(attempt, messages, temperature, seen)
        except Exception as e:  # API timeout / 5xx (or a harness bug): count it and retry the same prompt
            log.exception("[ERROR] attempt %d failed before grading: %s: %s", attempt, type(e).__name__, e)
            results.append((attempt, f"error: {type(e).__name__}", None, 0))
            lessons.append(f"Attempt {attempt}: no result (API or harness error, not your script's fault)")
            fails, broken_last = fails + 1, True
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("[DECISION] %d attempts in a row failed before grading - stopping", fails)
                break
            continue

        acc = r.get("acc")
        used = techniques(r.get("code", ""))
        tried |= used
        broken_last = acc is None
        fails = fails + 1 if acc is None else 0
        if acc is not None:
            experiments += 1
        if acc is not None and acc > best_acc + 0.005:
            best_acc, best_attempt, best_reply, stall = acc, attempt, reply, 0
            shutil.copy(r["graded_file"], BEST_MODEL)
            (GEN_DIR / "best.py").write_text(r["code"], encoding="utf-8")
            log.info("[DECISION] new best: attempt %d test=%.4f -> saved %s + generated/best.py",
                     attempt, acc, BEST_MODEL.relative_to(ROOT).as_posix())
        elif acc is not None:
            stall += 1
        if acc is not None and acc >= TARGET and r["gap"] <= MAX_GAP:
            log.info("[DECISION] test %.4f >= %.2f and gap %.4f <= %.2f -> TARGET REACHED, stopping",
                     acc, TARGET, r["gap"], MAX_GAP)
            results.append((attempt, "PASSED", acc, r["secs"]))
            break

        if r.get("duplicate_of"):
            r["duplicate_summary"] = lessons[r["duplicate_of"] - 1]
        status, fix_msg = feedback(r)
        results.append((attempt, status, acc, r.get("secs", 0)))
        lessons.append(f"Attempt {attempt}: {status}" + (
            f" | test {acc:.3f}, train {r['train_acc']:.3f}, {r['params']:,} params" if acc is not None else "")
            + (f" | techniques: {', '.join(sorted(used))}" if used else ""))
        if experiments >= MAX_ATTEMPTS or attempt >= MAX_LLM_CALLS:
            log.info("[DECISION] attempt %d %s -> budget spent, stopping", attempt, status)
            break
        if fails >= MAX_CONSECUTIVE_FAILS:
            log.error("[DECISION] %d attempts in a row never reached the grader - the model cannot repair its own "
                      "script; stopping", fails)
            break
        log.info("[DECISION] attempt %d %s -> asking LLM for a fix", attempt, status)
        if status.startswith("crashed:"):  # same error twice = patching is not working, go back to what ran
            errors[status] = errors.get(status, 0) + 1
            if errors[status] >= 2:
                log.warning("[DECISION] same error %d times: %s", errors[status], status[:80])
                fix_msg = (f"You have now hit this SAME error {errors[status]} times. Stop patching this script. "
                           f"Go back to the last script that actually trained" +
                           (f" (attempt {best_attempt}, test {best_acc:.4f}), shown above, " if best_attempt else " ") +
                           f"and make ONE small, safe change to it instead.\n\n") + fix_msg
        if stall >= 2 and best_attempt:
            log.warning("[DECISION] PLATEAU: no new best for %d attempts (best %.4f, attempt %d)",
                        stall, best_acc, best_attempt)
            fix_msg = (f"PLATEAU: the best test accuracy so far is {best_acc:.4f} (attempt {best_attempt}) and the "
                       f"last {stall} attempts did not beat it. Small tweaks to the same network are exhausted: make "
                       f"a STRUCTURAL change (a bigger or different architecture, data augmentation, a different "
                       f"training schedule).\n\n") + fix_msg
        # Short context: task + ONLY the last script + its feedback + a one-line lesson per earlier attempt.
        # A small model reading 4 broken scripts in a row gets confused; the lessons keep the memory.
        history = "\n".join(f"- {l}" for l in lessons)
        fix_msg += f"\n\nResults of ALL attempts so far (do not repeat these failures):\n{history}"
        untried = sorted(set(TECHNIQUES) - tried)
        if untried:
            fix_msg += ("\n\nTechniques NOT tried in any attempt yet - the accuracy you are missing is most likely in "
                        "this list:\n- " + "\n- ".join(untried))
        # Anchor on the best script when the last attempt was worse, so the search does not drift downhill.
        anchored = best_reply is not None and best_attempt != attempt
        anchor = best_reply if anchored else reply
        if anchored:
            fix_msg = (f"The script shown above is your BEST so far (attempt {best_attempt}, test {best_acc:.4f}). "
                       f"Build on THAT script. The feedback below is about your last attempt (attempt {attempt}), "
                       f"which did not beat it.\n\n") + fix_msg
            log.info("[DECISION] anchoring on best script (attempt %d, %.4f)", best_attempt, best_acc)
        messages = base + [{"role": "assistant", "content": anchor}, {"role": "user", "content": fix_msg}]
        log.info("[DECISION] next prompt: task + %s + %d lesson(s) + %d untried technique(s)",
                 "best script" if anchored else "last script", len(lessons), len(untried))

    log.info("=" * 70)
    log.info("[REPORT] model=%s total time %.1f min | %d generations, %d graded experiments",
             MODEL, (time.time() - t_start) / 60, attempt, experiments)
    for attempt, status, acc, secs in results:
        log.info("[REPORT] attempt %d | %-60s | verified_acc=%s | run=%.1f min", attempt, status[:60],
                 f"{acc:.4f}" if acc is not None else "  -   ", secs / 60)
    won = [x for x in results if x[1] == "PASSED"]
    log.info("[REPORT] %s", f"SUCCESS: attempt {won[0][0]} -> {won[0][2]:.4f}" if won
             else f"FAILED to reach {TARGET} in {experiments} graded experiments")
    if best_attempt:
        log.info("[REPORT] best model: attempt %d, test %.4f -> %s + generated/best.py",
                 best_attempt, best_acc, BEST_MODEL.relative_to(ROOT).as_posix())
    log.info("[REPORT] full log: %s", log_file)
    return 0 if won else 1


def selftest():
    assert extract_code("hi\n```python\nx=1\n```\n```python\nlonger=2\n```") == "longer=2"
    assert extract_code("x = 1") == "x = 1"
    assert parse_accuracy(["Epoch 1", "FINAL_TEST_ACCURACY=0.8123"]) == 0.8123
    assert parse_accuracy(["FINAL_TEST_ACCURACY=81.5"]) is None  # percent, not fraction
    assert parse_accuracy(["nothing"]) is None

    ok = ("import numpy as np\nimport tensorflow as tf\n"
          "d = np.load('data/cifar10_split.npz')\n"
          "x_train, x_val = d['x_train'], d['x_val']\n"
          "model = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255.0), tf.keras.layers.Conv2D(32, 3)])\n"
          "cb = tf.keras.callbacks.ModelCheckpoint('models/cifar10_cnn.keras')\n"
          "print('FINAL_TEST_ACCURACY')")
    assert check_code(ok) == [], check_code(ok)
    assert len(check_code("def (:")) == 6
    # a pretrained backbone with a Normalization layer is a CNN too
    pretrained = ("import tensorflow as tf\nd = 'data/cifar10_split.npz'\n'FINAL_TEST_ACCURACY'\n"
                  "cb = tf.keras.callbacks.ModelCheckpoint('models/cifar10_cnn.keras')\n"
                  "n = tf.keras.layers.Normalization()\nr = tf.keras.layers.Resizing(96, 96)\n"
                  "b = tf.keras.applications.EfficientNetB0(weights='imagenet')")
    assert check_code(pretrained) == [], check_code(pretrained)
    # the real Qwen bug: right callback, wrong filename -> rejected before training, with the name quoted back
    wrong = check_code(ok.replace("cifar10_cnn.keras", "cifar10_resnet.keras"))
    assert len(wrong) == 1 and "EXACTLY" in wrong[0] and "models/cifar10_resnet.keras" in wrong[0], wrong
    assert feedback({"problems": [], "exit_code": 0, "acc": 0.7, "train_acc": 0.8, "gap": 0.1,
                     "wrong_path": "cifar10_resnet.keras"})[1].startswith("WRONG CHECKPOINT PATH")
    # the exact bug from the Qwen log: scaled outside the model AND by the Rescaling layer
    bad = check_code(ok + "\nx_train = x_train.astype(np.float32) / 255.0\nx_val = x_val/255")
    assert len(bad) == 1 and "OUTSIDE the model" in bad[0] and "x_val = x_val/255" in bad[0], bad
    assert any("Rescaling" in p for p in check_code(ok.replace("Rescaling(1.0 / 255.0)", "")))
    assert any("TEST set" in p for p in check_code(ok + "\nm.fit(x, y, validation_data=(x_test, y_test))"))
    assert any("Lambda" in p for p in check_code(ok + "\nx = Lambda(f)"))
    assert any("PyTorch" in p for p in check_code(ok + "\nimport torch"))

    noisy = ["/usr/lib/keras/input_layer.py:27: UserWarning: Argument `input_shape` is deprecated.", "  warnings.warn(",
             "I0000 00:00:1790073713.126589    2507 gpu_device.cc:2020] Created device", "Epoch 1/50",
             "704/704 - 11s - accuracy: 0.0998 - loss: 2.3028 - val_accuracy: 0.1038"]
    assert clean(noisy) == noisy[3:], clean(noisy)
    ep = sum([[f"Epoch {i}/20", f"704/704 - loss: 2.30{i:02d}"] for i in range(1, 21)], [])
    c = epoch_curve(ep)
    assert c.startswith("Epoch 1/20: 704/704") and "12 epochs omitted" in c and c.endswith("loss: 2.3020"), c

    assert feedback({"problems": ["x"]})[0] == "static check failed: x"
    assert feedback({"problems": [], "timed_out": True, "exit_code": -9})[0] == "timeout"
    assert feedback({"problems": [], "exit_code": -9, "lines": ["Epoch 20/25"]})[0].startswith("killed externally")
    s = feedback({"problems": [], "exit_code": 1, "lines": ["Traceback (most recent call last):",
                  "NameError: name 'y_val' is not defined. Did you mean: 'x_val'?"]})[0]
    assert s == "crashed: NameError: name 'y_val' is not defined. Did you mean: 'x_val'?", s
    assert feedback({"problems": [], "exit_code": 0, "eval_error": "no model file"})[0] == "grader failed: no model file"
    assert feedback({"problems": [], "exit_code": 0, "acc": 0.71, "train_acc": 0.75, "gap": 0.04,
                     "secs": 600})[0] == "below target (0.7100) + underfit"
    # at/above target but memorised -> blocked; below target with a big gap -> regularise, never shrink
    assert feedback({"problems": [], "exit_code": 0, "acc": 0.96, "train_acc": 0.9999,
                     "gap": 0.16})[0].startswith("overfit at target")
    s, b = feedback({"problems": [], "exit_code": 0, "acc": 0.84, "train_acc": 0.9999, "gap": 0.16})
    assert s == "below target (0.8400) + overfit" and "Do NOT shrink" in b, s
    # sound recipe, just not strong enough -> asks for a bigger jump, not another tweak
    assert "bigger jump" in feedback({"problems": [], "exit_code": 0, "acc": 0.93, "train_acc": 0.96, "gap": 0.03})[1]

    assert techniques("RandomFlip('horizontal')\nBatchNormalization()\nCosineDecay(") == {
        "data augmentation", "batch normalization", "LR schedule"}
    assert techniques("applications.ResNet50(weights='imagenet')") >= {"pretrained backbone"}
    assert techniques("x = 1") == set()

    # the crash patterns from the qwen2.5-coder run, caught before any training
    assert any(p.startswith("'model' is used on line 3 before it is defined") for p in use_before_definition(
        "from keras import Sequential\ndef block(x, n): return x\nmodel = Sequential([block(model.layers[0].output, 64)])"))
    assert len(use_before_definition("x = f(1)\ndef f(a):\n    return a")) == 1      # defs are NOT hoisted
    assert use_before_definition("m = build()\nfor layer in m.layers:\n    layer.trainable = False") == [
        "'build' is used on line 1 before it is defined - the script would crash with NameError before training starts"]
    assert use_before_definition("import keras\nm = keras.Model()\nnames = [l.name for l in m.layers]") == []
    assert use_before_definition("def g():\n    return helper()\ndef helper():\n    return 1") == []
    assert undefined_names("def g():\n    return helper()") == ["helper"]
    assert undefined_names("import tensorflow as tf\ndef g():\n    return tf.keras") == []
    assert any("label_smoothing" in p for p in check_code(
        ok + "\nloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.1)"))
    # the Qwen log case: everything at chance -> NOT LEARNING, even though the /255 probe can't tell
    s, b = feedback({"problems": [], "exit_code": 0, "acc": 0.1048, "train_acc": 0.1077, "gap": 0.003,
                     "probe_acc": 0.1085, "claimed": 0.10, "secs": 180})
    assert s == "below target (0.1048) + NOT LEARNING" and b.startswith("NOT LEARNING") and "Root cause:" in b, s
    base = {"problems": [], "exit_code": 0, "acc": 0.24, "train_acc": 0.26, "gap": 0.02, "secs": 600}
    assert feedback({**base, "probe_acc": 0.68, "claimed": 0.69})[0] == "below target (0.2400) + preprocessing bug"
    assert feedback({**base, "probe_acc": 0.25, "claimed": 0.69})[0].endswith("claimed/verified mismatch")
    assert feedback({**base, "probe_acc": 0.25, "claimed": 0.245})[0] == "below target (0.2400) + underfit"

    # second Qwen log: identical scripts, underfit misread as overfit, promised-but-missing augmentation
    assert normalize("x = 1  # a\n\ny=2") == normalize("x=1\ny = 2  # changed comment")
    assert feedback({"problems": [], "duplicate_of": 7, "duplicate_summary": "Attempt 7: x"})[0] == "duplicate of attempt 7"
    s, b = feedback({"problems": [], "exit_code": 0, "acc": 0.7655, "train_acc": 0.8392, "gap": 0.0737,
                     "probe_acc": 0.0995, "claimed": 0.7655, "secs": 126})
    assert s == "below target (0.7655) + underfit" and "cannot fit even the TRAINING data" in b and "Compute:" in b, s
    # train 0.867 is far below the 0.95 target: capacity-limited, not memorising
    s, b = feedback({"problems": [], "exit_code": 0, "acc": 0.7607, "train_acc": 0.8673, "gap": 0.1066, "secs": 216})
    assert s == "below target (0.7607) + underfit" and "pretrained backbone" in b, s
    b = feedback({**base, "plan": "1. **Data Augmentation**: add layers", "code": "Conv2D Dropout"})[1]
    assert b.startswith("NOTE: your PLAN said you would add data augmentation"), b[:80]
    assert not feedback({**base, "plan": "augment", "code": "RandomFlip('horizontal')"})[1].startswith("NOTE")
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        client = make_client()
        sys.exit(main())
