"""Harness engine: an LLM writes a CIFAR-10 CNN, the harness runs it, grades it and loops.
Local twin of harness_colab.ipynb (same logic, same log tags).

Step 1: LLM client (NVIDIA NIM, or a local Ollama server) + logging.
Step 2: prompt (task + known-good reference script) -> generated code -> extract -> static checks -> save.
Step 3: sync the grader to the script (generated/grader_N.py: the model path it saves to, its helpers), then run
        the script in a subprocess, stream its output into the log, hard timeout.
Step 4: VERIFY: the synced grader loads the saved model itself (in a subprocess) and measures test accuracy, train
        accuracy (overfitting gap) and a /255 probe. The script's own printed accuracy is only cross-checked.
Step 5: diagnose + feedback -> the LLM gets the root cause (crash / timeout / OOM / not learning / scaling bug /
        overfit / low accuracy), the epoch curve, seconds per epoch, a one-line lesson per earlier attempt and
        ONE next experiment chosen by the harness (pretrained backbone, ResNet, 224px, mixup). Retry.
Step 6: final report.

Usage:  python harness.py              run the loop
        python harness.py --selftest   check checks/feedback logic, no API calls
"""
import ast
import builtins
import glob
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
MAX_GAP = 0.15            # train - test gap above which a result is diagnosed as overfitting. Not a pass rule:
                          # at test >= TARGET the gap is at most 1 - TARGET, and the held-out test set already
                          # rules out a memorised pass
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
    "You are a machine-learning engineer running experiments, not a tutorial writer. Each turn you receive "
    "measurements from your previous script (verified test and train accuracy, the epoch curve, seconds per epoch) "
    "and usually a NEXT EXPERIMENT chosen by the harness. Run exactly that experiment.\n"
    "Answer in exactly this shape:\n"
    "  Root cause: what the numbers say about the last script - one or two sentences, referring to specific values.\n"
    "  Change: the experiment you are running, which functions / CONFIG values change, and the expected gain.\n"
    "  Budget: EPOCHS x seconds per epoch = minutes, within the time limit.\n"
    "Then one complete, runnable Python script in a single ```python code block, and nothing after it.\n"
    "Edit the script you are shown: keep its data pipeline, callbacks and final evaluation, change build_model() "
    "and the CONFIG values. Rewriting working infrastructure from memory is how scripts crash.")

# Known-good starting point. A 7B model is a decent editor and a poor author: every from-scratch script in the
# Qwen logs re-invented the pipeline and ~half of them crashed (Add() shape mismatch, NameError, label_smoothing on
# a sparse loss). Given this, the model only has to change build_model() and the CONFIG block.
SCAFFOLD = r'''import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ---------------- CONFIG: the knobs to change between experiments
EPOCHS = 30              # keep EPOCHS x seconds-per-epoch under TIME_LIMIT_MIN
BATCH = 128
LR = 2e-3                # peak learning rate (AdamW, 5% linear warmup, then cosine decay to 0)
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
CUTOUT = 8               # side of the square erased from each training image, 0 = off
MIXUP = 0.0              # mixup alpha, e.g. 0.2; 0 = off
TIME_LIMIT_MIN = __TIME_LIMIT__      # training stops before this, whatever EPOCHS says
MODEL_PATH = "models/cifar10_cnn.keras"

if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def conv_bn(x, filters, stride=1):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def res_block(x, filters, stride=1):
    """Two 3x3 convs plus a shortcut; a 1x1 conv on the shortcut whenever the shape changes."""
    y = conv_bn(x, filters, stride)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(layers.Add()([x, y]))


def build_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))       # raw 0..255 pixels
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = conv_bn(x, 64)
    x = layers.MaxPooling2D()(x)
    x = conv_bn(x, 128)
    x = layers.MaxPooling2D()(x)
    x = conv_bn(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs)


def augment(x, y):
    """One uint8 training image: pad 4 + random crop, horizontal flip, cutout."""
    x = tf.image.resize_with_crop_or_pad(x, 40, 40)
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    if CUTOUT:
        cy = tf.random.uniform([], 0, 32, tf.int32)
        cx = tf.random.uniform([], 0, 32, tf.int32)
        rows, cols = tf.range(32)[:, None], tf.range(32)[None, :]
        hole = (tf.abs(rows - cy) < CUTOUT // 2 + 1) & (tf.abs(cols - cx) < CUTOUT // 2 + 1)
        x = tf.where(hole[..., None], tf.zeros_like(x), x)
    return x, y


def to_float(x, y):
    return tf.cast(x, tf.float32), tf.one_hot(y, 10)   # pixels stay 0..255: scaling happens inside the model


def mixup(x, y):
    """One batch: blend every image and its one-hot label with another image of the same batch."""
    g1, g2 = tf.random.gamma([], MIXUP), tf.random.gamma([], MIXUP)
    lam = g1 / (g1 + g2)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)


class TimeLimit(tf.keras.callbacks.Callback):
    """Stops training when the next epoch would not finish inside TIME_LIMIT_MIN."""
    def on_train_begin(self, logs=None):
        self.end = time.time() + TIME_LIMIT_MIN * 60

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() + (time.time() - self.t0) > self.end:
            print(f"time limit reached after epoch {epoch + 1}")
            self.model.stop_training = True


def main():
    d = np.load("data/cifar10_split.npz")
    x_train, y_train = d["x_train"], d["y_train"]
    x_val, y_val = d["x_val"], d["y_val"]
    x_test, y_test = d["x_test"], d["y_test"]          # final evaluation only

    auto = tf.data.AUTOTUNE
    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
             .map(augment, num_parallel_calls=auto).map(to_float, num_parallel_calls=auto)
             .batch(BATCH, drop_remainder=True))
    if MIXUP:
        train = train.map(mixup, num_parallel_calls=auto)
    train = train.prefetch(auto)
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(to_float).batch(256).prefetch(auto)

    model = build_model()
    steps = EPOCHS * (len(x_train) // BATCH)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=steps - steps // 20, warmup_target=LR, warmup_steps=steps // 20)
    model.compile(optimizer=tf.keras.optimizers.AdamW(schedule, weight_decay=WEIGHT_DECAY),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                  metrics=["accuracy"])
    print(f"params: {model.count_params():,}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.fit(train, epochs=EPOCHS, validation_data=val, verbose=2, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        TimeLimit()])

    best = tf.keras.models.load_model(MODEL_PATH, compile=False)   # the checkpoint the grader will read
    probs = best.predict(x_test.astype("float32"), batch_size=200, verbose=0)
    print(f"FINAL_TEST_ACCURACY={np.mean(probs.argmax(1) == y_test):.4f}")


if __name__ == "__main__":
    main()
'''.replace("__TIME_LIMIT__", str(TRAIN_BUDGET_MIN - 5))

TASK = f"""Train a convolutional neural network on CIFAR-10 and reach {TARGET:.0%} test accuracy.

This is a research loop: up to {MAX_ATTEMPTS} experiments, each graded by an independent grader, each result sent
back to you. After every result the harness names the NEXT EXPERIMENT to run. Follow it.

WHAT SUCCESS MEANS
- The grader loads the model file your script saves and measures accuracy on 10000 held-out test images: >= {TARGET:.2f}.
- It also measures train accuracy, to tell underfitting (train accuracy too low) from overfitting (train - test gap
  above {MAX_GAP:.2f}).
- Measured in earlier runs of this loop: a plain CNN reaches 0.76-0.84, a VGG-style CNN with BatchNorm and
  Adam 0.88. That is the ceiling of tweaking. {TARGET:.2f} needs one of:
    (a) an ImageNet-pretrained backbone fine-tuned at a higher resolution - the most reliable route, about 0.96
        in 10-15 epochs;
    (b) a ResNet-18-style network with crop + flip + cutout, a cosine schedule and 40+ epochs - about 0.94-0.95.

START FROM THIS REFERENCE SCRIPT
It runs as-is and meets every hard constraint. It already has a fast tf.data pipeline (random crop, flip, cutout,
optional mixup), mixed precision on GPU, AdamW with warmup + cosine decay, label smoothing, a checkpoint and a
time limit. Its build_model() is only a small baseline: replace it, and change the CONFIG values.

```python
""" + SCAFFOLD + f"""```

BUILDING BLOCKS - correct and tested, copy them rather than writing your own
- ResNet-18 style, with res_block and conv_bn from the reference script:
      x = layers.Rescaling(1.0 / 255)(inputs)
      x = conv_bn(x, 64)
      for filters, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
          x = res_block(x, filters, stride)
      x = layers.GlobalAveragePooling2D()(x)
- Pretrained backbone. EfficientNetV2 rescales raw 0..255 pixels itself, so NO Rescaling layer in front of it:
      IMG = 160                     # 224 is more accurate and about 2x slower per epoch
      def build_model():
          inputs = tf.keras.Input(shape=(32, 32, 3))
          x = layers.Resizing(IMG, IMG)(inputs)
          base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
                                                        input_shape=(IMG, IMG, 3), pooling="avg")
          x = base(x)
          x = layers.Dropout(0.3)(x)
          outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)
          return tf.keras.Model(inputs, outputs)
  Fine-tune everything with LR = 5e-4, BATCH = 64, EPOCHS = 10-15. Downloading the ImageNet weights is allowed.

HARD CONSTRAINTS (checked before your script runs; violations are rejected without training)
1. TensorFlow / tf.keras only.
2. Data comes only from "data/cifar10_split.npz", loaded as in the reference script: x_* are uint8
   (N, 32, 32, 3), y_* int64 (N,). Train on x_train, validate on x_val. x_test / y_test only for the final line.
3. The saved model takes RAW 0..255 pixels: the grader calls model.predict on float32 0..255 images of shape
   (N, 32, 32, 3). All scaling happens inside the model. Any "/ 255" outside the model is rejected.
4. Save the full model (not weights only) to a .keras file with ModelCheckpoint during training. The harness
   reads the path from your script and grades that file.
5. model.fit(..., verbose=2). The LAST printed line is FINAL_TEST_ACCURACY=<float 0..1, 4 decimals>.
6. Built-in Keras layers only: no Lambda layers, no custom Layer / Model subclasses. Custom callbacks are fine.
7. Guard the entry point with if __name__ == "__main__":
8. Training must end within {TRAIN_BUDGET_MIN} minutes (killed at {RUN_TIMEOUT // 60}). The harness tells you the seconds
   per epoch after each run: set EPOCHS so that EPOCHS x seconds stays under {TRAIN_BUDGET_MIN - 5} minutes.

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
    # top_p 0.8 is Qwen2.5-Coder's own recommended setting: trims the unlikely tokens that turn into typos
    extra = {"top_p": 0.8} if LLM_BACKEND == "local" else {}
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature,
                                          max_tokens=max_tokens, **extra)
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


def _str(node, consts):
    """Best-effort string value of an expression; '*' for the parts only known at run time."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.JoinedStr):  # f"models/ckpt_{epoch:02d}.keras"
        return "".join(v.value if isinstance(v, ast.Constant) else _str(v.value, consts) or "*" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _str(node.left, consts), _str(node.right, consts)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "join" and node.args:  # os.path.join
        parts = [_str(a, consts) for a in node.args]
        return "/".join(parts) if all(parts) else None
    return None


def model_paths(code):
    """Files the script saves a full model to (ModelCheckpoint / model.save / save_model), as glob patterns.
    This is what the grader is synced to, so a renamed checkpoint is graded instead of reported missing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    consts, paths = {}, []
    for n in ast.walk(tree):  # MODEL_PATH = "models/x.keras" -> ModelCheckpoint(MODEL_PATH)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            value = _str(n.value, consts)
            if value is not None:
                consts[n.targets[0].id] = value
    for n in ast.walk(tree):
        name = getattr(n, "func", None) and (getattr(n.func, "attr", None) or getattr(n.func, "id", ""))
        if name not in ("ModelCheckpoint", "save", "save_model") or any(
                k.arg == "save_weights_only" and getattr(k.value, "value", False) is True for k in n.keywords):
            continue
        for arg in n.args + [k.value for k in n.keywords]:
            s = _str(arg, consts)
            if s and s.endswith((".keras", ".h5")) and not s.endswith(".weights.h5"):
                s = re.sub(r"\{[^}]*\}", "*", s)  # "{epoch:02d}" in a plain string is filled in by Keras
                if s not in paths:
                    paths.append(s)
    return paths


# Backbones with their own pixel preprocessing: they take raw 0..255 and need no Rescaling in front.
SELF_RESCALING = re.compile(r"EfficientNet|ConvNeXt")


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
    if not model_paths(code):
        if re.search(r"save_weights_only\s*=\s*True", code):
            problems.append("save_weights_only=True saves weights, not a model the grader can load - remove it and "
                            "save the full model to a .keras file")
        elif not re.search(r"ModelCheckpoint\(|\.save\(|save_model\(", code):  # dynamic path: grader falls back
            problems.append('the script never saves the model - add tf.keras.callbacks.ModelCheckpoint('
                            '"models/cifar10_cnn.keras", monitor="val_accuracy", save_best_only=True)')
    if re.search(r"validation_data\s*=\s*\(\s*x_test", code):
        problems.append("uses the TEST set as validation_data - use validation_data=(x_val, y_val) (test leakage)")
    if "Rescaling(" not in code and "Normalization(" not in code and not SELF_RESCALING.search(code):
        problems.append("no Rescaling / Normalization layer inside the model - it would receive raw 0..255 pixels")
    if SELF_RESCALING.search(code) and "Rescaling(" in code and "include_preprocessing=False" not in code:
        problems.append("EfficientNet / ConvNeXt backbones rescale raw 0..255 pixels themselves: remove the "
                        "Rescaling layer, otherwise the backbone sees pixels divided by 255 twice")
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

# Written to generated/grader_N.py for every script by sync_grader(), run in a subprocess (keeps TensorFlow's RAM
# and GPU memory out of the harness). The harness writes it, never the LLM: the measurement stays independent,
# only WHERE the model lives and WHICH helpers it needs to load follow the script.
GRADER = r'''"""Grader for one generated training script (written by the harness, not by the LLM)."""
import glob, json, os, sys
import numpy as np
import tensorflow as tf

CFG = __CONFIG__


def find_model():
    """Newest file the script declared; if none exists, any model left in models/ (a path built at run time)."""
    for patterns in (CFG["paths"], ["models/*.keras", "models/*.h5"]):
        found = [p for pat in patterns for p in glob.glob(pat) if not os.path.basename(p).startswith("best.")]
        if found:
            return max(found, key=os.path.getmtime)
    raise FileNotFoundError(f"the script saved no model file (looked for {CFG['paths'] or 'models/*.keras'})")


def score(m, x, y):
    s = np.asarray(m.predict(x.astype("float32"), batch_size=200, verbose=0))
    if s.shape != (len(x), 10):
        raise ValueError(f"model output shape {s.shape}, expected ({len(x)}, 10)")
    return float(np.mean(s.argmax(1) == y))


try:
    ns = {}
    try:  # the script's own imports, functions and classes, so custom layers / losses in the file can load
        exec(CFG["prelude"], ns)
    except Exception as e:
        print("prelude failed:", e)
    path = find_model()
    # compile=False: a custom loss or optimizer in the checkpoint must not stop us from measuring the network
    m = tf.keras.models.load_model(path, custom_objects={k: v for k, v in ns.items() if callable(v)},
                                   compile=False, safe_mode=False)
    d = np.load(sys.argv[1])
    print("GRADE=" + json.dumps({"file": path,
                                 "acc": score(m, d["x_test"], d["y_test"]),
                                 "train_acc": score(m, d["x_train"][:10000], d["y_train"][:10000]),
                                 # diagnostic probe: does the model secretly expect pixels already divided by 255?
                                 "probe_acc": score(m, d["x_test"][:2000] / 255.0, d["y_test"][:2000]),
                                 "params": int(m.count_params())}))
except Exception as e:
    print("GRADE=" + json.dumps({"error": f"{type(e).__name__}: {e}"}))
'''


def sync_grader(attempt, code):
    """Rewrite the grader for THIS script: it grades the file the script actually saves (whatever it is called,
    epoch-numbered names included) and can load the script's own helpers. Returns (grader path, save paths)."""
    paths = model_paths(code)
    tree = ast.parse(code)
    prelude = "\n".join(ast.get_source_segment(code, n) for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)))
    grader = GEN_DIR / f"grader_{attempt}.py"
    grader.write_text(GRADER.replace("__CONFIG__", repr({"paths": paths, "prelude": prelude})), encoding="utf-8")
    log.info("[GRADER] wrote %s -> grades %s", grader.relative_to(ROOT).as_posix(),
             ", ".join(paths) or "newest model in models/ (no literal save path in the script)")
    return grader, paths


def verify_model(grader):
    """Runs the synced grader. Returns dict with file, acc, train_acc, probe_acc, params - or error."""
    out = subprocess.run([sys.executable, str(grader), str(DATA_FILE)], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         env={**TF_ENV, "TF_CPP_MIN_LOG_LEVEL": "3"}, timeout=900).stdout
    m = re.search(r"GRADE=(.*)", out)
    return json.loads(m.group(1)) if m else {"error": "grader crashed: " + out[-500:]}

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


AUG = re.compile(r"Random(Flip|Translation|Rotation|Crop|Zoom|Contrast)|random_(crop|flip)")

# Techniques the harness can recognise in a script, by USE rather than by mention: the reference script defines
# res_block() and mixup() whether or not they are used. Feeds the "not tried yet" list and the next-move choice.
TECHNIQUES = {
    "data augmentation": AUG,
    "cutout": re.compile(r"CUTOUT\s*=\s*[1-9]|cutout\(", re.I),
    "mixup / cutmix": re.compile(r"MIXUP\s*=\s*(0?\.0*[1-9]|[1-9])|cutmix", re.I),
    "batch normalization": re.compile(r"BatchNormalization"),
    "dropout": re.compile(r"Dropout\("),
    "weight decay / L2": re.compile(r"weight_decay|regularizers\.l2|kernel_regularizer"),
    "label smoothing": re.compile(r"label_smoothing\s*=\s*(0?\.0*[1-9]|LABEL)"),
    "residual / skip connections": re.compile(r"=\s*\w*(res_block|residual|identity_block|bottleneck)\w*\(|ResNet",
                                              re.I),
    "pretrained backbone": re.compile(r"weights\s*=\s*[\"']imagenet[\"']"),
    "high resolution (224px)": re.compile(r"IMG\s*=\s*2[2-9]\d|Resizing\(\s*2[2-9]\d"),
    "LR schedule": re.compile(r"CosineDecay|LearningRateScheduler|ReduceLROnPlateau|ExponentialDecay|PiecewiseConstant|warmup|OneCycle", re.I),
    "SGD + momentum": re.compile(r"SGD\("),
    "AdamW": re.compile(r"AdamW"),
    "mixed precision": re.compile(r"mixed_float16|set_global_policy"),
    "tf.data pipeline": re.compile(r"tf\.data|from_tensor_slices"),
    "global average pooling": re.compile(r"GlobalAveragePooling2D|pooling\s*=\s*[\"']avg"),
}


def techniques(code):
    """Which known techniques a script uses (best-effort, by pattern)."""
    return {name for name, rx in TECHNIQUES.items() if rx.search(code or "")}


# The harness does the research planning, the LLM the implementation: a 7B model given a list of 14 untried
# techniques picks the easiest (RandomFlip) every time; given ONE concrete experiment it implements it.
# (technique, applies to the best script's techniques, instruction), in order of expected gain on this task.
MOVES = [
    ("pretrained backbone", lambda have: True,
     "Replace build_model() with the pretrained EfficientNetV2B0 pattern from BUILDING BLOCKS (IMG = 160, no "
     "Rescaling layer) and set LR = 5e-4, BATCH = 64, EPOCHS = 12. Keep the data pipeline, callbacks and final "
     "evaluation exactly as they are. Fine-tuning an ImageNet backbone is the most reliable way past 0.93."),
    ("residual / skip connections", lambda have: "pretrained backbone" not in have,
     "Replace build_model() with the ResNet-18-style network from BUILDING BLOCKS (res_block, stages 64-128-256-512, "
     "two blocks per stage) and set EPOCHS as high as the measured seconds per epoch allow."),
    ("high resolution (224px)", lambda have: "pretrained backbone" in have,
     "Keep the pretrained backbone and raise IMG to 224 (and input_shape with it), BATCH = 64. An epoch takes about "
     "2x longer: set EPOCHS from the measured seconds per epoch so the run stays inside the time limit."),
    ("mixup / cutmix", lambda have: True,
     "Keep the network. Set MIXUP = 0.2 (keep CUTOUT) and train about 30% more epochs if time allows: mixup "
     "regularises, so it needs longer training to pay off."),
]


def next_move(best_code, asked):
    """First move the best script does not use yet and that has not been asked for twice. (name, text) or Nones."""
    have = techniques(best_code)
    for name, applies, how in MOVES:
        if name not in have and applies(have) and asked.get(name, 0) < 2:
            return name, how
    return None, None


def epoch_seconds(lines):
    """Median seconds per epoch from Keras verbose=2 lines like '351/351 - 42s - 120ms/step - accuracy: ...'."""
    secs = sorted(int(m.group(1)) for l in lines if (m := re.match(r"\d+/\d+ - (\d+)s ", l)))
    return secs[len(secs) // 2] if secs else None


def budget_note(r):
    """Seconds per epoch and how much of the compute budget was left on the table: a 7B model cannot work out
    EPOCHS from 'use the budget', it can from 'one epoch = 40 s, the budget fits 57 epochs'."""
    used, sec, note = r.get("secs", 0) / 60, epoch_seconds(r.get("lines", [])), ""
    if sec:
        note += (f"\nTiming: one epoch took about {sec}s, so {TRAIN_BUDGET_MIN - 5} minutes fit about "
                 f"{(TRAIN_BUDGET_MIN - 5) * 60 // sec} epochs of this network.")
    if 0 < used < TRAIN_BUDGET_MIN / 3:
        note += (f"\nCompute: training used only {used:.1f} of the {TRAIN_BUDGET_MIN} minute budget, so roughly "
                 f"{TRAIN_BUDGET_MIN / used:.0f}x more compute is available for a bigger network or longer training.")
    return note


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
        body = (f"The grader could not use the model your script saved: {r['eval_error']}\nThe grader looked for "
                f"{', '.join(r.get('save_paths') or ['models/*.keras'])}. The saved file must be a full model "
                f"(not weights only) that loads with tf.keras.models.load_model, accepts raw 0..255 images of shape "
                f"(N, 32, 32, 3) and outputs 10 scores.\nEpochs:\n```\n{curve}\n```")
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
                head += (f"Train accuracy is only {tr:.4f}, below the {TARGET} target, so the model cannot fit the "
                         f"TRAINING data well enough to pass, however well it generalises: it is too small or trained "
                         f"too briefly (underfitting relative to the target), and this is NOT overfitting. Do NOT add more Dropout. Add "
                         f"capacity (deeper / wider, residual blocks, or an ImageNet-pretrained backbone) and train "
                         f"longer.")
            else:
                head += (f"Train accuracy is {tr:.4f} with a healthy gap of {gap:.4f}: the recipe is sound but not "
                         f"strong enough. A bigger jump is needed - a different architecture family or a pretrained "
                         f"backbone, not another small tweak.")
        body = head + f"\nEpochs:\n```\n{curve}\n```" + budget_note(r)
    asked = r.get("directive")
    if asked and r.get("code") and asked not in techniques(r["code"]):
        body = (f"NOTE: the experiment you were asked to run was '{asked}', but your script does not contain it. "
                f"Write the code the experiment calls for.\n\n") + body
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
    grader, r["save_paths"] = sync_grader(attempt, code)
    for pattern in r["save_paths"] + ["models/*.keras", "models/*.h5"]:  # no earlier attempt's file may be graded
        for stale in map(Path, glob.glob(str(ROOT / pattern))):
            if stale.is_file() and not stale.name.startswith("best.") and ROOT.resolve() in stale.resolve().parents:
                stale.unlink()
    r["exit_code"], r["lines"], r["secs"], r["timed_out"] = run_script(path)
    claimed = parse_accuracy(r["lines"])
    log.info("[VERIFY] %s: grading on 10k test + first 10k train images", grader.relative_to(ROOT).as_posix())
    g = verify_model(grader)
    if g.get("error"):
        r["eval_error"] = g["error"]
        log.warning("[VERIFY] %s", r["eval_error"])
        return reply, r
    r["acc"], r["train_acc"], r["probe_acc"], r["claimed"] = g["acc"], g["train_acc"], g["probe_acc"], claimed
    r["gap"], r["params"], r["graded_file"] = r["train_acc"] - r["acc"], g["params"], ROOT / g["file"]
    log.info("[VERIFY] graded %s", g["file"])
    log.info("[METRIC] test_acc=%.4f | train_acc=%.4f | gap=%.4f (overfit above %.2f) | claimed=%s | probe(/255)=%.4f | "
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
    log.info("[SETUP] backend=%s model=%s task=CIFAR-10 target=%.2f overfit_gap=%.2f max_attempts=%d train_budget=%dmin "
             "kill=%dmin total_budget=%dmin log=%s", LLM_BACKEND, MODEL, TARGET, MAX_GAP, MAX_ATTEMPTS,
             TRAIN_BUDGET_MIN, RUN_TIMEOUT // 60, TOTAL_BUDGET_MIN, log_file.name)
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    messages = base
    results, lessons, seen, t_start = [], [], {}, time.time()
    best_acc, best_attempt, best_reply, stall = -1.0, None, None, 0  # stall = experiments in a row without a new best
    best_code, asked, directive, save_paths = "", {}, None, None  # asked = how often each MOVE was requested
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
        r["directive"] = directive
        if r.get("save_paths") is not None and r["save_paths"] != save_paths:
            log.info("[GRADER] save path changed: %s -> %s (grader updated to match)", save_paths, r["save_paths"])
            save_paths = r["save_paths"]
        used = techniques(r.get("code", ""))
        tried |= used
        broken_last = acc is None
        fails = fails + 1 if acc is None else 0
        if acc is not None:
            experiments += 1
        if acc is not None and acc > best_acc + 0.005:
            best_acc, best_attempt, best_reply, best_code, stall = acc, attempt, reply, r["code"], 0
            shutil.copy(r["graded_file"], BEST_MODEL.with_suffix(r["graded_file"].suffix))
            (GEN_DIR / "best.py").write_text(r["code"], encoding="utf-8")
            log.info("[DECISION] new best: attempt %d test=%.4f -> saved %s + generated/best.py",
                     attempt, acc, BEST_MODEL.relative_to(ROOT).as_posix())
        elif acc is not None:
            stall += 1
        if acc is not None and acc >= TARGET:
            log.info("[DECISION] test %.4f >= %.2f (gap %.4f) -> TARGET REACHED, stopping", acc, TARGET, r["gap"])
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
        # After a measured result the harness picks the next experiment; after a crash the job is the repair.
        directive, how = next_move(best_code, asked) if acc is not None else (None, None)
        if directive:
            asked[directive] = asked.get(directive, 0) + 1
            log.info("[DECISION] next experiment: %s (request %d)", directive, asked[directive])
            fix_msg = f"NEXT EXPERIMENT (chosen by the harness from all results so far): {how}\n\n" + fix_msg
        elif stall >= 2 and best_attempt:
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
        if untried and not directive and acc is not None:
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
    # the real Qwen case: a renamed checkpoint is no longer a failed attempt - the grader follows the script
    assert check_code(ok.replace("cifar10_cnn.keras", "cifar10_resnet.keras")) == []
    assert model_paths(ok.replace("cifar10_cnn.keras", "cifar10_resnet.keras")) == ["models/cifar10_resnet.keras"]
    assert model_paths("P = 'models/a.keras'\ncb = ModelCheckpoint(P)\nmodel.save('models/final.h5')") == [
        "models/a.keras", "models/final.h5"]
    assert model_paths("ModelCheckpoint(f'models/ep_{epoch:02d}.keras')\nModelCheckpoint('m/{epoch}.keras')") == [
        "models/ep_*.keras", "m/*.keras"]
    assert model_paths("D = 'models'\nsave_model(m, os.path.join(D, 'x.keras'))\nnp.save('a.npy', x)") == [
        "models/x.keras"]
    assert model_paths("ModelCheckpoint('w.weights.h5', save_weights_only=True)") == []
    weights = ok.replace("ModelCheckpoint('models/cifar10_cnn.keras')",
                         "ModelCheckpoint('models/w.keras', save_weights_only=True)")
    assert any("save_weights_only" in p for p in check_code(weights)), check_code(weights)
    assert check_code(SCAFFOLD) == [], check_code(SCAFFOLD)
    assert model_paths(SCAFFOLD) == ["models/cifar10_cnn.keras"]
    # EfficientNetV2 rescales itself: no Rescaling needed, and a Rescaling in front of it is rejected
    effnet = SCAFFOLD.replace("layers.Rescaling(1.0 / 255)(inputs)", "layers.Resizing(160, 160)(inputs)") + \
        "\nbase = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet')"
    assert check_code(effnet) == [], check_code(effnet)
    assert any("twice" in p for p in check_code(effnet + "\nlayers.Rescaling(1.0 / 255)"))
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
    # below target with a big gap -> regularise, never shrink
    s, b = feedback({"problems": [], "exit_code": 0, "acc": 0.84, "train_acc": 0.9999, "gap": 0.16})
    assert s == "below target (0.8400) + overfit" and "Do NOT shrink" in b, s
    # sound recipe, just not strong enough -> asks for a bigger jump, not another tweak
    assert "bigger jump" in feedback({"problems": [], "exit_code": 0, "acc": 0.93, "train_acc": 0.96, "gap": 0.03})[1]

    assert techniques("RandomFlip('horizontal')\nBatchNormalization()\nCosineDecay(") == {
        "data augmentation", "batch normalization", "LR schedule"}
    assert techniques("applications.ResNet50(weights='imagenet')") >= {"pretrained backbone"}
    assert techniques("x = 1") == set()
    # the reference script DEFINES res_block and mixup; only their use counts
    base_t = techniques(SCAFFOLD)
    assert {"cutout", "label smoothing", "AdamW", "mixed precision", "data augmentation"} <= base_t, base_t
    assert not base_t & {"residual / skip connections", "mixup / cutmix", "pretrained backbone"}, base_t
    assert "residual / skip connections" in techniques(SCAFFOLD + "\n    x = res_block(x, 64)")
    assert "mixup / cutmix" in techniques(SCAFFOLD.replace("MIXUP = 0.0", "MIXUP = 0.2"))
    assert "residual / skip connections" not in techniques("x = layers.Rescaling(1/255)(inputs)\nx = Rescaling(2)")
    # the harness picks the research direction: pretrained first, ResNet only while not pretrained, then 224px
    assert next_move(SCAFFOLD, {})[0] == "pretrained backbone"
    assert next_move(SCAFFOLD, {"pretrained backbone": 2})[0] == "residual / skip connections"
    assert next_move(effnet, {})[0] == "high resolution (224px)"
    assert next_move(effnet.replace("160", "224") + "\nMIXUP = 0.2", {}) == (None, None)
    b = feedback({**{"problems": [], "exit_code": 0, "acc": 0.84, "train_acc": 0.86, "gap": 0.02, "secs": 600},
                  "directive": "pretrained backbone", "code": SCAFFOLD})[1]
    assert b.startswith("NOTE: the experiment you were asked to run was 'pretrained backbone'"), b[:80]
    assert epoch_seconds(["Epoch 1/3", "351/351 - 50s - 1ms/step - accuracy: 0.4", "Epoch 2/3",
                          "351/351 - 41s - 1ms/step - accuracy: 0.5", "351/351 - 40s - x"]) == 41
    assert "fit about 58 epochs" in budget_note({"secs": 2000, "lines": ["351/351 - 41s - x"]})

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
    assert s == "below target (0.7655) + underfit" and "cannot fit the TRAINING data well enough" in b and "Compute:" in b, s
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
