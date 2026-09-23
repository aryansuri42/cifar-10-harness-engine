# Harness Engine

A test harness that makes a **small LLM** improve a CIFAR-10 CNN by itself: the model writes a training script,
the harness runs it, **independently grades the saved model**, and feeds a diagnosis back for the next attempt.

The point is not the CNN. It is how much of the gap between a weak model and a hard task can be closed by the
harness around it: verification the model cannot fake, checks that reject broken code before it costs GPU time,
and feedback specific enough to act on.

```
        ┌──────────────────────────── feedback: root cause + epoch curve + lessons ───────────────────────────┐
        │                                                                                                    │
        ▼                                                                                                    │
   task prompt ──► LLM (Qwen2.5-Coder-7B local, or NVIDIA NIM) ──► attempt_N.py ──► static checks ──► train ──┴─► grader
                                                                                   (reject early)   (subprocess)  (test + train acc)
```

## What the harness does that a plain retry loop does not

| Problem it solves | How |
|---|---|
| The script can report any accuracy it likes | The grader loads `models/cifar10_cnn.keras` itself and scores 10k test images. The printed number is only cross-checked; mismatches are reported. |
| A memorised model "passes" | Train accuracy is measured too; the train−test gap must stay within `MAX_GAP` at the target. |
| Wasted GPU time on code that cannot work | Static checks before running: syntax, undefined names, **use before definition**, pixel scaling outside the model, test-set leakage, wrong checkpoint path, missing `Rescaling`, known API traps. |
| "Below target" tells the model nothing | Diagnoses **not learning** (train accuracy at chance = pipeline bug), **pre-scaled input**, **underfitting** vs **overfitting**, unused compute budget, and a PLAN that promises augmentation the code lacks. |
| The model re-submits the same script | Scripts are compared with comments and whitespace stripped; a duplicate is never trained. |
| It drifts downhill after a good attempt | The next prompt is anchored on the **best** script, not the last one. |
| It keeps tuning the same two knobs | 16 techniques are detected by pattern; every message lists what has **never been tried** (augmentation, mixup, pretrained backbone, AdamW, mixed precision, …). |
| Crash loops burn the budget | Crashes do not consume the experiment budget; repeated identical errors trigger "go back to the last script that trained"; hard caps on generations and consecutive failures. |

## Results (Qwen2.5-Coder-7B, 4-bit, Colab T4)

| Run | Harness features | Best verified test accuracy |
|---|---|---|
| 1 | basic loop | 0.10 — every script scaled pixels three times, so the model never learned |
| 2 | scaling check, overfit gap, short context | 0.7731 — but 7 of 10 attempts re-ran identical code |
| 3 | duplicate detection, best-so-far anchor, plateau notice | 0.8759 — 13 of 25 attempts crashed at high temperature |
| 4 | temperature cap, NameError checks, crash budget | pending |

MNIST, the warm-up task, passes on the first attempt at **0.9916**.

## Files

| File | Purpose |
|---|---|
| `harness.py` | The engine: prompt, checks, runner, grader, diagnosis, loop. Run it directly. |
| `harness_colab.ipynb` | The same logic as a Colab notebook, one cell per step, with a local Qwen served by Ollama. **Generated** — do not edit by hand. |
| `build_notebook.py` | Builds the notebook from `harness.py` by copying the shared functions, so the two never drift apart. |
| `generated/attempt_N.py` | What the LLM wrote on each attempt; `best.py` is the best-scoring one. |
| `models/cifar10_cnn.keras` | The checkpoint the grader reads; `best.keras` is the best model across attempts. |
| `logs/run_<timestamp>.log` | Full log: prompts, the model's reasoning, generated code, training output, metrics, decisions. |
| `data/cifar10_split.npz` | CIFAR-10 pre-split into train 45k / val 5k / test 10k, built once from the Hugging Face mirror. |

## Running it

**Colab (recommended, GPU + local LLM):** upload `harness_colab.ipynb`, choose a **T4 GPU** runtime, Run all.
It installs Ollama, pulls `qwen2.5-coder:7b` (~4.7 GB) and runs the loop. No API key needed.

**Locally:**
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python harness.py              # run the loop
.venv/Scripts/python harness.py --selftest   # check the harness logic, no API calls, no training
```
The local default backend is NVIDIA NIM: put `NVIDIA_KEY=...` in `.env` (never printed or logged). Set
`LLM_BACKEND = "local"` to use an Ollama server instead.

After changing `harness.py`, rebuild the notebook: `python build_notebook.py`.

## Configuration

All at the top of `harness.py`:

| Setting | Default | Meaning |
|---|---|---|
| `TARGET` | `0.95` | verified test accuracy needed to pass |
| `MAX_GAP` | `0.15` | largest train−test gap allowed at the target |
| `MAX_ATTEMPTS` | `25` | **graded** experiments; crashes do not count |
| `MAX_LLM_CALLS` / `MAX_CONSECUTIVE_FAILS` | `60` / `8` | hard stops for crash loops |
| `MAX_TEMPERATURE` | `0.7` | above this a 7B model writes broken code instead of new ideas |
| `TRAIN_BUDGET_MIN` / `RUN_TIMEOUT` | `45` / `60 min` | budget told to the model, and the hard kill |
| `TOTAL_BUDGET_MIN` | `360` | wall clock for the whole run |
| `LOCAL_MODEL` / `NIM_MODEL` | `qwen2.5-coder:7b` / `openai/gpt-oss-20b` | the LLM under test |

## Log tags

`[SETUP] [DATA] [ATTEMPT] [PROMPT] [THINK] [LLM] [EXTRACT] [CODE] [CHECK] [EXEC] [VERIFY] [METRIC] [DECISION] [REPORT]`

`[THINK]` is the model's own reasoning: a native reasoning stream when the model has one, otherwise the PLAN it
writes before the code. `[METRIC]` is the line that matters:

```
[METRIC] test_acc=0.8759 | train_acc=0.9944 | gap=0.1185 (max 0.15) | claimed=0.8759 | probe(/255)=0.0995 | params=4,698,186 | train=24.5 min | exit=0
```

## Notes and limitations

- Generated code runs as a plain subprocess with a timeout — fine on your own machine or a disposable Colab VM,
  but it is **not a sandbox**.
- `MAX_GAP` is checked at the target, so it never blocks progress at lower accuracy.
- The 95% target is deliberately out of easy reach: a plain CNN plateaus near 0.78, so the loop has to find
  residual networks, strong augmentation or a pretrained backbone.
- `harness.py --selftest` covers extraction, every static check, the diagnoses and the feedback branches, with
  cases taken from real failures in the logs.
