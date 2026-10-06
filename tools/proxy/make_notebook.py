"""Build the self-contained Colab notebook from proxy_gpt.py (library part inlined).

    python tools/proxy/make_notebook.py tools/proxy/proxy_gpt.py tools/proxy/nanogpt_proxy_screen.ipynb
"""
import json, re, sys
src = open(sys.argv[1]).read()
out = sys.argv[2]
lib = src[:src.index("\ndef main():")]
lib = lib[lib.index('"""', 3) + 3:].lstrip("\n")          # drop the module docstring (CLI usage)
lib = "# 4. The proxy: model, optimizers, training loop, and the ideas as flags (VARIANTS).\n" + lib.rstrip() + "\n"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = [
md('''
# NanoGPT speedrun: architecture and optimizer screen (one GPU)

The speedrun record needs 8xH100. This notebook screens ideas cheaply first, on one Colab GPU. It trains a
compact speedrun-style GPT (RoPE, QK-norm, ReLU², zero-init projections, untied head, softcap, Muon + Adam) on
the speedrun's own GPT-2-tokenized FineWeb. It needs no code from GitHub: everything is in the cells below, and
the data comes from Hugging Face.

**How to run:** Runtime → Change runtime type → **A100** or **H100** GPU (L4 works, ~3x slower; T4 is too slow,
so use `QUICK = True` there). Then Runtime → **Run all**.

**Two rounds:**
1. Every idea against the baseline:
   - **Calibration:** techniques the record already uses (Polar Express, NorMuon, cautious WD, value embeddings,
     U-net and embedding skips, smear, sparse attention gate, bigram hash, MTP). If the proxy is predictive,
     these should beat the baseline. If they don't, its verdicts on new ideas are not worth much either.
   - **Candidates:** ideas not in the record: Canon layers, differential attention, a Snoo outer Nesterov
     optimizer, a power-law LR cooldown, z-loss, and content-aware gating of the n-gram embedding.
2. Each candidate that won in round 1, re-tested on top of the stack of calibration winners. That is the real
   question, since the record already has those.

Each run appends to a results file, and **re-running skips finished runs**, so a disconnect loses at most one run.
Round 1 takes about an hour on an A100 (about 40 min on an H100); round 2 adds 15-30 min.

**Two GPUs?** Open a second copy of this notebook on the other GPU and set `FIRST_SEED = 2` there. It runs
different seeds of the same variants, so the two summaries together give twice the seeds in the same wall time.

**When it finishes, copy the last cell's output (the summary) back to Claude**, from each copy if you ran two.

A proxy win is a screen, not proof: the record is ~124M params plus an 84.6M-row n-gram table trained on ~330M
tokens. Ideas that win here, especially in round 2, are the ones worth 8xH100 time.
'''),
code('''
# 1. GPU check, and the one package Colab lacks (PyTorch and NumPy are preinstalled).
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip install -q huggingface_hub
'''),
code('''
# 2. Settings
SEEDS = 2           # seeds per variant: seed noise at this scale is a few millinats, so keep >= 2
FIRST_SEED = 0      # a second copy on another GPU: set 2 here so it runs seeds 2 and 3
QUICK = False       # True: 1 seed and 600 steps per run, a ~20-minute first look (noisier). Use on a T4.
DATA_DIR = "/content/fineweb10B"
RESULTS_FILE = "/content/proxy_results.jsonl"  # every finished run is appended; re-running skips them
SAVE_TO_DRIVE = False  # True: keep results in Google Drive across disconnects (asks for permission)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_FILE = "/content/drive/MyDrive/proxy_results.jsonl"
'''),
code('''
# 3. Data: the speedrun's own GPT-2-tokenized FineWeb from Hugging Face: the val shard and 1 of the 103 train
# shards (~400 MB). One run reads 20M tokens.
import os
from huggingface_hub import hf_hub_download

os.makedirs(DATA_DIR, exist_ok=True)
for name in ("fineweb_val_000000.bin", "fineweb_train_000001.bin"):
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=name, repo_type="dataset", local_dir=DATA_DIR)
print(sorted(os.listdir(DATA_DIR)))
'''),
code(lib),
code('''
# 5. Runner: resumable. A run is keyed by its variant, seed and scale; finished runs are skipped.
import json, os, statistics


def combine(name):
    changes = {}
    for part in name.split("+"):
        changes.update(VARIANTS[part])
    return changes


def run_key(name, seed, cfg):
    return f"{name}|{seed}|{cfg.steps}|{cfg.n_layer}|{cfg.d_model}|{cfg.batch_seqs}|{cfg.seq_len}"


def load_results():
    done = {}
    if os.path.exists(RESULTS_FILE):
        for line in open(RESULTS_FILE):
            r = json.loads(line)
            done[r["key"]] = r
    return done


def run(names, seeds, base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = load_results()
    results = {}
    for name in names:
        for seed in range(FIRST_SEED, FIRST_SEED + seeds):
            cfg = base.update(**combine(name), seed=seed)
            key = run_key(name, seed, cfg)
            if key not in done:
                print(f"== {name} seed {seed}: {combine(name)}", flush=True)
                r = train(cfg, DATA_DIR, device, log_every=400)
                r.update(key=key, variant=name, seed=seed,
                         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
                with open(RESULTS_FILE, "a") as f:
                    f.write(json.dumps(r) + "\\n")
                done[key] = r
                print(f"   val {r['val_loss']:.4f}  {r['seconds']:.0f}s", flush=True)
            results.setdefault(name, []).append(done[key])
    return results


def table(results, names, title):
    base = [r["val_loss"] for r in results["baseline"]] if "baseline" in results else None
    ref = statistics.mean(base) if base else None
    ref_s = statistics.mean(r["seconds"] for r in results["baseline"]) if base else None
    print(f"\\n{title}\\n{'variant':>44} {'n':>2} {'val':>8} {'sd':>7} {'d val (mnat)':>13} {'time x':>7}")
    for name in sorted(names, key=lambda n: statistics.mean(r["val_loss"] for r in results[n])):
        vals = [r["val_loss"] for r in results[name]]
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        d = 1000 * (statistics.mean(vals) - ref) if ref is not None else float("nan")
        t = statistics.mean(r["seconds"] for r in results[name]) / ref_s if ref_s else float("nan")
        print(f"{name:>44} {len(vals):>2} {statistics.mean(vals):8.4f} {sd:7.4f} {d:+13.1f} {t:7.2f}")
    print("(d val: against baseline, in millinats; negative is better. At the record's rate 1 mnat ~ 164 ms of "
          "8xH100 time,\\n before any per-step cost the idea adds: see 'time x'.)")
'''),
code('''
# 6. Round 1: every idea against the baseline.
BASE = Config(compile=AMP)  # torch.compile on bf16-capable GPUs
if QUICK:
    SEEDS, BASE = 1, BASE.update(steps=600)

CALIBRATION = ["polar_express", "normuon", "cautious_wd", "value_embeds", "unet", "x0_mix", "smear", "attn_gate",
               "bigram_hash", "mtp"]  # already in the record: these should beat the baseline
CANDIDATES = ["canon", "diff_attn", "snoo", "powercool", "zloss", "ngram_gate"]  # not in the record

round1 = run(["baseline"] + CALIBRATION + CANDIDATES, SEEDS, BASE)
table(round1, ["baseline"] + CALIBRATION, "CALIBRATION (in the record already: expect negative d val)")
table(round1, ["baseline"] + CANDIDATES, "CANDIDATES (not in the record)")
'''),
code('''
# 7. Round 2: the candidates that won, on top of the stack of calibration winners (the record has those already).
WIN_MNAT = 2.0  # a "win": at least this many millinats better than the baseline
mean_val = lambda name: statistics.mean(r["val_loss"] for r in round1[name])
ref = mean_val("baseline")
calib_wins = [n for n in CALIBRATION if 1000 * (mean_val(n) - ref) <= -WIN_MNAT]
# ngram_gate is measured against bigram_hash (it gates that table), the others against the baseline
cand_ref = lambda n: mean_val("bigram_hash") if n == "ngram_gate" else ref
cand_wins = [n for n in CANDIDATES if 1000 * (mean_val(n) - cand_ref(n)) <= -WIN_MNAT]
print("calibration winners:", calib_wins or "none")
print("candidate winners:  ", cand_wins or "none")
if calib_wins and cand_wins:
    STACK = "+".join(calib_wins)
    names = [STACK] + [STACK + "+" + c for c in cand_wins if c not in calib_wins]
    round2 = run(names, SEEDS, BASE)
    round2["baseline"] = round1["baseline"]
    table(round2, ["baseline"] + names, "ROUND 2: candidates on top of the calibration stack")
else:
    round2 = {}
    print("Round 2 skipped: it needs at least one calibration winner and one candidate winner.")
'''),
code('''
# 8. Summary: copy this whole output back to Claude.
summary = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "seeds": SEEDS,
           "first_seed": FIRST_SEED,
           "steps": BASE.steps, "runs": {}}
for res in (round1, round2):
    for name, rs in res.items():
        vals = [r["val_loss"] for r in rs]
        summary["runs"][name] = {"n": len(vals), "val": round(statistics.mean(vals), 5),
                                 "vals": [round(v, 5) for v in vals],
                                 "sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else None,
                                 "seconds": round(statistics.mean(r["seconds"] for r in rs), 1)}
runs = summary.pop("runs")
print("{" + json.dumps(summary)[1:-1] + ', "runs": {')
print(",\\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in runs.items()))
print("}}")
'''),
]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
