"""Build the self-contained Colab notebooks from proxy_gpt.py (library part inlined): the general screen
(nanogpt_proxy_screen.ipynb, rounds 1-2) and the Canon screen on a record-like base (nanogpt_canon_screen.ipynb).

    python tools/proxy/make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "proxy_gpt.py")).read()
lib = src[:src.index("\ndef main():")]
lib = lib[lib.index('"""', 3) + 3:].lstrip("\n")          # drop the module docstring (CLI usage)
lib = "# 4. The proxy: model, optimizers, training loop, and the ideas as flags (VARIANTS).\n" + lib.rstrip() + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


GPU_CHECK = code('''
# 1. GPU check, and the one package Colab lacks (PyTorch and NumPy are preinstalled).
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip install -q huggingface_hub
''')

DATA = code('''
# 3. Data: the speedrun's own GPT-2-tokenized FineWeb from Hugging Face: the val shard and 1 of the 103 train
# shards (~400 MB). One run reads 20M tokens.
import os
from huggingface_hub import hf_hub_download

os.makedirs(DATA_DIR, exist_ok=True)
for name in ("fineweb_val_000000.bin", "fineweb_train_000001.bin"):
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=name, repo_type="dataset", local_dir=DATA_DIR)
print(sorted(os.listdir(DATA_DIR)))
''')

# Shared by both notebooks; KEY_PREFIX keeps their results (and those of older rounds) apart.
RUNNER = '''
# 5. Runner and analysis. Resumable: a run is keyed by its arm, seed and scale; finished runs are skipped.
import json, math, os, statistics
import numpy as np

KEY_PREFIX = "@KEY_PREFIX@"  # this notebook and code version: results of other notebooks and rounds never match


def combine(name):
    changes = {}
    for part in name.split("+"):
        changes.update(VARIANTS[part])
    return changes


def run_key(name, seed, cfg):
    return f"{KEY_PREFIX}|{name}|{seed}|{cfg.steps}|{cfg.n_layer}|{cfg.d_model}|{cfg.batch_seqs}|{cfg.seq_len}"


def load_results():
    done = {}
    if os.path.exists(RESULTS_FILE):
        for line in open(RESULTS_FILE):
            r = json.loads(line)
            done[r["key"]] = r
    return done


def run(names, seeds, base):
    """Every arm in order, all of this copy's seeds each (one compile per arm)."""
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
                r.update(key=key, variant=name, seed=seed, steps=cfg.steps, torch=torch.__version__,
                         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
                with open(RESULTS_FILE, "a") as f:
                    f.write(json.dumps(r) + "\\n")
                done[key] = r
                print(f"   val {r['val_loss']:.4f}  ema {r['val_loss_ema'] or float('nan'):.4f}  "
                      f"{r['ms_per_step']:.1f} ms/step  (first step {r['first_step_s']:.0f}s, total {r['seconds']:.0f}s)",
                      flush=True)
            if not math.isfinite(done[key]["val_loss"]):
                print(f"   {name} seed {seed} diverged: left out of the analysis", flush=True)
            results.setdefault(name, []).append(done[key])
    return results


try:
    from scipy.stats import t as _student_t
    def t_sf(x, df): return float(_student_t.sf(x, df))
except ImportError:  # normal approximation: anti-conservative at small dof
    def t_sf(x, df): return 0.5 * math.erfc(x / math.sqrt(2))


def by_seed(rs, key="val_loss"):
    return {r["seed"]: r[key] for r in rs if r.get(key) is not None and math.isfinite(r[key])}  # not diverged


def paired(results, name, ref, key="val_loss"):
    """Per-seed differences name - ref in millinats (same seed = same init of every shared parameter)."""
    a, b = by_seed(results[name], key), by_seed(results[ref], key)
    seeds = sorted(set(a) & set(b))
    d = [1000 * (a[s] - b[s]) for s in seeds]
    n = len(d)
    mean = statistics.mean(d) if d else float("nan")
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if n > 1 and se > 0 else float("nan")
    p = 2 * t_sf(abs(t), n - 1) if n > 1 and se > 0 else float("nan")
    return dict(n=n, mean=mean, se=se, t=t, p=p, seeds=seeds, diffs=d)


def two_way(results, names, key="val_loss"):
    """Additive model val[v, s] = mu + variant_v + seed_s + noise on the complete variant x seed grid. The seed
    effect is estimated from every variant; the noise sd is pooled over (V-1)(S-1) dof (vs n-1 for one pair).
    None with fewer than 2 seeds common to every variant."""
    seeds = sorted(set.intersection(*(set(by_seed(results[n], key)) for n in names)))
    if len(names) < 2 or len(seeds) < 2:
        return None
    Y = 1000 * np.array([[by_seed(results[n], key)[s] for s in seeds] for n in names])
    V, S = Y.shape
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    df = (V - 1) * (S - 1)
    sigma = math.sqrt((resid ** 2).sum() / df)
    return dict(seeds=seeds, S=S, df=df, sigma=sigma, means=dict(zip(names, Y.mean(1))),
                seed_effect=dict(zip(seeds, Y.mean(0) - Y.mean())))


def contrast(tw, name, ref):
    """Two-way estimate of name - ref: (d, se, p)."""
    d, se = tw["means"][name] - tw["means"][ref], tw["sigma"] * math.sqrt(2 / tw["S"])
    return d, se, 2 * t_sf(abs(d / se), tw["df"]) if se > 0 else float("nan")


def ms_ratio(results, name, ref):
    """Steady-state ms/step of name over ref, paired by seed (each pair ran in one session, on one GPU)."""
    a, b = by_seed(results[name], "ms_per_step"), by_seed(results[ref], "ms_per_step")
    seeds = set(a) & set(b)
    return statistics.mean(a[s] / b[s] for s in seeds) if seeds else float("nan")


def paired_table(results, names, title, ref_of=lambda n: "baseline", key="val_loss"):
    rows = [n for n in names if ref_of(n) != n and ref_of(n) in results]
    if not rows:
        return
    grid = list(dict.fromkeys([ref_of(n) for n in rows] + rows))
    count = {n: len(by_seed(results[n], key)) for n in grid}  # an arm with a diverged run leaves the two-way grid
    tw = two_way(results, [n for n in grid if count[n] == max(count.values())], key) if len(grid) > 2 else None
    wn, wr = max(len(n) for n in rows), max(len(ref_of(n)) for n in rows)
    print(f"\\n{title}  [{key}, millinats; d = mean per-seed difference vs ref; negative is better]")
    print(f"{'variant':>{wn}} {'ref':>{wr}} {'n':>2} {'d':>7} {'se':>5} {'p':>6} | {'2-way d':>7} {'se':>5} "
          f"{'p':>6} | {'ms/step x':>9}  per-seed d")
    for n in rows:
        ref = ref_of(n)
        s = paired(results, n, ref, key)
        in_tw = tw and {n, ref} <= set(tw["means"])
        tw_cols = "{:+7.1f} {:5.1f} {:6.3f}".format(*contrast(tw, n, ref)) if in_tw else ""
        print(f"{n:>{wn}} {ref:>{wr}} {s['n']:>2} {s['mean']:+7.1f} {s['se']:5.1f} {s['p']:6.3f} | "
              f"{tw_cols:>20} | {ms_ratio(results, n, ref):9.3f}  " + " ".join(f"{x:+.0f}" for x in s["diffs"]))
    if tw:
        print(f"2-way (variant + seed): pooled noise sd {tw['sigma']:.1f} mnat per run on {tw['df']} dof; seed effects "
              + ", ".join(f"s{s} {e:+.0f}" for s, e in tw["seed_effect"].items()))


def table(results, names, title, ref_of=lambda n: "baseline"):
    paired_table(results, names, title, ref_of)
    paired_table(results, names, title + " -- tail-EMA weights", ref_of, key="val_loss_ema")


def summary_lines(*results):
    """One compact JSON line per arm and step count, every seed listed: what to paste back (copies' lines pool)."""
    arms = {}
    for res in results:
        for name, rs in res.items():
            arms[(name, rs[0]["steps"])] = rs
    rnd = lambda x, n: None if x is None else round(x, n)
    return [json.dumps(dict(arm=name, steps=steps, first_seed=FIRST_SEED, seeds=[r["seed"] for r in rs],
                            vals=[rnd(r["val_loss"], 5) for r in rs], vals_ema=[rnd(r["val_loss_ema"], 5) for r in rs],
                            ms_per_step=[rnd(r["ms_per_step"], 2) for r in rs],
                            seconds=[rnd(r.get("seconds"), 1) for r in rs],  # total, compile included
                            first_step_s=[rnd(r.get("first_step_s"), 1) for r in rs],
                            compiled_graphs=[r.get("compiled_graphs") for r in rs],  # 1 for an arm's first seed, then 0
                            gpu=",".join(sorted({r["gpu"] for r in rs})), torch=",".join(sorted({r["torch"] for r in rs}))))
            for (name, steps), rs in arms.items()]
'''


def runner(prefix):
    return code(RUNNER.replace("@KEY_PREFIX@", prefix))


# ------------------------------------------------------------------------------------------------ general screen

SCREEN = [
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

Under one seed every variant starts from the same init of every parameter it shares with the baseline, so the
tables compare per seed (paired), and also fit variant + seed effects over all variants (two-way), for the final
weights and for a tail EMA of the weights. "ms/step x" is the steady-state step time against the reference, with
compile and warmup excluded.

Each run appends to a results file, and **re-running skips finished runs**: after a disconnect, Run all resumes
where it stopped. Colab wipes `/content` when it recycles a runtime, so set `SAVE_TO_DRIVE = True` to keep the
results through that. Round 1 takes about 35-45 minutes on an A100 (less on an H100); round 2 adds 10-20 min.

**Two GPUs?** Open a second copy of this notebook on the other GPU and set `FIRST_SEED = 2` there. It runs
different seeds of the same variants, so the two summaries together give twice the seeds in the same wall time.

**When it finishes, copy the last cell's output (the summary) back to Claude**, from each copy if you ran two.

A proxy win is a screen, not proof: the record is ~124M params plus an 84.6M-row n-gram table trained on ~330M
tokens. Ideas that win here, especially in round 2, are the ones worth 8xH100 time. The Canon follow-up on a
record-like base is `nanogpt_canon_screen.ipynb`.
'''),
GPU_CHECK,
code('''
# 2. Settings
SEEDS = 2           # seeds per variant in this copy. Same seed = same init of every shared parameter (paired)
FIRST_SEED = 0      # a second copy on another GPU: set 2 here so it runs seeds 2 and 3
QUICK = False       # True: 1 seed and 600 steps per run, a ~20-minute first look (noisier). Use on a T4.
DATA_DIR = "/content/fineweb10B"
RESULTS_FILE = "/content/proxy_results.jsonl"  # every finished run is appended; re-running skips them
SAVE_TO_DRIVE = False  # True: keep results in Google Drive across disconnects (asks for permission)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_FILE = f"/content/drive/MyDrive/proxy_results_s{FIRST_SEED}.jsonl"
'''),
DATA,
code(lib),
runner("r3"),
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
table(round1, ["baseline"] + CANDIDATES, "CANDIDATES (not in the record)",
      ref_of=lambda n: "bigram_hash" if n == "ngram_gate" else "baseline")  # ngram_gate gates the bigram table
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
    print("STACK =", STACK)
    short = {n.replace(STACK, "STACK"): rs for n, rs in round2.items()}  # narrower table rows
    table(short, [n.replace(STACK, "STACK") for n in names],
          "ROUND 2: candidates on top of the calibration stack (paired with the stack)",
          ref_of=lambda n: "STACK" if n != "STACK" else "baseline")
else:
    round2 = {}
    print("Round 2 skipped: it needs at least one calibration winner and one candidate winner.")
'''),
code('''
# 8. Summary: copy this whole output back to Claude (one line per variant, every seed).
print("\\n".join(summary_lines(round1, round2)))
'''),
]

# ------------------------------------------------------------------------------------------------ Canon screen

CANON = [
md('''
# NanoGPT speedrun, round 3: does Canon survive on a record-like base?

**What this tests, and why.** In rounds 1-2 of this one-GPU proxy, Canon layers (Allen-Zhu 2025: a causal
per-channel 4-tap convolution with a residual, before attention and before the MLP; zero-init here) were the
biggest new win: -207 millinats (mnat) of val loss on the plain baseline, and -48 to -92 on a stack of record
techniques. But those bases lacked two record features that overlap Canon: the **partial key offset** (each key's
non-rotating dims come from the previous token, a hard-coded one-tap Canon on the keys, PR #169) and the
**hashed n-gram table**. This notebook re-tests on **REC**, a record-like base: Polar Express, NorMuon, cautious
weight decay, value embeddings, x0 mix, attention gate, smear and the key offset. It asks:

1. **Does Canon survive?** `REC+canon` against `REC`, and `REC+trigram_hash+canon` against `REC+trigram_hash`.
2. **Which form?** Sites: `canon_a` (before attention only) and `canon_c` (before the MLP only). Kernel size:
   `canon_k2` and `canon_k3`. Placement of the norm: `canon_prenorm` is norm(x + conv(x)) and `canon_renorm` is
   norm(n + conv(n)) with n = norm(x); these two fit the record's static fp8 scales. `canon_from1` skips layer 0.
   `canon_bk` is a learned key offset: Canon on the keys of every layer, starting as the record's hard offset where
   that is on. `v_shift` is Canon on the values.
3. **Is it cheap?** Steady-state ms/step against REC, with compile and warmup excluded.

`REC+rec_rep` re-runs REC under the same seeds, so its differences measure the GPU's run-to-run nondeterminism.
That is the floor under every comparison. With `RUN_LONG`, REC and REC+canon also run at 3x the steps, to see
whether the gain shrinks with more tokens.

**Fixed since rounds 1-2.** Their losses stand, but their time column does not: after 8 distinct configs, dynamo
silently ran every new config eagerly, which made it 2.7-3.5x slower. Each config now compiles fresh, and step time
is measured in the steady state. Under one seed every arm now starts from the same init of every shared parameter,
so arms are compared per seed. A tail EMA of the weights is evaluated next to the final weights.

**How to run:**
1. Upload this notebook to Colab (File → Upload notebook).
2. Runtime → Change runtime type → the biggest GPU offered (RTX PRO 6000, H100 or A100).
3. Runtime → **Run all**.

It needs no code from GitHub. The data, about 400 MB of the speedrun's GPT-2-tokenized FineWeb, comes from
Hugging Face.

**Two GPUs:** open a second copy on the other GPU and set `FIRST_SEED = 2` in cell 2. It runs seeds 2 and 3 of the
same arms, and the two summaries pool into 4 seeds. To see the pooled verdict, paste one copy's summary into the
last cell of the other.

**Time per copy:** the main cell is 15 arms × 2 seeds = 30 runs of 1200 steps (20M tokens each). At about 50 s per
run on an RTX PRO 6000 or A100, plus up to a minute of compile per arm, that is **about 35-40 minutes**. The
long-horizon cell (2 arms × 2 seeds × 3600 steps) adds **about 12 minutes**. An H100 is faster. Every finished run
is appended to a results file and **re-running skips finished runs**: after a disconnect, Run all resumes where it
stopped. Colab wipes `/content` when it recycles a runtime, so set `SAVE_TO_DRIVE = True` to keep the results
through that. The most important arms run first.

**What to paste back to Claude:** the **Summary** cell's output (one JSON line per arm) from each copy. The verdict
table is welcome too.

A proxy win is a screen, not proof: the record is ~124M params plus an 84.6M-row n-gram table, trained on ~330M
tokens with fp8 matmuls.
'''),
GPU_CHECK,
code('''
# 2. Settings
SEEDS = 2           # seeds per arm in this copy. Same seed = same init of every shared parameter (paired)
FIRST_SEED = 0      # in a second copy (another GPU) set 2: it runs seeds 2 and 3, and the two summaries pool
RUN_LONG = True     # also REC vs REC+canon at 3x the steps (~12 more minutes): does the gain shrink with tokens?
DATA_DIR = "/content/fineweb10B"
RESULTS_FILE = "/content/canon_results.jsonl"  # every finished run is appended; re-running skips them
SAVE_TO_DRIVE = False  # True: keep results in Google Drive across disconnects (asks for permission)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_FILE = f"/content/drive/MyDrive/canon_results_s{FIRST_SEED}.jsonl"
'''),
DATA,
code(lib),
runner("c3"),
code('''
# 6. Round 3: Canon on the record-like base REC. Arms in order of importance; each runs all of this copy's seeds
# before the next starts. For a verdict on an unfinished copy, cut ARMS to the finished arms: re-running trains nothing.
assert AMP, "needs a bf16 GPU (RTX PRO 6000, H100, A100, L4): Runtime -> Change runtime type"
BASE = Config(compile=AMP)  # torch.compile on bf16-capable GPUs
REC = "polar_express+normuon+cautious_wd+value_embeds+x0_mix+attn_gate+smear+key_offset"
VARIANTS["REC"] = combine(REC)
ARMS = ["REC", "REC+canon", "REC+canon_a", "REC+canon_c", "REC+canon_bk",  # the main questions first
        "REC+rec_rep",                                                       # REC again: the noise floor
        "REC+canon_k2", "REC+canon_k3", "REC+canon_prenorm", "REC+canon_renorm", "REC+canon_from1",
        "REC+v_shift", "REC+canon+canon_bk", "REC+trigram_hash", "REC+trigram_hash+canon"]
REF = lambda arm: "REC+trigram_hash" if arm == "REC+trigram_hash+canon" else "REC"

main = run(ARMS, SEEDS, BASE)
table(main, ARMS, "ROUND 3: every arm against REC (REC+trigram_hash+canon against REC+trigram_hash)", ref_of=REF)
'''),
code('''
# 7. Long horizon: REC and REC+canon at 3x the steps (59M tokens), same seeds. Does Canon's gain shrink?
long = {}
if RUN_LONG:
    long = run(["REC", "REC+canon"], SEEDS, BASE.update(steps=3 * BASE.steps))
    table(long, ["REC", "REC+canon"], f"LONG HORIZON ({3 * BASE.steps} steps)", ref_of=lambda arm: "REC")
'''),
code('''
# 8. Verdict: each arm against its reference, on the final and the tail-EMA weights; then the three questions.
def estimate(results, tw, weights, key):
    """sum_a w_a val[a] in millinats, e.g. {arm: 1, ref: -1}, over the seeds of the two-way fit tw: (d, se, p, own).
    se is the two-way one (noise pooled over every arm), or the per-seed one (S-1 dof) where larger (own=True): the
    two-way model assumes every arm reacts to a seed alike, which a structural change such as Canon need not."""
    vals = {a: by_seed(results[a], key) for a in weights}
    d_s = [1000 * sum(w * vals[a][s] for a, w in weights.items()) for s in tw["seeds"]
           if all(s in v for v in vals.values())]
    S = len(d_s)
    if S < 2:
        return float("nan"), float("nan"), float("nan"), False
    d, df, own = statistics.mean(d_s), tw["df"], False
    se = tw["sigma"] * math.sqrt(sum(w * w for w in weights.values()) / S)
    if statistics.stdev(d_s) / math.sqrt(S) > se:
        se, df, own = statistics.stdev(d_s) / math.sqrt(S), S - 1, True
    return d, se, 2 * t_sf(abs(d / se), df) if se > 0 else float("nan"), own


def call(cf, ce):
    """BETTER/WORSE: both readouts p < 0.05 in the same direction; better?/worse?: one of them, the other agreeing."""
    sig = [c for c in (cf, ce) if c[2] < 0.05]
    if not sig or cf[0] * ce[0] <= 0:
        return "mixed" if sig else "no clear effect"
    word = "BETTER" if cf[0] < 0 else "WORSE"
    return word if len(sig) == 2 else word.lower() + "?"


col = lambda c: f"{c[0]:+7.1f} {c[1]:5.1f}{'*' if c[3] else ' '} {c[2]:6.3f}"
txt = lambda c: f"{c[0]:+.1f} ± {c[1]:.1f}{'*' if c[3] else ''} (p {c[2]:.3f})"


def verdict(results, arms, ref_of, long=None):
    n = {a: len(by_seed(results[a])) for a in arms if a in results}
    arms = [a for a in n if n[a] == max(n.values())]
    if len(arms) < len(n):
        print("left out (fewer seeds than the rest: diverged or unfinished):", ", ".join(a for a in n if a not in arms))
    f, e = two_way(results, arms), two_way(results, arms, "val_loss_ema")
    if f is None or e is None:
        print("The verdict needs >= 2 seeds of every arm.")
        return
    readouts = lambda w, res=results: (estimate(res, f, w, "val_loss"), estimate(res, e, w, "val_loss_ema"))
    rows = [a for a in arms if ref_of(a) != a and ref_of(a) in arms]
    print(f"\\nVERDICT on {f['S']} seeds {f['seeds']}: millinats, negative = better. se: the two-way model's (noise "
          f"pooled over all arms: {f['sigma']:.1f} final, {e['sigma']:.1f} EMA per run, {f['df']} dof), or * the "
          f"arm's own per-seed one where larger ({f['S'] - 1} dof"
          + ("; pooling the other copy's seeds, last cell, gives firmer calls" if f["S"] < 4 else "")
          + f").\\n{len(rows)} comparisons: expect about {0.05 * len(rows):.1f} of them at p < 0.05 by chance alone.")
    print(f"{'arm':>24} {'vs':>16} | {'d final':>7} {'se':>6} {'p':>6} | {'d EMA':>7} {'se':>6} {'p':>6} | "
          f"{'ms/step x':>9} | verdict")
    for arm in rows:
        cf, ce = readouts({arm: 1, ref_of(arm): -1})
        note = "replicate: the noise floor" if arm.endswith("rec_rep") else call(cf, ce)
        print(f"{arm:>24} {ref_of(arm):>16} | {col(cf)} | {col(ce)} | {ms_ratio(results, arm, ref_of(arm)):9.3f} | "
              + note)
    canons = [a for a in rows if ("canon" in a or "v_shift" in a) and ref_of(a) == "REC"]
    if "REC+canon" in canons:
        print("\\nWHICH FORM? Each Canon arm against REC+canon (A+C, k=4, after the norm: the paper's form)")
        for arm in canons:
            if arm != "REC+canon":
                cf, ce = readouts({arm: 1, "REC+canon": -1})
                print(f"{arm:>24} | final {txt(cf)} | EMA {txt(ce)} | {call(cf, ce)}")
        mde = 2.8 * f["sigma"] * math.sqrt(2 / f["S"])  # 80% power at p < 0.05, at the two-way se
        print(f"A difference under ~2.8 se (~{mde:.0f} mnat final here) easily goes unseen: 'no clear effect' does not "
              "make two forms equal.")
        cf, ce = readouts({"REC+canon": 1, "REC": -1})
        print(f"\\n1. Does Canon survive on REC? REC+canon: {txt(cf)} final, {txt(ce)} EMA: {call(cf, ce)}")
        if "REC+trigram_hash+canon" in rows:
            cf, ce = readouts({"REC+trigram_hash+canon": 1, "REC+trigram_hash": -1})
            print(f"   ... with an n-gram table (vs REC+trigram_hash): {txt(cf)} final, {txt(ce)} EMA: {call(cf, ce)}")
            cf, ce = readouts({"REC+trigram_hash+canon": 1, "REC+trigram_hash": -1, "REC+canon": -1, "REC": 1})
            print(f"   overlap, Canon's gain with the table minus without (positive: the table takes some of it): "
                  f"{txt(cf)} final, {txt(ce)} EMA")
            if f["means"]["REC+trigram_hash"] >= f["means"]["REC"]:
                print("   but REC+trigram_hash is no better than REC: unlike the record's table (gated, at several "
                      "layers), this one does not help, so the overlap says little about the record")
        best = min(canons, key=lambda a: f["means"][a] + e["means"][a])
        print(f"2. Best form (mean of both readouts): {best}: {f['means'][best] - f['means']['REC']:+.1f} final, "
              f"{e['means'][best] - e['means']['REC']:+.1f} EMA against REC (forms within ~2 se of it are a tie)")
        drift = (f"; REC+rec_rep, REC itself later in the session, reads "
                 f"{ms_ratio(results, 'REC+rec_rep', 'REC'):.3f}: the drift" if "REC+rec_rep" in rows else "")
        print("3. Cost, steady-state ms/step against REC: "
              + ", ".join(f"{a[4:]} {ms_ratio(results, a, 'REC'):.3f}" for a in canons) + drift)
    if "REC+rec_rep" in rows:
        rep = paired(results, "REC+rec_rep", "REC")["diffs"]
        sd = math.sqrt(statistics.mean(x * x for x in rep) / 2)  # a replicate's true difference is 0
        print(f"Noise: same-seed replicate differences {' '.join(f'{x:+.1f}' for x in rep)} mnat, i.e. GPU "
              f"nondeterminism ~{sd:.1f} mnat per run; all noise sources together: {f['sigma']:.1f} per run")
    if long and "REC+canon" in long and "REC" in long and "REC+canon" in arms:
        res = {**results, "REC@3x": long["REC"], "REC+canon@3x": long["REC+canon"]}
        far = readouts({"REC+canon@3x": 1, "REC@3x": -1}, res)
        change = readouts({"REC+canon@3x": 1, "REC@3x": -1, "REC+canon": -1, "REC": 1}, res)
        print(f"Long horizon: REC+canon - REC at 3x the steps {txt(far[0])} final, {txt(far[1])} EMA; the change from "
              f"the base steps {txt(change[0])} final, {txt(change[1])} EMA (positive: the gain shrinks with tokens)")


verdict(main, ARMS, REF, long)
'''),
code('''
# 9. Summary: copy this whole output back to Claude (from both copies if you ran two).
SUMMARY = "\\n".join(summary_lines(main, long))
print(SUMMARY)
'''),
code('''
# 10. Optional: pool both copies. Paste the other copy's Summary output between the quotes and run this cell to get
# the verdict on every seed. (A copy's session and GPU are part of its seeds' effects, which the two-way model removes.)
OTHER_COPY = """
"""


def from_summary(text, steps):
    """{arm: [run, ...]} with the fields the analysis reads, from summary lines at this step count."""
    out = {}
    for line in text.splitlines():
        if line.strip().startswith('{"arm"'):
            s = json.loads(line)
            if s["steps"] == steps:
                out.setdefault(s["arm"], []).extend(
                    dict(seed=seed, val_loss=v, val_loss_ema=ema, ms_per_step=ms)
                    for seed, v, ema, ms in zip(s["seeds"], s["vals"], s["vals_ema"], s["ms_per_step"]))
    return out


seeds_in = lambda text: {seed for line in text.splitlines() if line.strip().startswith('{"arm"')
                         for seed in json.loads(line)["seeds"]}
common = seeds_in(SUMMARY) & seeds_in(OTHER_COPY)
if common:  # the same (arm, seed) twice: one copy's runs would silently replace the other's
    print(f"Both summaries have seeds {sorted(common)}: not pooled. Set FIRST_SEED = 2 in one of the two copies.")
elif OTHER_COPY.strip():
    both = SUMMARY + "\\n" + OTHER_COPY
    verdict(from_summary(both, BASE.steps), ARMS, REF, from_summary(both, 3 * BASE.steps))
else:
    print("To pool: paste the other copy's summary into OTHER_COPY and run this cell again.")
'''),
]


def write(cells, name):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    path = os.path.join(HERE, name)
    json.dump(nb, open(path, "w"), indent=1)
    print("wrote", path, len(cells), "cells")


write(SCREEN, "nanogpt_proxy_screen.ipynb")
write(CANON, "nanogpt_canon_screen.ipynb")
