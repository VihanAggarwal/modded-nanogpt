"""The record folder of a finished attempt: every leg's log, statistics.py, and a README with every number filled in.

attempt.py calls build() after the certification pool. It rebuilds the folder from the phase ledgers each time, so
re-running is safe; it never touches a folder this attempt did not write (runs/record_folder.txt names it). The
README's numbers come from record_stats.py, which the folder carries as statistics.py, so `python statistics.py` in the
folder reproduces them from the logs.
"""
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import record_stats

HERE = Path(__file__).resolve().parent
MASTER_PUBLISHED_S = 39.9   # record #92 (ANVIL2) as published
PR379_PUBLISHED_S = 36.009  # #379's own same-node pool
DEFAULT_NAMES = {"stack": "CPLMNormHashesSystems", "mlonly": "CPLMNormHashes",
                 "streamret": "CPLMNormHashesSystemsStreamRetrieval"}
RETRIEVAL_RE = re.compile(r"^step:\d+ stream_retrieval val_loss_lm:([\d.]+) val_loss_mixed:([\d.]+) gain:(-?[\d.]+)mnat", re.M)

CHANGES_ML = """\
- **#379, CPLM** (copy-sink pointer LM): the next-token distribution is a mixture of the LM softmax and a pointer
  over the document's previous tokens (one QK-normed d=128 head with a learned sink key), fused into the fp8
  cross-entropy kernel, with an 8192-token copy band at validation. Record #92's 1194 trained steps become 1050
  (`NUM_SCHEDULED_ITERATIONS` 978). Code and description: PR #379.
- **#375, token-normalized n-gram hashes**: the bigram/trigram hashes read each token's normalization class
  (NFKC, accents stripped, lowercased, whitespace collapsed; `track_1_short/token_norm.py`) instead of its raw
  id. Embeddings, targets and the data stream keep raw GPT-2 ids. #375's own -15 steps were tuned on #92; here the
  step count is #379's, or the pre-registered cut below."""

CHANGES_SYSTEMS = """\
- **Systems patches** (no change to the model, optimizer, schedules or token stream; `tools/RULES_CHECK.md` in the fork):
  - the canonical-mask builder is spawned as a fresh interpreter at step 25 instead of forking the warmed-up
    trainer at t0 (the fork's copy-on-write faults sat on rank 0's launch path);
  - data loader: a numpy BOS index; the first shard's partial-index span is read before the first batch and the
    rest after; a failed shard load raises instead of hanging;
  - a loader thread overlaps the first shard read and the final validation's reads with GPU work (both stay on
    the clock); the pinned batch staging it shares takes a lock;
  - every rank starts its clock after a barrier, so rank 0's pre-clock mask setup no longer lets other ranks'
    on-clock work start before the reported clock."""


CHANGES_RETRIEVAL = """\
- **Stream-only retrieval at the final validation** (`STREAM_RETRIEVAL=1`, `track_1_short/stream_memory.py` with the
  C helper `stream_memory.c` and its parts `stream_lowtables.c` / `stream_pointer.c`): the CPLM probability p of each
  val token is mixed with next-token distributions read from memories of the tokens this run trained on, and nothing
  else: rank 0's loader passes the document spans it already computes for all 8 ranks to a C helper process, which
  reads them from the shards and indexes them (a hash chain over 6-token contexts; with `STREAM_RETRIEVAL_LOW=1` also
  exact count tables of orders 1-5) as batches are fetched, on the clock. At the last step the helper writes, before
  the clock stops, per val position: the memory's match record (#367's StreamIndex rule: up to 32 most recent
  occurrences, match levels 6-32), the low orders' counts, and per segment a pointer beam, its vote and a copy from
  source documents of the memory; each rank copies its rows to the GPU. In the untimed eval a stick-breaking chain of
  gated Kneser-Ney components over the orders (#380's recipe) and a softmax over [chain, pointer, vote, source copy]
  mix them with p, gated on target-independent features (counts, the model's entropy and log-probs, causal histories
  of earlier positions); constants fitted on a dev run's own last training batches at the record's step count (the
  same batches as the record runs' own last batches), never on val. No model, training or token-stream change: every
  log has `val_loss_lm` (the same weights unmixed) next to the mixed `val_loss`, the gain measured in-run."""

# Where the retrieval's gain comes from, measured on CPU proxies (tools/stream_retrieval/README.md): disclosed in
# every streamret record, since it is what a maintainer weighs when deciding whether the memory is acceptable.
RETRIEVAL_CONCENTRATION = (
    "- **Where the retrieval's gain comes from** (CPU proxies on the real 1050-step stream; not measured on this "
    "model): from val documents that share long verbatim passages with documents the run trained on, mostly web "
    "boilerplate. The top 1% of val documents carry 30-47% of the gain and the top 5% 65-92%; positions matched at 32 "
    "tokens are 0.3-0.4% of val and carry 31-37%, and at 97% of them every retrieved continuation is the target. The "
    "largest single contributor in the first 1M val tokens is a local-news site's \"most read\" sidebar whose 276-token "
    "list also appears in a trained document (10% of that proxy's whole gain). The memory never holds a val token: "
    "these are passages the training stream itself contains.")

RETRIEVAL_CREDITS = (
    "No code from another PR; the design builds on two. PR #367 (Herman Brunborg): exact-match retrieval from training "
    "data on this track, and its StreamIndex, whose stream layout (each rank's documents step by step, a STOP after "
    "each) and row rule (6-token key, the most recent occurrences, the deepest level reached, the next tokens of the "
    "occurrences at least that deep, (length, count, top share) as features) this memory follows. PR #380 (Deven): "
    "mixing CPLM's p at the output with retrieved count distributions under sigmoid gates fitted on training "
    "positions, chained over the match orders; the exact low-order tables (`STREAM_RETRIEVAL_LOW`) follow #380's "
    "recipe (exact counts, a gated chain over orders, fitted on training positions), with no #380 code. kNN-LM "
    "(Khandelwal et al., 2020), Infini-gram (Liu et al., 2024), interpolated Kneser-Ney (Chen & Goodman, 1998); the "
    "LZ77/zlib hash chain. This PR's part: the memories restricted to the run's own consumed stream and used at the "
    "final validation, the hash-chain index and C helper fed from the loader's spans on the clock, the records and "
    "their GPU-side gated chain with model-aware features, the pointer beam / vote / source copy, the fit on the run's "
    "own last batches. Nothing from #381.")


def gate_fit(arm_dir: Path, low: bool, trained: int) -> dict | None:
    """The 'fit' provenance of the streamret arm's constants for runs of `trained` steps (GATE_V2 / GATE_V2_LOW in its
    track_1_short/stream_memory.py, read from the source as the run logs embed it), or None."""
    block = "GATE_V2_LOW" if low else "GATE_V2"
    try:
        text = (arm_dir / "track_1_short/stream_memory.py").read_text()
        body = text[text.index(f"# {block}_BEGIN\n"):text.index(f"# {block}_END")]
        specs = json.loads(ast.literal_eval(re.search(rf"{block}_JSON = (\(.*\))\s*$", body, re.S).group(1)))
    except (OSError, ValueError, AttributeError, SyntaxError):
        return None
    return next((sp.get("fit") for sp in (specs if isinstance(specs, list) else [specs])
                 if (sp.get("fit") or {}).get("total_steps") == trained), None)


def gate_text(fit: dict | None, low: bool, trained: int) -> str:
    block = "`GATE_V2" + ("_LOW" if low else "") + "` in `stream_memory.py`"
    if not fit or fit.get("proxy"):
        return (f"- **WARNING: the gate's constants** ({block}) for {trained} trained steps are "
                + ("a placeholder fitted on a proxy model's outputs" if fit else "not in the arm's source")
                + ": the preflight refuses this; these runs do not meet the protocol.")
    return (f"- **The gate's constants** ({block}, the spec for {trained} trained steps) are hyperparameters, never "
            "fitted on val: `tools/stream_retrieval/fit_gate_v2.py` fitted them on the helper's rows of the last "
            f"{fit.get('batches', 16)} training batches of a {trained}-step dev run with `STREAM_RETRIEVAL_FIT` "
            "(queried against the memory as it stood before them) and on that run's model outputs there. The loader is "
            "deterministic, so those batches are these runs' own last batches; the model outputs are a dev run's of the "
            "same code and step count, not each record run's" + (f" ({fit['note']})" if fit.get("note") else "")
            + ". Provenance in the constants' `fit` field.")


def retrieval_gain(runs) -> str:
    """The in-run paired gain (val_loss_lm - val_loss on the same weights) over the finished runs' logs."""
    found = [m for r in runs if r.val is not None for m in RETRIEVAL_RE.findall(Path(r.path).read_text(errors="replace"))]
    if not found:
        return "n/a"
    gains = [float(g) for _, _, g in found]
    lm = sum(float(a) for a, _, _ in found) / len(found)
    sd = (sum((g - sum(gains) / len(gains)) ** 2 for g in gains) / max(len(gains) - 1, 1)) ** 0.5
    return (f"{sum(gains) / len(gains):.2f} millinats (sd {sd:.2f}, {len(gains)} runs; unmixed val_loss_lm mean "
            f"{lm:.5f})")


def git(directory: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def log_stack(log: str) -> dict:
    """Python, torch, CUDA, triton, driver and GPUs as the run log records them."""
    text = Path(log).read_text(errors="replace")
    find = lambda pattern: (m.group(1).strip() if (m := re.search(pattern, text, re.M)) else "?")
    gpus = re.findall(r"^\|\s+\d+\s+(NVIDIA [^|]+?)\s+(?:On|Off)\s+\|", text, re.M)
    return dict(python=find(r"^Running Python (\S+)"), torch=find(r"^Running PyTorch (\S+)"),
                torch_cuda=find(r"^Running PyTorch \S+ compiled for CUDA (\S+)"), triton=find(r"^Running Triton version (\S+)"),
                driver=find(r"Driver Version: (\S+)"), cuda=find(r"CUDA Version: (\S+)"),
                gpus=f"{len(gpus)}x {gpus[0]}" if gpus else "?")


def fmt(x: float, digits: int) -> str:
    return "n/a" if x != x else f"{x:.{digits}f}"


def collect(runs: Path, folder: Path, decision: dict) -> dict:
    """Copy every leg's log (and a crashed leg's stdout) into the folder; return what was found per phase.

    A leg that ended before the trainer wrote its run log (an import, NCCL or kernel-load failure) gets a stand-in
    log, crashed_<phase>_<arm>_leg<k>.txt, holding its stdout indented: it has no step lines, so statistics.py counts
    it as a run without a final validation, like any other crash."""
    found = dict(crashed=[], recovered=[], interrupted=[], hung=[])
    dest_of = {decision.get("arm"): "this_pr", "master": "baseline", "pr379": "baseline_pr379"}
    folder.mkdir(parents=True, exist_ok=True)
    for phase_dir in sorted(p for p in runs.iterdir() if (p / "ledger.jsonl").exists()):
        phase = phase_dir.name
        (folder / "ledgers").mkdir(exist_ok=True)
        for name in ("ledger.jsonl", "plan.json", "interrupted.jsonl"):
            if (phase_dir / name).exists():
                shutil.copy(phase_dir / name, folder / "ledgers" / f"{phase}_{name}")
        for record in read_jsonl(phase_dir / "ledger.jsonl"):
            if phase == "cert":
                sub = dest_of.get(record["arm"], f"cert/{record['arm']}")
            else:
                sub = f"{phase.split('_')[0]}/{record['arm']}"
            (folder / sub).mkdir(parents=True, exist_ok=True)
            stdout = Path(record["stdout"])
            if record["log"]:
                shutil.copy(record["log"], folder / sub)
            else:
                text = stdout.read_text(errors="replace") if stdout.exists() else ""
                (folder / sub / f"crashed_{phase}_{record['arm']}_leg{record['leg']:03d}.txt").write_text(
                    f"# {phase} leg {record['leg']} ({record['arm']}) ended (exit {record['exit_code']}) before the "
                    "trainer wrote its run log. It counts as a run without a final validation. Its stdout, indented:\n"
                    + "".join(f"    {line}\n" for line in text.splitlines()))
            leg = f"{phase} {record['arm']} leg {record['leg']}" + (" (timed out)" if record.get("timed_out") else "")
            if record.get("recovered"):
                found["recovered"].append(leg)
            if record.get("timed_out") and record.get("val_loss") is not None:
                found["hung"].append(leg)
            if record.get("val_loss") is None:
                found["crashed"].append(leg)
                (folder / "crashes").mkdir(exist_ok=True)
                if stdout.exists():
                    shutil.copy(stdout, folder / "crashes" / f"{phase}_{record['arm']}_leg{record['leg']:03d}.stdout")
        for note in read_jsonl(phase_dir / "interrupted.jsonl"):
            found["interrupted"].append(f"{phase} {note['arm']} leg {note['leg']}")
        if (phase_dir / "interrupted").is_dir():
            shutil.copytree(phase_dir / "interrupted", folder / "interrupted" / phase, dirs_exist_ok=True)
    return found


def arm_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| | runs (finished) | trained steps | train time (mean ± sd) | final val CE (mean ± sd) | p (val < 3.28) | runs > 3.28 |",
             "|---|---|---|---|---|---|---|"]
    for label, s in rows:
        steps = "/".join(map(str, s["steps"])) or "-"
        lines.append(f"| {label} | {s['n']} ({s['finished']}) | {steps} | {fmt(s['wall_mean'], 3)} ± {fmt(s['wall_sd'], 3)} s "
                     f"| {fmt(s['val_mean'], 5)} ± {fmt(s['val_sd'], 5)} | {s['p']:.2g} | {len(s['above_gate'])} |")
    return lines


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' * (n != 1)}"


def delta_line(label: str, c: dict | None) -> str:
    if c is None:
        return f"- {label}: fewer than 2 finished runs on a side."
    return (f"- {label}: train time **{c['wall']:+.3f} s** (se {c['wall_se']:.3f}, {c['wall_pct']:+.2f} %, Welch t = "
            f"{c['t']:.2f}, one-sided p = {c['p_faster']:.2g}); val {c['val_mnat']:+.2f} millinats (se "
            f"{c['val_se_mnat']:.2f}, two-sided p = {c['p_val']:.2g}); at equal loss (164 ms per millinat) "
            f"{c['adj']:+.3f} s (se {c['adj_se']:.3f}).")


def verdict(cand: dict, vs_master: dict | None, vs_pr379: dict | None, found: dict, flags: list[str],
            legs_pilot: int, cand_arm: str, increment: str, source: str) -> str:
    passed = cand["finished"] > 1 and cand["p"] < 0.01
    faster = lambda c: c is not None and c["wall"] < 0 and c["p_faster"] < 0.01
    margin = 1000 * (record_stats.GATE - cand["val_mean"])
    lines = ["VERDICT",
             f"  rule 2 (mean val < 3.28 at p < 0.01): {'PASS' if passed else 'FAIL'}  p = {cand['p']:.3g}, mean "
             f"{fmt(cand['val_mean'], 5)} over {cand['finished']} finished of {cand['n']} runs",
             f"  faster than master (rule 4): {'YES' if faster(vs_master) else 'NOT SHOWN'}"
             + (f"  {vs_master['wall']:+.3f} s (se {vs_master['wall_se']:.3f}, one-sided p = {vs_master['p_faster']:.2g})"
                if vs_master else ""),
             f"  faster than #379 (the increment over #379: {increment}): {'YES' if faster(vs_pr379) else 'NOT SHOWN'}"
             + (f"  {vs_pr379['wall']:+.3f} s (se {vs_pr379['wall_se']:.3f}, p = {vs_pr379['p_faster']:.2g}); val "
                f"{vs_pr379['val_mnat']:+.2f} millinats; at equal loss {vs_pr379['adj']:+.3f} s (se {vs_pr379['adj_se']:.3f})"
                if vs_pr379 else ""),
             f"  loss buffer: mean val {margin:.1f} millinats under 3.28 ({'keeps' if margin >= 2 else 'eats into'} the "
             "~2 millinat buffer)",
             f"  crashed legs: {len(found['crashed'])}" + (f" ({'; '.join(found['crashed'])})" if found["crashed"] else ""),
             "  flags: " + ("; ".join(flags) if flags else "none")]
    cand_crashes = [c for c in found["crashed"] if c.startswith(f"cert {cand_arm} ")]
    base_crashes = [c for c in found["crashed"] if c.startswith("cert ") and c not in cand_crashes]
    if base_crashes:
        lines.append(f"  baseline legs that crashed (counted in their arm, they do not block the PR): "
                     f"{'; '.join(base_crashes)}")
    ready = passed and faster(vs_master) and not cand_crashes
    lines.append(f"  READY FOR A PR: {'yes' if ready else 'no'}"
                 + ("" if ready else " (needs rule 2 PASS, faster than master, and no crashed certification leg of "
                                     "the candidate)"))
    if ready and not faster(vs_pr379):
        lines.append("  but the increment over #379 is not shown: maintainers credit #379's gain to #379, so this PR's own "
                     "claim would be ~0")
    lines.append(f"  PR source: {source}")
    lines += ["  caveats: the maintainers re-time on their own nodes; these deltas are same-node and same-session.",
              f"  The step count came from a {legs_pilot}-leg-per-arm pilot under the pre-registered rule (disclosed, not",
              "  pooled). #375 and #379 had not run together before this attempt. With a run-to-run val sd of ~1.6",
              f"  millinats, {cand['n']} runs pin the mean to about {1.6 / max(cand['n'], 1) ** 0.5:.1f} millinats (1 se)."]
    return "\n".join(lines)


def earlier_attempts(runs: Path, folder: Path) -> str:
    """The other attempts in this WORK (run.sh keeps every RUNS directly in WORK): disclosed, never pooled. Each one's
    ledgers and logs go to earlier_attempts/<its RUNS name>/, with its certification pool (if it reached one)
    recomputed from them."""
    notes = ""
    for other in sorted(p.parent for p in runs.parent.glob("*/attempt.json") if p.parent != runs):
        decision = json.loads((other / "decision.json").read_text()) if (other / "decision.json").exists() else {}
        dest = folder / "earlier_attempts" / other.name
        found = collect(other, dest, decision)
        for name in ("verdict.txt", "PREREGISTRATION.txt", "decision.json"):
            if (other / name).exists():
                shutil.copy(other / name, dest / name)
        verdict_lines = (other / "verdict.txt").read_text().splitlines() if (other / "verdict.txt").exists() else []
        if verdict_lines and verdict_lines[0].startswith("STOPPED"):
            outcome = verdict_lines[0]
        elif verdict_lines:
            outcome = "finished: " + "; ".join(
                re.sub(r"^record folder: .*/", "record folder: ", line.strip()) for line in verdict_lines
                if line.strip().startswith(("rule 2", "READY", "record folder")))
        else:
            outcome = "never finished (no verdict)"
        pool = record_stats.summarize(record_stats.load(str(dest / "this_pr")))
        cert = (f" Its certification pool ({decision.get('arm')}): {pool['n']} runs, {pool['finished']} finished, mean "
                f"val {fmt(pool['val_mean'], 5)}, p = {pool['p']:.3g}." if pool["n"] else " It never reached certification.")
        crashed = f" Crashed legs: {'; '.join(found['crashed'])}." if found["crashed"] else ""
        notes += (f"- **Another attempt on this node** (`{other.name}/`, not pooled; its ledgers and logs are in "
                  f"`earlier_attempts/{other.name}/`): {outcome.rstrip('.')}.{cert}{crashed}\n")
    return notes


def build(runs: Path, state: dict, decision: dict, arms: dict[str, Path], records_dir: Path, name: str | None,
          environment: Path | None) -> tuple[Path, str]:
    family, steps = decision["candidate"], decision["steps"]
    folder = records_dir / f"{state['date']}_{name or DEFAULT_NAMES[family]}"
    ours = runs / "record_folder.txt"
    mine = ours.read_text().strip() if ours.exists() else None
    if name is None and folder.exists() and str(folder) != mine:  # another attempt's folder holds the default name
        folder = folder.with_name(f"{folder.name}_{runs.name}")
    if folder.exists():
        if str(folder) != mine:
            raise SystemExit(f"{folder} exists and this attempt did not write it: move it away, or set RECORD_NAME")
        shutil.rmtree(folder)
    if mine and mine != str(folder) and Path(mine).is_dir():  # renamed (RECORD_NAME): no stale copy stays behind
        shutil.rmtree(mine)
    folder.mkdir(parents=True)
    ours.write_text(f"{folder}\n")
    found = collect(runs, folder, decision)
    shutil.copy(HERE / "record_stats.py", folder / "statistics.py")
    for name_ in ("PREREGISTRATION.txt", "decision.json"):
        shutil.copy(runs / name_, folder / name_)
    if environment is not None and environment.exists():
        shutil.copy(environment, folder / "environment.txt")
    env_text = (folder / "environment.txt").read_text() if (folder / "environment.txt").exists() else ""

    load = lambda sub: record_stats.load(str(folder / sub))
    this_pr, master, pr379 = load("this_pr"), load("baseline"), load("baseline_pr379")
    cand = record_stats.summarize(this_pr)
    s_master, s_pr379 = record_stats.summarize(master), record_stats.summarize(pr379)
    vs_master, vs_pr379 = record_stats.compare(this_pr, master), record_stats.compare(this_pr, pr379)
    pilot = record_stats.subdirs(str(folder), "pilot")
    smoke = record_stats.subdirs(str(folder), "smoke")
    shipped = record_stats.shipped(str(folder))
    s_shipped = record_stats.summarize([r for runs_ in shipped.values() for r in runs_])
    flags = [f for prefix, group in (("", {"this_pr": this_pr, "baseline": master, "baseline_pr379": pr379}),
                                     ("pilot/", pilot), ("smoke/", smoke))
             for n, r in group.items() for f in record_stats.flags(prefix + n, r)]

    heads = {k: git(v, "rev-parse", "HEAD") or "n/a" for k, v in arms.items()}
    cand_dir, stack_dir = arms[family], arms["stack"]
    rule3 = git(cand_dir, "diff", heads["master"], "--", "train_gpt.py", "track_1_short/") if heads["master"] != "n/a" else ""
    rule3_hits = [l for l in rule3.splitlines() if l.startswith("+") and re.search(r"_inductor|torch\.compile|dynamo\.config", l)]
    # The protocol's scripts live in the fork (the stack checkout), whichever candidate was certified.
    origin = re.sub(r"//[^/@]+@", "//", git(stack_dir, "remote", "get-url", "origin")) or "<this fork>"
    branch = git(stack_dir, "rev-parse", "--abbrev-ref", "HEAD") or "<branch>"
    if branch == "HEAD":
        branch = "<branch>"
    fork = f"{origin} @ `{heads['stack'][:7]}`"
    owner = m.group(1) if (m := re.search(r"github\.com[:/]([^/]+)/", origin)) else None
    first_done = next((r.path for r in this_pr if r.val is not None), None)
    stack_info = log_stack(first_done) if first_done else {}
    trained = steps + 72
    retrieval = family == "streamret"
    low = bool(state["settings"].get("streamret_low"))
    env_var = (("STREAM_RETRIEVAL=1 " + ("STREAM_RETRIEVAL_LOW=1 " if low else "")) if retrieval else "") \
        + f"NUM_SCHEDULED_ITERATIONS={steps}"
    if family == "mlonly":
        merged = {k: git(cand_dir, "rev-parse", ref) or "n/a" for k, ref in  # make_mlonly_arm.sh's two merges
                  (("master", "HEAD^1^1"), ("pr375", "HEAD^1^2"), ("pr379", "HEAD^2"), ("tree", "HEAD^{tree}"))}
        mlonly_what = (f"upstream master `{merged['master'][:7]}` + #375 `{merged['pr375'][:7]}` + #379 "
                       f"`{merged['pr379'][:7]}` (tree `{merged['tree'][:7]}`)")
        pr_source = (f"{cand_dir}: {mlonly_what.replace('`', '')}, a local merge with no systems patches "
                     "(make_mlonly_arm.sh rebuilds it), plus this record folder, which was written into the stack "
                     "checkout's records/")
        increment = "#375" + (", with the 963-step cut" if steps == 963 else "")
    elif retrieval:
        pr_source = (f"the streamret arm {cand_dir} (`{heads['streamret'][:7]}`: the stack {branch} @ {heads['stack'][:7]} "
                     "plus tools/stream_retrieval's overlay, one commit by make_streamret_arm.sh), plus this record "
                     "folder, which was written into the stack checkout's records/")
        increment = f"#375 + the systems patches + stream retrieval, with the {steps}-step cut"
    else:
        pr_source = f"the stack checkout {cand_dir} ({branch} @ {heads['stack'][:7]}), with this record folder in it"
        increment = "#375 + the systems patches" + (", with the 963-step cut" if steps == 963 else "")
    verdict_text = verdict(cand, vs_master, vs_pr379, found, flags, state["settings"]["legs_pilot"], decision["arm"],
                           increment, pr_source)
    published = (f"Record #92 is {MASTER_PUBLISHED_S} s as published and #379 measured {PR379_PUBLISHED_S} s on its own "
                 f"node; here they took {fmt(s_master['wall_mean'], 3)} s and {fmt(s_pr379['wall_mean'], 3)} s, so this node "
                 f"is {100 * (s_master['wall_mean'] / MASTER_PUBLISHED_S - 1):+.1f} % against #92's. The deltas above are "
                 "same-node (rule 4).")
    what = {"stack": "CPLM (#379) + token-normalized n-gram hashes (#375) + host-side systems patches",
            "streamret": "CPLM (#379) + token-normalized n-gram hashes (#375) + host-side systems patches + stream-only "
                         "retrieval"}.get(family, "CPLM (#379) + token-normalized n-gram hashes (#375)")
    source = (cand_dir / "train_gpt.py").read_text() if (cand_dir / "train_gpt.py").exists() else ""
    default = m.group(1) if (m := re.search(r'"NUM_SCHEDULED_ITERATIONS", "(\d+)"', source)) else "?"
    caches = "empty for every leg" if state["settings"]["cold"] else "warm after the first leg of each arm"
    systems_alone = ""
    if retrieval:
        base = decision.get("base", "")
        c = record_stats.compare(pilot[decision["arm"]], pilot[base]) if base in pilot and decision["arm"] in pilot else None
        systems_alone = (delta_line(f"Stream retrieval with its step cut (pilot only, not pooled: {decision['arm']} vs "
                                    f"{base})", c) + f"\n- Stream retrieval's in-run gain (val_loss_lm - val_loss, same "
                         f"weights) in the certification pool: {retrieval_gain(this_pr)}.")
    if family == "stack":
        c = (record_stats.compare(pilot["stack978"], pilot["mlonly978"])
             if "stack978" in pilot and "mlonly978" in pilot else None)
        systems_alone = (delta_line("Systems patches alone (pilot only, not pooled: stack978 vs mlonly978, both at 978 "
                                    "steps)", c) if c else
                         "- Systems patches alone: not measured on this node (no ML-only arm in the pilot).")

    def pilot_rows() -> list[str]:
        rows = ["| arm | runs (finished) | trained steps | train time (mean ± sd) | val per run | mean val |", "|---|---|---|---|---|---|"]
        for group, arms_ in (("smoke", smoke), ("pilot", pilot)):
            for arm, rs in arms_.items():
                s = record_stats.summarize(rs)
                per_run = ", ".join("crash" if r.val is None else f"{r.val:.4f}" for r in rs)
                rows.append(f"| {group} {arm} | {s['n']} ({s['finished']}) | {'/'.join(map(str, s['steps'])) or '-'} | "
                            f"{fmt(s['wall_mean'], 3)} ± {fmt(s['wall_sd'], 3)} s | {per_run} | {fmt(s['val_mean'], 5)} |")
        return rows

    n_int = len(found["interrupted"])
    interrupted = (f"{plural(n_int, 'leg')} {'was' if n_int == 1 else 'were'} cut short because the bench itself was "
                   f"stopped while {'it' if n_int == 1 else 'they'} ran ({'; '.join(found['interrupted'])}). Nothing in "
                   f"{'its log' if n_int == 1 else 'their logs'} reached the final validation; "
                   f"{'it' if n_int == 1 else 'each'} ran again at the same place in the plan, and what "
                   f"{'it' if n_int == 1 else 'they'} left is in `interrupted/`."
                   if n_int else "No leg was interrupted.")
    n_rec = len(found["recovered"])
    recovered = (f" {plural(n_rec, 'leg')} finished after the bench was stopped and {'was' if n_rec == 1 else 'were'} "
                 f"kept from {'its log' if n_rec == 1 else 'their logs'} ({'; '.join(found['recovered'])})."
                 if n_rec else "")
    crashed = (f"Crashed legs (counted, their stdout in `crashes/`): {'; '.join(found['crashed'])}."
               if found["crashed"] else "No leg crashed.")
    if found["hung"]:
        crashed += (f" Killed by the leg timeout after their final validation, so kept as finished runs: "
                    f"{'; '.join(found['hung'])}.")
    notes = "".join(f"- {n}\n" for n in state.get("notes", [])) + earlier_attempts(runs, folder)
    step_disclosure = (
        f"- **Step count.** Every candidate leg set `{env_var}` in its environment (ab_bench `--arm-env`), and each "
        f"log shows `step:N/{trained}`. "
        + ("That equals the default in `train_gpt.py`." if default == str(steps) and retrieval else
           "That equals the default in `train_gpt.py`, so the logged source is the shipped source." if default == str(steps)
           else f"The logged source's default is {default}: before merging, `train_gpt.py`'s default must become {steps}, "
           "the only difference from the logged source (as in #379's own README)."))
    all_runs = (f"- All {s_shipped['n']} runs of the shipped configuration ({' + '.join(f'`{k}/`' for k in shipped)}; "
                f"{s_shipped['finished']} finished): mean val {fmt(s_shipped['val_mean'], 5)}, p = {s_shipped['p']:.2g}. A "
                "check, not the claim: the pilot legs fed the pre-registered decision, so only the fresh pool above "
                "enters rule 2." if shipped else "")
    sxm = (", SXM (NVLink NV18 in `nvidia-smi topo -m`, `environment.txt`)" if "NV18" in env_text else "")
    if retrieval:
        cuts = ",".join(map(str, state["settings"].get("streamret_cuts") or [steps]))
        reproduce = [
            f"git clone -b {branch} {origin} modded-nanogpt && cd modded-nanogpt   # the fork: the stack and the protocol",
            f"{'STREAMRET_LOW=1 ' if low else ''}STREAMRET_CUTS={cuts} bash tools/record_attempt/run.sh   # the whole "
            "protocol, with the streamret arm",
            "# the certified code: the stack plus the stream-retrieval overlay (tools/stream_retrieval/arm):",
            "ARM=$(bash tools/stream_retrieval/make_streamret_arm.sh)   # prints ../record_work/streamret",
            f'cd "$ARM" && python data/cached_fineweb10B.py 9 && {env_var} ./run.sh',
            f"python records/track_1_short/{folder.name}/statistics.py   # from the fork, where this folder was written",
        ]
    elif family == "mlonly":
        reproduce = [
            f"git clone -b {branch} {origin} modded-nanogpt && cd modded-nanogpt   # the fork: the protocol's scripts",
            "bash tools/record_attempt/run.sh   # the whole protocol: setup, smoke, pilot, decision, certification, this folder",
            f"# the certified code, {mlonly_what.replace('`', '')}:",
            "ARM=$(bash tools/record_attempt/make_mlonly_arm.sh)   # prints ../record_work/mlonly",
            f'cd "$ARM" && python data/cached_fineweb10B.py 9 && {env_var} ./run.sh',
            f"python records/track_1_short/{folder.name}/statistics.py   # from the fork, where this folder was written",
        ]
    else:
        reproduce = [
            f"git clone -b {branch} {origin} modded-nanogpt && cd modded-nanogpt",
            "bash tools/record_attempt/run.sh   # the whole protocol: setup, smoke, pilot, decision, certification, this folder",
            "# one run of the certified configuration:",
            "python data/cached_fineweb10B.py 9",
            f"{env_var} ./run.sh",
            f"python records/track_1_short/{folder.name}/statistics.py",
        ]
    lines = [
        f"# {what}: {fmt(cand['wall_mean'], 3)} s",
        "",
        f"GPT-2 (124M-class) on FineWeb10B, 8×H100, track 1 (≤ 3.28 val CE). Built on record #92 (master "
        f"`{heads['master'][:7]}`) and the open PRs #379 and #375, credited below. {steps} scheduled steps, "
        f"{trained} trained.",
        "",
        "**Certification pool (one 8×H100 node, one session, legs interleaved, every leg counted):**",
        "",
        *arm_table([("**this PR**", cand), (f"record #92 (master `{heads['master'][:7]}`)", s_master),
                    (f"PR #379 (`{heads['pr379'][:7]}`)", s_pr379)]),
        "",
        f"- Rule 2: `scipy.stats.ttest_1samp(vals, 3.28, alternative='less')`: t = {cand['t']:.2f}, **p = {cand['p']:.2g}** "
        f"({cand['finished']} finished of {cand['n']} runs). {crashed}",
        delta_line("vs record #92 (rule 4)", vs_master),
        delta_line(f"vs PR #379 (the increment over #379: {increment}; #375 is an open PR credited on its own)", vs_pr379),
        *[line for line in (systems_alone, all_runs) if line],
        f"- {published}",
        f"- Logs: `this_pr/` ({cand['n']}), `baseline/` ({s_master['n']}), `baseline_pr379/` ({s_pr379['n']}); "
        "`python statistics.py` recomputes every number in this README from them.",
        "",
        "```",
        verdict_text,
        "```",
        "",
        "## Changes",
        "",
        CHANGES_ML,
        *([CHANGES_SYSTEMS, CHANGES_RETRIEVAL] if retrieval else [CHANGES_SYSTEMS] if family == "stack" else
          [f"- No systems patches: the ML-only arm, {mlonly_what}, was certified (`tools/record_attempt/make_mlonly_arm.sh` "
           f"in the fork, {fork}, builds it)."]),
        f"- **Step count**: {steps} scheduled steps ({trained} trained), chosen by the pre-registered rule below.",
        "",
        "## Credits",
        "",
        "- #379 (CPLM): @NathanGodey and @yoavartzi.",
        "- #375 (token-normalized n-gram hashes): Daniel Monroe.",
        f"- {'Systems patches, integration' if family in ('stack', 'streamret') else 'Integration'} and this "
        f"certification: {'@' + owner if owner else '(author)'}.",
        *([f"- Stream-only retrieval: {'@' + owner if owner else '(author)'}. {RETRIEVAL_CREDITS}"] if retrieval else []),
        "",
        "## Rules checklist",
        "",
        "1. **Data pipelines untouched.** #375 normalizes ids only inside the n-gram hashes; #379 changes the output "
        "distribution only." + (" The systems patches change when and on which thread the loader reads, not what it "
                                "yields: `tools/tests/test_token_stream_identical.py` (in the fork, " + fork + ") replays "
                                "the record's schedule (1122 scheduled steps) and the stack's at 978, plus the validation, "
                                "through master's loader and this one: byte-identical"
                                + (" (963 is not replayed there)." if steps == 963 else ".")
                                if family in ("stack", "streamret") else "")
        + (" Stream retrieval reads the span lists the loader already computes and nothing else; its memory is a verbatim "
           "copy of the tokens this run trained on (every fetched timed batch, no unread shard byte, no val token) that "
           "returns exact continuations at validation, a nonparametric store unlike the learned n-gram table (whether it "
           "is acceptable is a question for the maintainers); `tools/stream_retrieval/test_stream_retrieval.py` checks "
           "the batches stay byte-identical with the tap on, the memory equals every rank's trained tokens, and nothing "
           "else is read."
           if retrieval else ""),
        f"2. **Mean val ≤ 3.28 at p < 0.01**: p = {cand['p']:.2g} over {cand['finished']} runs, all counted "
        f"({'PASS' if cand['p'] < 0.01 else 'FAIL'})."
        + (" The retrieval mixture is a valid probability model: every component sums to <= 1 over the vocabulary and "
           "depends on the memory, val[<= t] and earlier positions' outcomes only, every gate on target-independent "
           "features only, and stick-breaking and the softmax are convex combinations, so the mixture sums to <= 1 at "
           "every position; it is forward-only, nothing is learned from val, and the memories are built and queried "
           "inside the timed region."
           if retrieval else ""),
        "3. **No new compile or inductor flags**: `git diff " + heads["master"][:7] + " -- train_gpt.py track_1_short/ "
        "| grep -E '_inductor|torch.compile|dynamo.config'` adds "
        + ("nothing." if rule3 and not rule3_hits else
           f"{len(rule3_hits)} lines: check them." if rule3_hits else "(not checked: not a git checkout)."),
        "4. **Faster than the prior record on the same hardware**: "
        + (f"{vs_master['wall']:+.3f} s against master, same node, same session, interleaved." if vs_master else "n/a"),
        "",
        f"**Loss buffer (discretionary rule 2).** Mean val {1000 * (record_stats.GATE - cand['val_mean']):.1f} millinats "
        f"under 3.28"
        + (f"; {vs_pr379['val_mnat']:+.2f} millinats against #379 on this node." if vs_pr379 else ".")
        + (" The 963-step cut spends part of #375's margin; the pre-registered rule allowed it only with a pilot mean "
           "≤ 3.2765 (≥ 3.5 millinats under the gate)." if steps == 963 else ""),
        "",
        "## Pre-registration and development history (full disclosure)",
        "",
        f"The rule was written to `PREREGISTRATION.txt` at {state['created']}, before the first leg (with the default "
        f"leg counts it is also in `tools/record_attempt/README.md` in the fork, {fork}):",
        "",
        "```",
        state["rule"].rstrip(),
        "```",
        "",
        "Every leg of every phase is in this folder: `smoke/`, `pilot/` (not pooled), then the certification pool.",
        "",
        *pilot_rows(),
        "",
        "Decision (`decision.json`): " + "; ".join(decision["reasons"]) + ".",
        "",
        "## Other disclosures",
        "",
        step_disclosure,
        f"- **Interruptions.** {interrupted}{recovered}",
        f"- **Crashes.** {crashed}",
        "- **Seeds.** Unseeded: `TRAIN_SEED` was unset in every leg, as in #92's certification.",
        f"- **Compile caches** were {caches}; "
        "compilation, warmup and graph capture are untimed in every arm.",
        "- **Off-by-default code in the logged source**: #379's development knobs (`CPLM_*`, `ALLOW_4_GPUS`, `FA3_*`); "
        "`run.sh` unset them, so every leg ran the defaults.",
        "- **#375's normalization map** is built at import, before the clock (~45-70 ms); the record's convention puts "
        "tokenizer-derived tables on the clock, so a reviewer may ask for it to move.",
        "- **#379's validation NLL** is `-log(p + 1e-9)`, within 5e-5 nats of a normalized model (disclosed in #379).",
        *([f"- **Stream retrieval is off by default in the logged source**: every candidate leg set "
           f"`{env_var.rsplit(' ', 1)[0]}`. "
           "Before merging, the record settings at the top of `train_gpt.py` must also set it (with the step count), the "
           "only other difference from the logged source.",
           "- **Before the clock** the stream retrieval only compiles its C helper (`cc -O2`), spawns it and lets it "
           "allocate and prefault its arrays (~5.3 GB of host RAM with its query arenas"
           + (", plus ~13 GB for the low-order tables" if low else "") + ") and its rows file (in /dev/shm), "
           "maps that file on every rank, and sizes the model's eval side-output buffers: the same kind of setup as the "
           "canonical mask's buffer. Everything it computes happens on the clock: insertion as batches are fetched"
           + (" (the low-order tables on 8 insertion threads, ~4 cores on average)" if low else "") + ", the val read "
           "and the queries after the last step, the copy to the GPUs before the clock stops. A checksum of each rank's "
           "val chunk is verified after the clock stops (a check only).",
           gate_text(gate_fit(cand_dir, low, trained), low, trained),
           "- **The C helper** (`track_1_short/stream_memory.c`, `stream_lowtables.c`, `stream_pointer.c`, ~3,400 lines "
           "with comments; C11 and pthreads, no new dependency) is new to the repo; the run logs embed it with the "
           "Python source.",
           RETRIEVAL_CONCENTRATION]
          if retrieval else []),
        notes.rstrip(),
        "",
        "## Hardware and stack",
        "",
        f"- {stack_info.get('gpus', '?')}{sxm}, driver {stack_info.get('driver', '?')} (CUDA {stack_info.get('cuda', '?')}); "
        f"Python {stack_info.get('python', '?')}, torch {stack_info.get('torch', '?')} (CUDA {stack_info.get('torch_cuda', '?')}), "
        f"triton {stack_info.get('triton', '?')} (from the run logs).",
        "- FA3 `devenpzak/flash-attn3-12864` @ 64c1e6d1, the in-code sha256 check passed in every finished leg.",
        *(["- `environment.txt`: the node as `run.sh` found it (GPUs, topology, CPU, memory, packages)."]
          if env_text else []),
        f"- Arms: master `{heads['master']}`, #379 `{heads['pr379']}`, candidate `{heads[family]}`"
        + (f" (a local merge: {mlonly_what})." if family == "mlonly" else "."),
        "",
        "## Reproduce",
        "",
        f"The protocol's scripts (`tools/record_attempt/`, `tools/speedrun_ab/`) are in the fork, {fork}.",
        "",
        "```bash",
        *reproduce,
        "```",
        "",
        "## Files",
        "",
        "- `this_pr/`, `baseline/`, `baseline_pr379/`: the certification pool (one log per leg; each embeds the full "
        "source; a leg that died before its log has a `crashed_*.txt` stand-in with its stdout).",
        "- `pilot/<arm>/`, `smoke/<arm>/`: the development legs. `crashes/`, `interrupted/`: what failed legs left, if any.",
        "- `ledgers/`: each phase's plan and ledger (leg order, exit codes, start times). `decision.json`: the rule's "
        "inputs and outcome.",
        *(["- `earlier_attempts/`: every other attempt on this node, disclosed above, never pooled."]
          if (folder / "earlier_attempts").exists() else []),
        "- `statistics.py` (needs scipy), `PREREGISTRATION.txt`" + (", `environment.txt`." if env_text else "."),
    ]
    (folder / "README.md").write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n")
    return folder, verdict_text
