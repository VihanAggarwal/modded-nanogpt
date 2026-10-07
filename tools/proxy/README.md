# Proxy screen for architecture and optimizer ideas (one GPU, Colab works)

The record needs 8xH100. `proxy_gpt.py` screens ideas first on one GPU: a compact speedrun-style GPT
(RoPE, QK-norm, ReLU^2, zero-init projections, Muon + Adam, logit softcap) trained on the same FineWeb
shards. Every variant sees the same tokens in the same order, and under one seed every parameter it shares with
the baseline starts the same (except in `swiglu`, which reshapes the MLP, and `tied`, which re-draws the
embedding), so per-seed val-loss differences come from the idea and from trajectory noise.
Ideas that win clearly here go to the 8xH100 sweep (`tools/speedrun_ab`).

**Easiest:** upload a notebook to Colab (File → Upload notebook), pick the biggest GPU runtime and Run all. Both
are self-contained: the code is inline, and the data comes from Hugging Face. With two GPUs, run a second copy
with `FIRST_SEED = 2` for twice the seeds in the same time. Paste the Summary cell's output back.

- `nanogpt_canon_screen.ipynb` (round 3): does Canon survive on a record-like base (REC: Polar Express, NorMuon,
  cautious WD, value embeddings, x0 mix, attention gate, smear and the record's partial key offset), which form
  (sites A/C, kernel 2/3/4, pre-norm / re-normed, from layer 1, a learned key offset `canon_bk`, a V shift), and is
  it cheap? 15 arms x 2 seeds, about 35-40 min on an RTX PRO 6000 or A100, plus ~12 min for a 3x-steps check.
- `nanogpt_proxy_screen.ipynb` (rounds 1-2): the record's own techniques as calibration, plus new candidates,
  then the winners re-tested on a stack of the calibration winners.

Regenerate both after editing `proxy_gpt.py`: `python tools/proxy/make_notebook.py`. Run keys carry a per-notebook
prefix, so results of different notebooks and rounds never mix. Tests (CPU, ~15 s):
`python -m pytest tools/proxy/test_proxy.py -q`.

Or, from a clone, in a Colab notebook with an A100, L4 or H100 runtime (a T4 works but is ~10x slower):

```
!git clone -b claude/nanogpt-optimization-n49ur2 https://github.com/VihanAggarwal/modded-nanogpt
%cd modded-nanogpt
!pip install -q huggingface_hub tiktoken
!python data/cached_fineweb10B.py 1
!python tools/proxy/proxy_gpt.py --seeds 2 --set compile=1 --variants baseline,polar_express,normuon,cautious_wd,snoo,powercool,value_embeds,unet,smear,attn_gate,bigram_hash,trigram_hash,ngram_gate,mtp,swiglu
```

- `data/cached_fineweb10B.py 1` downloads the val shard and one train shard (~400 MB). One run reads 20M tokens.
- The default is 8 layers, d=512 (~77M parameters with the embeddings), 1200 steps of 16k tokens: about a
  minute per run on an A100 or RTX PRO 6000, so the list above (15 variants x 2 seeds) takes about 35 minutes.
- Combine ideas with `+`, e.g. `value_embeds+unet+smear`. Override any `Config` field with `--set key=value`,
  e.g. `--set steps=2000` or `--set key_offset_layers=2,5`.
- Paste the final table back. `d val (mnat)` is each variant's val loss against the first variant, in millinats;
  `ms/step x` is the steady-state step time (compile and warmup excluded). Seed noise at this scale is tens of
  millinats per run, so compare per seed and trust only differences well beyond it.
- Each new config starts from an empty dynamo cache and compiles with `fullgraph=True`: past dynamo's
  recompile limit (8 configs) it used to fall back to eager silently, which made later runs 2.7-3.5x slower.

Every variant is a few lines in `VARIANTS` and `Config`; add new ideas there. Optional parameters are built
without drawing random numbers (`torch.zeros`, or a forked RNG for random tables), so they keep the init pairing.
