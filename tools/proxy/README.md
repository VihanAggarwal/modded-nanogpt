# Proxy screen for architecture and optimizer ideas (one GPU, Colab works)

The record needs 8xH100. `proxy_gpt.py` screens ideas first on one GPU: a compact speedrun-style GPT
(RoPE, QK-norm, ReLU^2, zero-init projections, Muon + Adam, logit softcap) trained on the same FineWeb
shards. Every variant sees the same tokens in the same order, so the val-loss differences come from the idea
and from seed noise. Ideas that win clearly here go to the 8xH100 sweep (`tools/speedrun_ab`).

In a Colab notebook with an A100, L4 or H100 runtime (a T4 works but is ~10x slower):

```
!git clone -b claude/nanogpt-optimization-n49ur2 https://github.com/VihanAggarwal/modded-nanogpt
%cd modded-nanogpt
!pip install -q huggingface_hub tiktoken
!python data/cached_fineweb10B.py 1
!python tools/proxy/proxy_gpt.py --seeds 2 --set compile=1 --variants baseline,polar_express,normuon,cautious_wd,snoo,powercool,value_embeds,unet,smear,attn_gate,bigram_hash,trigram_hash,ngram_gate,mtp,swiglu
```

- `data/cached_fineweb10B.py 1` downloads the val shard and one train shard (~400 MB). One run reads 20M tokens.
- The default is 8 layers, d=512 (~77M parameters with the embeddings), 1200 steps of 16k tokens. That takes
  about 1-3 minutes per run on an A100, so the list above (15 variants x 2 seeds) takes about an hour.
- Combine ideas with `+`, e.g. `value_embeds+unet+smear`. Override any `Config` field with `--set key=value`,
  e.g. `--set steps=2000`.
- Paste the final table back. `d val (mnat)` is each variant's val loss against the first variant, in millinats.
  Seed noise at this scale is a few millinats, so trust only differences well beyond the `+/-` column.

Every variant is a few lines in `VARIANTS` and `Config`; add new ideas there.
