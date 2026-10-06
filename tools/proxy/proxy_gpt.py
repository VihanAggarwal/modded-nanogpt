"""A one-GPU proxy for screening architecture and optimizer ideas before spending 8xH100 time.

The record's trainer needs 8 H100s (FA3, fp8, an n-gram table sharded 8 ways). This is a compact GPT in the
speedrun's style -- RoPE, QK-norm, ReLU^2, zero-init projections, untied head, logit softcap, Muon on the hidden
matrices and Adam elsewhere -- that trains on the same FineWeb .bin shards on any single GPU (Colab works) or,
tiny, on CPU. Each idea is a flag (see VARIANTS); every variant sees the same tokens in the same order, so
differences in final val loss come from the idea (and seed noise: run >= 2 seeds).

A proxy win is evidence, not proof: the record is ~124M params + an 84.6M-row n-gram table trained on ~330M
tokens. Promote ideas that win clearly here to the 8xH100 sweep (tools/speedrun_ab).

    python tools/proxy/proxy_gpt.py --variants baseline,polar_express,value_embeds --seeds 2 --data data/fineweb10B
"""
import argparse
import dataclasses
import glob
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = 50256
VOCAB = 50304  # 50257 padded to a multiple of 128, as the record


@dataclass
class Config:
    # scale (the default fits a 16 GB GPU and runs a few minutes on an A100)
    n_layer: int = 8
    d_model: int = 512
    head_dim: int = 128
    seq_len: int = 1024
    batch_seqs: int = 16          # 16k tokens/step: the fp32 logits stay ~3 GB
    steps: int = 1200
    val_tokens: int = 2 * 1024 * 1024
    # optimization
    lr_hidden: float = 0.03       # Muon
    lr_embed: float = 0.3         # Adam: embeddings
    lr_head: float = 0.008        # Adam: lm_head
    lr_scalar: float = 0.02       # Adam: scalars and small vectors
    momentum: float = 0.95
    weight_decay: float = 0.0
    cooldown_frac: float = 0.5    # decay to lr_floor over the last fraction of steps
    lr_floor: float = 0.1
    cooldown_power: float = 1.0   # 1 = linear; >1 decays faster early in the cooldown ("PowerCool")
    snoo_every: int = 0           # >0: Snoo-style outer Nesterov step on slow weights every this many steps
    snoo_lr: float = 0.7
    snoo_momentum: float = 0.5
    warmup_steps: int = 0
    # ideas (flags)
    orth: str = "ns5"             # ns5 | polar (Polar Express) | none (plain momentum SGD, normalized)
    normuon: bool = False         # per-row second-moment normalization after orthogonalization
    cautious_wd: bool = False     # weight decay only where update and weight agree in sign
    hidden_opt: str = "muon"      # muon | adam
    softcap: float = 15.0
    act: str = "relu2"            # relu2 | gelu | swiglu
    value_embeds: int = 0         # number of value-embedding tables mixed into V of the first/last layers
    unet: bool = False            # skip connections from the first half of the layers to the second half
    x0_mix: bool = False          # learnable mix of the embedding back into every block's input
    smear: bool = False           # gated 1-token look-back on the embeddings
    attn_gate: bool = False       # per-head sigmoid output gate from the block input
    bigram_rows: int = 0          # hashed bigram embedding table rows (0 = off)
    trigram_rows: int = 0         # hashed trigram embedding table rows (0 = off)
    ngram_gate: bool = False      # content-aware sigmoid gate on the hashed n-gram embeddings (Engram-style)
    mtp: float = 0.0              # weight of a t+2 prediction loss through the same head
    zloss: float = 0.0            # weight of the softmax normalizer z-loss
    qk_norm: bool = True
    attn_scale: float = 0.0       # 0 = 1/sqrt(head_dim)
    tie_embed: bool = False
    seed: int = 0
    compile: bool = False

    def update(self, **kw):
        return dataclasses.replace(self, **kw)


# Each variant: what it changes relative to `baseline`. Add ideas here; keep them one flag or a few each.
VARIANTS = {
    "baseline": {},
    "polar_express": {"orth": "polar"},
    "normuon": {"normuon": True},
    "cautious_wd": {"cautious_wd": True, "weight_decay": 0.025},
    "plain_wd": {"weight_decay": 0.025},
    "adam_hidden": {"hidden_opt": "adam", "lr_hidden": 0.003},
    "gelu": {"act": "gelu"},
    "swiglu": {"act": "swiglu"},
    "value_embeds": {"value_embeds": 2},
    "unet": {"unet": True},
    "x0_mix": {"x0_mix": True},
    "smear": {"smear": True},
    "attn_gate": {"attn_gate": True},
    "bigram_hash": {"bigram_rows": 1 << 18},
    "mtp": {"mtp": 0.3},
    "zloss": {"zloss": 1e-4},
    "softcap30": {"softcap": 30.0},
    "no_qk_norm": {"qk_norm": False},
    "cooldown_0.8": {"cooldown_frac": 0.8},
    "tied": {"tie_embed": True, "lr_embed": 0.01},  # the shared matrix is the head too: a head-sized lr
    "momentum_0.9": {"momentum": 0.9},
    "lr_hidden_x1.5": {"lr_hidden": 0.045},
    "powercool": {"cooldown_power": 2.0},
    "snoo": {"snoo_every": 8},
    "trigram_hash": {"bigram_rows": 1 << 18, "trigram_rows": 1 << 18},
    "ngram_gate": {"bigram_rows": 1 << 18, "ngram_gate": True},
}


# ------------------------------------------------------------------------------------------------ data

def load_tokens(path: str, limit: int | None = None) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f"{path}: not a FineWeb .bin shard"
    n = int(header[2]) if limit is None else min(int(header[2]), limit)
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(n,))


class TrainStream:
    """Consecutive windows of seq_len + 1 tokens through the train shards, in file order: every variant and
    every seed reads the same tokens (seeds change the init only)."""

    def __init__(self, pattern: str, seq_len: int):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"no shards match {pattern}"
        self.seq_len, self.file, self.pos = seq_len, 0, 0
        self.tokens = load_tokens(self.files[0])

    def next(self, batch_seqs: int) -> torch.Tensor:
        n = batch_seqs * self.seq_len + 1
        if self.pos + n > len(self.tokens):
            self.file = (self.file + 1) % len(self.files)
            self.tokens, self.pos = load_tokens(self.files[self.file]), 0
        chunk = torch.from_numpy(self.tokens[self.pos:self.pos + n].astype(np.int64))
        self.pos += n - 1
        return chunk


# ------------------------------------------------------------------------------------------------ model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        freqs = (1 / 1024) ** torch.linspace(0, 1, dim // 4)
        freqs = torch.cat([freqs, freqs.new_zeros(dim // 4)])  # half the dims unrotated (as the record)
        theta = torch.outer(torch.arange(max_len, dtype=torch.float32), freqs)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):  # [B, T, H, D]
        cos, sin = self.cos[None, :x.size(1), None, :].type_as(x), self.sin[None, :x.size(1), None, :].type_as(x)
        x1, x2 = x.float().chunk(2, dim=-1)
        x1, x2 = x1.type_as(x), x2.type_as(x)
        return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.cfg, self.layer, self.n_head = cfg, layer, d // hd
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = 4 * d if cfg.act != "swiglu" else int(8 * d / 3) // 64 * 64
        self.fc = nn.Linear(d, hidden * (2 if cfg.act == "swiglu" else 1), bias=False)
        self.out = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.out.weight)
        self.rotary = Rotary(hd, cfg.seq_len)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))  # (x, x0) mix, used with x0_mix
        self.ve_lambda = nn.Parameter(torch.tensor(0.5))
        if cfg.attn_gate:
            self.gate = nn.Linear(12, self.n_head, bias=False)
            nn.init.zeros_(self.gate.weight)

    def forward(self, x, x0, ve):
        if self.cfg.x0_mix:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        B, T, _ = x.shape
        h = norm(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_head, -1).unbind(2)
        if self.cfg.qk_norm:
            q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.ve_lambda * v + (1 - self.ve_lambda) * ve.view_as(v)
        scale = self.cfg.attn_scale or None
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           scale=scale).transpose(1, 2)
        if self.cfg.attn_gate:
            y = y * torch.sigmoid(self.gate(h[..., :12]) + 2.0)[..., None]  # starts near open
        x = x + self.proj(y.reshape(B, T, -1))
        h = self.fc(norm(x))
        if self.cfg.act == "relu2":
            h = F.relu(h).square()
        elif self.cfg.act == "gelu":
            h = F.gelu(h)
        else:
            a, g = h.chunk(2, dim=-1)
            h = F.silu(g) * a
        return x + self.out(h)


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)
        nn.init.zeros_(self.head.weight)
        if cfg.tie_embed:
            nn.init.normal_(self.embed.weight, std=0.02)  # it is also the head now: small logits at init
            self.head.weight = self.embed.weight
        self.ve = nn.ModuleList(nn.Embedding(VOCAB, cfg.d_model) for _ in range(cfg.value_embeds))
        if cfg.unet:
            self.skip_w = nn.Parameter(torch.ones(cfg.n_layer // 2))
        if cfg.smear:
            self.smear_gate = nn.Linear(12, 1, bias=False)
            nn.init.zeros_(self.smear_gate.weight)
        if cfg.bigram_rows:
            self.bigram = nn.Embedding(cfg.bigram_rows, cfg.d_model)
            nn.init.zeros_(self.bigram.weight)
        if cfg.trigram_rows:
            self.trigram = nn.Embedding(cfg.trigram_rows, cfg.d_model)
            nn.init.zeros_(self.trigram.weight)
        if cfg.ngram_gate:
            self.ngram_gate_w = nn.Linear(12, 1, bias=False)
            nn.init.zeros_(self.ngram_gate_w.weight)

    def value_embed_for(self, layer: int, ves):
        # U-net pattern: table i feeds layer i and layer n_layer - 1 - i
        n = len(ves)
        if layer < n:
            return ves[layer]
        if layer >= self.cfg.n_layer - n:
            return ves[self.cfg.n_layer - 1 - layer]
        return None

    def forward(self, idx, targets):
        cfg = self.cfg
        x = self.embed(idx)
        if cfg.bigram_rows:
            prev = F.pad(idx[:, :-1], (1, 0), value=BOS)
            ngram = self.bigram((prev * 36313 + idx * 27191) % cfg.bigram_rows)
            if cfg.trigram_rows:
                prev2 = F.pad(idx[:, :-2], (2, 0), value=BOS)
                ngram = ngram + self.trigram((prev2 * 1000003 + prev * 36313 + idx * 27191) % cfg.trigram_rows)
            if cfg.ngram_gate:  # starts open (2 * sigmoid(0) = 1): the gate learns what to trust
                ngram = ngram * 2 * torch.sigmoid(self.ngram_gate_w(x[..., :12]))
            x = x + ngram
        if cfg.smear:
            x = torch.cat([x[:, :1], x[:, 1:] + torch.sigmoid(self.smear_gate(x[:, 1:, :12])) * x[:, :-1]], dim=1)
        x = x0 = norm(x)
        ves = [ve(idx) for ve in self.ve]
        skips = []
        half = cfg.n_layer // 2
        for i, block in enumerate(self.blocks):
            if cfg.unet and i >= cfg.n_layer - half:
                x = x + self.skip_w[cfg.n_layer - 1 - i] * skips.pop()
            x = block(x, x0, self.value_embed_for(i, ves) if ves else None)
            if cfg.unet and i < half:
                skips.append(x)
        logits = self.head(norm(x)).float()
        logits = cfg.softcap * torch.tanh(logits / cfg.softcap)
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.reshape(-1))
        if self.training:
            if cfg.mtp:
                loss = loss + cfg.mtp * F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), targets[:, 1:].reshape(-1))
            if cfg.zloss:
                loss = loss + cfg.zloss * torch.logsumexp(logits, dim=-1).square().mean()
        return loss


# ------------------------------------------------------------------------------------------------ optimizer

NS5 = [(3.4445, -4.7750, 2.0315)] * 5
# Polar Express (Amsel et al. 2025) coefficients as used in the record (#38), with the safety factor folded in.
POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(G: torch.Tensor, coeffs, safety: float = 1.0) -> torch.Tensor:
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() * safety + 1e-7)
    for a, b, c in coeffs:
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if transpose else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, cfg: Config):
        super().__init__(params, dict(lr=cfg.lr_hidden))
        self.cfg = cfg

    @torch.no_grad()
    def step(self):
        cfg = self.cfg
        coeffs = POLAR_EXPRESS if cfg.orth == "polar" else NS5
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state["buf"] = torch.zeros_like(p)
                    if cfg.normuon:
                        state["v"] = torch.zeros(p.size(0), 1, device=p.device)
                buf = state["buf"].lerp_(p.grad, 1 - cfg.momentum)
                g = p.grad.lerp(buf, cfg.momentum)  # Nesterov
                u = orthogonalize(g, coeffs, 1.01 if cfg.orth == "polar" else 1.0) if cfg.orth != "none" else g / (g.norm() + 1e-7) * math.sqrt(g.numel() / max(g.shape))
                if cfg.normuon:
                    before = u.norm()
                    state["v"].lerp_(u.square().mean(dim=1, keepdim=True), 1 - 0.95)
                    u = u / (state["v"].sqrt() + 1e-8)
                    u = u * (before / (u.norm() + 1e-8))
                u = u * max(1, p.size(0) / p.size(1)) ** 0.5
                lr = group["lr"]
                if cfg.weight_decay:
                    decay = lr * cfg.weight_decay * p
                    if cfg.cautious_wd:
                        decay = decay * (u * p > 0)
                    p.sub_(decay)
                p.add_(u.to(p.dtype), alpha=-lr)


def build_optimizers(model: GPT, cfg: Config):
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "blocks" in n and "gate" not in n]
    hidden_ids = {id(p) for p in hidden}
    embeds = [p for n, p in model.named_parameters() if ("embed" in n or n.startswith("ve.") or "gram" in n)
              and "gate" not in n and id(p) not in hidden_ids]
    head = [] if cfg.tie_embed else [model.head.weight]
    taken = hidden_ids | {id(p) for p in embeds + head}
    scalars = [p for p in model.parameters() if id(p) not in taken]
    adam = torch.optim.Adam([dict(params=embeds, lr=cfg.lr_embed), dict(params=head, lr=cfg.lr_head),
                             dict(params=scalars, lr=cfg.lr_scalar)], betas=(0.8, 0.95), eps=1e-10)
    if cfg.hidden_opt == "adam":
        adam.add_param_group(dict(params=hidden, lr=cfg.lr_hidden))
        opts = [adam]
    else:
        opts = [adam, Muon(hidden, cfg)]
    for opt in opts:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return opts


def lr_mult(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    cd_start = int(cfg.steps * (1 - cfg.cooldown_frac))
    if step < cd_start:
        return 1.0
    t = (step - cd_start) / max(1, cfg.steps - cd_start)
    return cfg.lr_floor + (1.0 - cfg.lr_floor) * (1 - t) ** cfg.cooldown_power


# ------------------------------------------------------------------------------------------------ run

@torch.no_grad()
def evaluate(model: GPT, val: np.ndarray, cfg: Config, device) -> float:
    model.eval()
    n = cfg.val_tokens // cfg.seq_len
    losses = []
    for i in range(0, n, cfg.batch_seqs):
        rows = min(cfg.batch_seqs, n - i)
        chunk = torch.from_numpy(val[i * cfg.seq_len:(i + rows) * cfg.seq_len + 1].astype(np.int64)).to(device)
        x, y = chunk[:-1].view(rows, cfg.seq_len), chunk[1:].view(rows, cfg.seq_len)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses.append(model(x, y).item() * rows)
    model.train()
    return sum(losses) / n


def train(cfg: Config, data_dir: str, device, log_every: int = 100) -> dict:
    torch.manual_seed(cfg.seed)
    model = GPT(cfg).to(device)
    opts = build_optimizers(model, cfg)
    fwd = torch.compile(model) if cfg.compile else model
    stream = TrainStream(f"{data_dir}/fineweb_train_*.bin", cfg.seq_len)
    # Snoo: slow weights take a Nesterov step toward the fast weights every snoo_every steps; fast restart there.
    slow = [p.detach().clone() for p in model.parameters()] if cfg.snoo_every else None
    outer_m = [torch.zeros_like(p) for p in model.parameters()] if cfg.snoo_every else None
    val = load_tokens(sorted(glob.glob(f"{data_dir}/fineweb_val_*.bin"))[0], cfg.val_tokens + 1)
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        chunk = stream.next(cfg.batch_seqs).to(device, non_blocking=True)
        x = chunk[:-1].view(cfg.batch_seqs, cfg.seq_len)
        y = chunk[1:].view(cfg.batch_seqs, cfg.seq_len)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = fwd(x, y)
        loss.backward()
        mult = lr_mult(step, cfg)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * mult
            opt.step()
        model.zero_grad(set_to_none=True)
        if cfg.snoo_every and (step + 1) % cfg.snoo_every == 0:
            with torch.no_grad():
                for p, s, m in zip(model.parameters(), slow, outer_m):
                    delta = p - s  # the inner steps' displacement (the negative outer gradient)
                    m.mul_(cfg.snoo_momentum).add_(delta)
                    s.add_(delta + cfg.snoo_momentum * m, alpha=cfg.snoo_lr)
                    p.copy_(s)
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{cfg.steps} train loss {loss.item():.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    val_loss = evaluate(model, val, cfg, device)
    tokens = cfg.steps * cfg.batch_seqs * cfg.seq_len
    return dict(val_loss=val_loss, seconds=seconds, tokens_per_s=tokens / seconds,
                params_m=sum(p.numel() for p in model.parameters()) / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="baseline", help=f"comma list from: {', '.join(VARIANTS)}; "
                        "combine ideas with '+', e.g. value_embeds+unet")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--data", default="data/fineweb10B", help="directory with fineweb_train_*.bin and fineweb_val_*.bin")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a Config field for all")
    parser.add_argument("--out", default="proxy_results.jsonl")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Config()
    for kv in args.set:
        key, _, value = kv.partition("=")
        kind = type(getattr(base, key))
        base = base.update(**{key: (value.lower() in ("1", "true")) if kind is bool else kind(value)})
    results = {}
    for name in args.variants.split(","):
        changes = {}
        for part in name.split("+"):
            changes.update(VARIANTS[part])
        for seed in range(args.seeds):
            cfg = base.update(**changes, seed=seed)
            print(f"== {name} seed {seed}: {changes}", flush=True)
            r = train(cfg, args.data, device)
            r.update(variant=name, seed=seed, config=dataclasses.asdict(cfg))
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            results.setdefault(name, []).append(r)
            print(f"   val {r['val_loss']:.4f}  {r['seconds']:.0f}s  {r['tokens_per_s'] / 1e3:.0f}k tok/s  "
                  f"{r['params_m']:.1f}M params", flush=True)
    base_val = np.mean([r["val_loss"] for r in results[next(iter(results))]])
    base_s = np.mean([r["seconds"] for r in results[next(iter(results))]])
    print(f"\n{'variant':>28} {'val':>8} {'+/-':>7} {'d val (mnat)':>12} {'time x':>7}")
    for name, rs in sorted(results.items(), key=lambda kv: np.mean([r["val_loss"] for r in kv[1]])):
        vals = [r["val_loss"] for r in rs]
        print(f"{name:>28} {np.mean(vals):8.4f} {np.std(vals, ddof=1) if len(vals) > 1 else float('nan'):7.4f} "
              f"{1000 * (np.mean(vals) - base_val):+12.1f} {np.mean([r['seconds'] for r in rs]) / base_s:7.2f}")
    print("(d val: vs the first variant listed; at the record's rate, 1 mnat ~ 164 ms of 8xH100 time, before any"
          " per-step cost the idea adds)")


if __name__ == "__main__":
    main()
