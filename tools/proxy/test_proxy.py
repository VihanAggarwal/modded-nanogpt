"""CPU checks of the proxy's round-3 pieces: the key offset, Canon (its forms, on keys, on values), causality,
init pairing across variants and the optimizer groups.

    python -m pytest tools/proxy/test_proxy.py -q
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_gpt as P  # noqa: E402

# head_dim 32: dims 8..15 and 24..31 are stationary. Layers 0 and 2 have the key offset.
SMALL = P.Config(n_layer=3, d_model=64, head_dim=32, seq_len=24, key_offset_layers=(0, 2))
REC = "polar_express+normuon+cautious_wd+value_embeds+x0_mix+attn_gate+smear+key_offset"
CANON_FORMS = ["canon", "canon_a", "canon_c", "canon_k2", "canon_k3", "canon_prenorm", "canon_renorm", "canon_from1"]
NEW = CANON_FORMS + ["key_offset", "canon_bk", "v_shift", "canon+canon_bk"]


def cfg_of(name, seed=0):
    changes = {}
    for part in name.replace("REC", REC).split("+"):
        changes.update(P.VARIANTS[part])
    return SMALL.update(**changes, seed=seed)


def built(name, seed=0):
    torch.manual_seed(seed)
    return P.GPT(cfg_of(name, seed))


def tokens(seed=0, T=SMALL.seq_len):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 50257, (2, T + 1), generator=g)


def randomize_zeros(model, seed=1):
    """Give every all-zero weight (projections, head, gates, taps) random values, so every path matters."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().max() == 0:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))


def test_rotary_stationary_dims():
    rot = P.Rotary(128, 16)
    assert rot.stationary.nonzero().flatten().tolist() == list(range(32, 64)) + list(range(96, 128))
    x = torch.randn(2, 16, 3, 128)
    assert torch.equal(rot(x)[..., rot.stationary], x[..., rot.stationary])  # the rotary leaves them alone


def test_key_offset_matches_reference():
    k, stat = torch.randn(2, 9, 3, 32), P.Rotary(32, 9).stationary
    ref = k.clone()
    for t in range(1, k.size(1)):
        ref[:, t, :, stat] = k[:, t - 1, :, stat]
    assert torch.equal(P.key_offset(k, stat), ref)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("edge", [False, True])
def test_canon_equals_conv1d(k, edge):
    h, w = torch.randn(2, 11, 6, dtype=torch.float64), torch.randn(k, 6, dtype=torch.float64)
    weight = w.T.flip(-1)[:, None, :]  # Conv1d [d, 1, k]: weight[:, 0, k-1-j] = w[j]
    padded = F.pad(h.transpose(1, 2), (k - 1, 0), mode="replicate" if edge else "constant")
    ref = h + F.conv1d(padded, weight, groups=6).transpose(1, 2)
    torch.testing.assert_close(P.canon(w, h, edge=edge), ref, rtol=0, atol=1e-12)


def test_canon_bk_init_is_the_key_offset():
    block = P.Block(cfg_of("key_offset+canon_bk"), layer=0)
    assert not block.key_offset  # the learned taps replace the hard offset
    k = P.norm(torch.randn(2, 9, block.n_head, 32))
    out = P.canon(block.canon_bk, k.flatten(2), edge=True).view_as(k)
    assert torch.equal(out, P.key_offset(k, block.rotary.stationary))
    assert P.Block(cfg_of("canon_bk"), layer=0).canon_bk.abs().max() == 0  # elsewhere: zero-init
    assert P.Block(cfg_of("key_offset+canon_bk"), layer=1).canon_bk.abs().max() == 0  # not a key-offset layer


@pytest.mark.parametrize("name,ref", [("canon", lambda w, x: P.canon(w, P.norm(x))),
                                      ("canon_prenorm", lambda w, x: P.norm(P.canon(w, x))),
                                      ("canon_renorm", lambda w, x: P.norm(P.canon(w, P.norm(x))))])
def test_canon_forms(name, ref):
    block = P.Block(cfg_of(name), layer=0)
    w, x = torch.randn(4, SMALL.d_model), torch.randn(2, 7, SMALL.d_model)
    assert torch.equal(block.sublayer_input(x, w), ref(w, x))


def test_canon_sites_and_layers():
    sites = lambda name: [(b.canon_a is not None, b.canon_c is not None) for b in built(name).blocks]
    assert sites("canon") == [(True, True)] * 3 and sites("canon_a") == [(True, False)] * 3
    assert sites("canon_c") == [(False, True)] * 3 and sites("canon_from1") == [(False, False)] + [(True, True)] * 2
    assert built("canon_k3").blocks[0].canon_a.shape == (3, SMALL.d_model)


@pytest.mark.parametrize("name", ["REC+" + n for n in NEW if n != "key_offset"] + ["canon_bk", "v_shift", "canon"])
def test_zero_init_is_identity(name):
    """Every new flag starts as its base: same loss given the same weights (canon_bk on a key-offset base starts as
    the hard offset). The base model's zero-init weights are randomized first, so every path is exercised."""
    base_name = "REC" if name.startswith("REC+") else "baseline"
    base, model = built(base_name), built(name)
    randomize_zeros(base)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("canon" in n for n in missing)
    x = tokens()
    want, got = base(x[:, :-1], x[:, 1:]), model(x[:, :-1], x[:, 1:])
    if "renorm" in name:  # norm(norm(x)) is norm(x) to rounding
        torch.testing.assert_close(got, want, rtol=1e-6, atol=0)
    else:
        assert torch.equal(got, want)
    got.backward()
    taps = [p for n, p in model.named_parameters() if "canon" in n]
    assert taps and all(p.grad is not None and p.grad.abs().max() > 0 for p in taps)


CAUSAL = ["REC+canon+canon_bk+v_shift", "REC+canon_prenorm", "REC+canon_renorm", "REC+canon_k2", "key_offset"]


@pytest.mark.parametrize("name", CAUSAL)
def test_block_is_causal(name):
    """d out[t] / d in[t' > t] == 0 through every op of a block, with every new weight non-zero."""
    block = P.Block(cfg_of(name), layer=0)
    randomize_zeros(block)
    with torch.no_grad():
        for n, p in block.named_parameters():
            if "canon" in n:
                p.add_(0.1 * torch.randn_like(p))
    T, d = 12, SMALL.d_model
    x, x0, ve = (torch.randn(2, T, d, requires_grad=True) for _ in range(3))
    out = block(x, x0, ve)
    for t in (0, 4, T - 2):
        grads = torch.autograd.grad(out[:, t].sum(), (x, x0, ve), retain_graph=True, allow_unused=True)
        for g in grads:
            assert g is None or g[:, t + 1:].abs().max() == 0
        assert grads[0][:, :t + 1].abs().max() > 0


def test_model_is_causal():
    """Changing token t' changes no logit before t' (embeddings, smear, n-gram hashes, every block)."""
    model = built("REC+canon+canon_bk+v_shift+trigram_hash")
    randomize_zeros(model)
    logits = []
    model.head.register_forward_hook(lambda m, i, o: logits.append(o))
    x = tokens()
    y = x.clone()
    y[:, 10] = (y[:, 10] + 1) % 50257
    model(x[:, :-1], x[:, 1:]), model(y[:, :-1], y[:, 1:])
    assert torch.equal(logits[0][:, :10], logits[1][:, :10])
    assert not torch.equal(logits[0][:, 10:], logits[1][:, 10:])


@pytest.mark.parametrize("base_name", ["baseline", "REC"])
def test_init_is_paired_across_variants(base_name):
    """Under one seed every parameter a variant shares with its base starts the same (bitwise)."""
    for seed in (0, 3):
        base = dict(built(base_name, seed).named_parameters())
        for name in NEW + ["trigram_hash", "trigram_hash+canon", "rec_rep"]:
            model = dict(built(f"{base_name}+{name}", seed).named_parameters())
            for n, p in base.items():
                assert torch.equal(model[n], p), (name, n)
            assert all("canon" in n or "gram" in n for n in set(model) - set(base)), name


def test_taps_on_adam_not_muon():
    cfg = cfg_of("REC+canon+canon_bk+v_shift+trigram_hash")
    torch.manual_seed(0)
    model = P.GPT(cfg)
    adam, muon = P.build_optimizers(model, cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    muon_names = [names[id(p)] for g in muon.param_groups for p in g["params"]]
    assert muon_names and all(n.split(".")[-2] in ("qkv", "proj", "fc", "out") for n in muon_names)
    taps = [n for n in names.values() if "canon" in n]
    assert len(taps) == 4 * cfg.n_layer  # canon_a, canon_c, canon_bk, canon_bv on every layer
    scalar_group = [g for g in adam.param_groups if g["lr"] == cfg.lr_scalar]
    assert len(scalar_group) == 1 and scalar_group[0]["weight_decay"] == 0
    assert set(taps) <= {names[id(p)] for p in scalar_group[0]["params"]}
    every = [id(p) for opt in (adam, muon) for g in opt.param_groups for p in g["params"]]
    assert sorted(every) == sorted(names)  # each parameter in exactly one group


@pytest.mark.parametrize("name", ["REC+canon+canon_bk+v_shift+trigram_hash", "REC+canon_prenorm+canon_k3",
                                  "REC+canon_renorm+canon_from1"])
def test_traces_as_one_graph(name):
    torch._dynamo.reset()
    model = built(name)
    x = tokens()
    torch.compile(model, fullgraph=True, backend="eager")(x[:, :-1], x[:, 1:]).backward()


@pytest.mark.parametrize("change", [{"canon_sites": "a"}, {"canon_norm": "prenorm"}])
def test_unknown_canon_form_raises(change):
    with pytest.raises(AssertionError):
        P.GPT(SMALL.update(canon=True, **change))
