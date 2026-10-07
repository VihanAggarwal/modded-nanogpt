"""CPU tests for the Canon layers behind CANON_LAYERS (track_1_short/model/gpt.py, track_1_short/training.py).

Run from the repo root: python -m pytest tools/tests -q
The op-level tests import model/gpt.py in this process, with its GPU-only imports stubbed (gpt_cpu_stubs.py). The
model reads its CANON_LAYERS* flags at import, so each model-level configuration builds the full model in its own
subprocess (this file run as a script), a few at a time; the torch.compile checks run in one too. Needs the GPT-2
tokenizer files in tiktoken's cache (TIKTOKEN_CACHE_DIR): the model's import builds a token table. Not covered (needs
a GPU): the fp8 training forward and backward, CUDA graph capture, and the per-step cost.
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import gpt_cpu_stubs  # noqa: E402

FLAG_PREFIXES = ("CANON_LAYERS", "CPLM")
T = 256  # tokens in the model-level forward, documents starting at BOS_POSITIONS
BOS_POSITIONS = [0, 37, 38, 120, 200]
DIM = 768

# Model-level configurations: their flags, beyond a clean environment.
CONFIGS = {
    "off": {},
    "AC": {"CANON_LAYERS": "AC"},  # the defaults: post, K = 4, no BOS mask
    "AC_pre_bos_k3": {"CANON_LAYERS": "AC", "CANON_LAYERS_NORM": "pre", "CANON_LAYERS_BOS_MASK": "1",
                      "CANON_LAYERS_K": "3"},
    "A_renorm": {"CANON_LAYERS": "A", "CANON_LAYERS_NORM": "renorm"},
    "C_bos": {"CANON_LAYERS": "C", "CANON_LAYERS_BOS_MASK": "1"},
    # CPLM (the record's default) adds replicated labels of its own. Its copy head has no CPU stand-ins, so this one
    # checks only the optimizer tables: the forward and gradient checks above run without CPLM.
    "AC_cplm": {"CANON_LAYERS": "AC", "CPLM": "1", "CPLM_QK_NORM": "1", "CPLM_EXT_GATE": "1"},
}
SITES = {"A": 6, "C": 10, "AC": 16}


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- model level (one subprocess per configuration)

def probe() -> dict:
    """Build the model under this process's flags; its init, optimizer tables, and the validation forward and
    backward at zero taps and at random taps."""
    gpt_cpu_stubs.install()
    import track_1_short.model.gpt as gpt
    import track_1_short.training as training
    from track_1_short.config import LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT
    from track_1_short.ngram_table import NGRAM_DIM
    from track_1_short.schedule import TrainingSchedule

    torch.manual_seed(0)
    model = gpt.GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=DIM, max_seq_len=T,
                    ngram_dim=NGRAM_DIM, world_size=8, device=torch.device("cpu"))
    out = {"rng": digest(torch.get_rng_state())}
    model.cast_matrix_weights_bf16()
    out["params"] = {name: digest(p) for name, p in model.named_parameters()}
    taps = getattr(model, "canon_layer_taps", None)
    out["taps"] = None if taps is None else [list(taps.shape), str(taps.dtype), taps.label, taps.abs().max().item()]

    class LabelCheck:
        """AnvilAndAdam.__init__'s label checks (optim/anvil.py), without its comms and buffers."""
        def __init__(self, named_params, param_table, scatter_order, work_order, **kwargs):
            by_label = {}
            for _, param in named_params:
                assert param.label in param_table and param.label not in by_label, param.label
                by_label[param.label] = param
            assert set(scatter_order) == set(by_label) and set(work_order) == set(by_label)
            assert len(scatter_order) == len(set(scatter_order)) and len(work_order) == len(set(work_order))

        def reset(self):
            pass

    training.AnvilAndAdam = LabelCheck
    schedule = TrainingSchedule(TRAINING_STAGES, 978, 20, device=torch.device("cpu"), cooldown_frac=LR_COOLDOWN_FRAC,
                                split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT)
    manager = training.TrainingManager(model, schedule, bank_update=None)
    out["table"] = manager.param_table.get("canon_layer_taps")
    out["work_order"] = manager.work_order
    if model.cplm:
        return out

    gpt_cpu_stubs.use_torch_kernels()
    with torch.no_grad():  # c_proj and the head start at zero: give every path signal
        model.mlp_bank[:, 1].normal_(0, 0.02)
        model.lm_head.weight.normal_(0, 0.02)
        model.ngram_cache.normal_(0, 1)
    model.eval()
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 50000, (T + 1,), generator=gen)
    ids[BOS_POSITIONS] = 50256
    seqlens = torch.tensor(BOS_POSITIONS + [T] * 11, dtype=torch.int32)
    slots = torch.randint(0, 2 * T, (2 * T,), generator=gen).int()
    cfg = gpt.ForwardScheduleConfig(mtp_weights=torch.tensor([1.0]), prefix_weight=torch.tensor([0.0]), ws_short=384,
                                    ws_long=896, train_max_seq_len=T)

    def loss_bits() -> int:
        loss = model(ids[:-1].int(), ids[1:].long(), seqlens, slots, cfg).mean()
        assert torch.isfinite(loss)
        loss.backward()
        return loss.detach().float().view(torch.int32).item()

    out["loss_bits"] = loss_bits()
    out["grads"] = {name: digest(p.grad) for name, p in model.named_parameters()
                    if p.grad is not None and name != "canon_layer_taps"}
    if taps is not None:
        out["sites_with_grad"] = int((taps.grad.abs().sum((1, 2)) > 0).sum())
        with torch.no_grad():
            taps.normal_(0, 0.2)
        out["loss_bits_random_taps"] = loss_bits()
        if gpt.CANON_LAYERS_BOS_MASK:  # the forward reads the flag on every call
            gpt.CANON_LAYERS_BOS_MASK = False
            out["loss_bits_random_taps_unmasked"] = loss_bits()
    return out


class ScriptResult:
    """This file run as a script (`mode`: probe or compile) in a clean environment plus `flags`; [key] -> its output's
    value, or the script's error."""
    def __init__(self, mode: str, **flags):
        env = {k: v for k, v in os.environ.items() if not k.startswith(FLAG_PREFIXES)}
        # One thread each: several run at once, and oversubscribed OpenMP threads slow them down several-fold.
        env.update(flags, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, __file__, mode], env=env, capture_output=True, text=True, timeout=900)
        self.label, self.stderr = f"{mode} {flags}", proc.stderr[-4000:]
        self.result = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None

    def __getitem__(self, key: str):
        assert self.result is not None, f"{self.label} failed:\n{self.stderr}"
        return self.result[key]


@pytest.fixture(scope="module")
def probes():
    """Every configuration's probe, three at a time (each peaks at ~2.5 GB)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(CONFIGS, pool.map(lambda name: ScriptResult("probe", **CONFIGS[name]), CONFIGS)))


@pytest.mark.parametrize("name", [n for n in CONFIGS if n not in ("off", "AC_cplm")])
def test_init_draws_no_rng_and_leaves_other_params(probes, name):
    off, on = probes["off"], probes[name]
    assert on["rng"] == off["rng"], "the taps' init consumed RNG"
    assert set(on["params"]) - set(off["params"]) == {"canon_layer_taps"}
    assert all(on["params"][n] == off["params"][n] for n in off["params"])
    k = int(CONFIGS[name].get("CANON_LAYERS_K", 4))
    assert on["taps"] == [[SITES[CONFIGS[name]["CANON_LAYERS"]], k, DIM], "torch.float32", "canon_layer_taps", 0.0]


@pytest.mark.parametrize("name", ["off", "AC", "AC_cplm"])
def test_optimizer_tables_cover_taps(probes, name):
    out = probes[name]  # the probe ran AnvilAndAdam's label checks on the tables
    if name == "off":
        assert out["table"] is None and "canon_layer_taps" not in out["work_order"]
        return
    assert out["table"] == {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 1.0,
                            "wd_mul": 0.0}
    assert out["work_order"].index("canon_layer_taps") < out["work_order"].index("mlp_bank")


@pytest.mark.parametrize("name", [n for n in CONFIGS if n not in ("off", "AC_cplm")])
def test_zero_taps_leave_forward_unchanged_and_every_site_learns(probes, name):
    off, on = probes["off"], probes[name]
    assert on["loss_bits"] == off["loss_bits"]  # bitwise
    if CONFIGS[name].get("CANON_LAYERS_NORM") != "renorm":  # renorm's backward rounds differently at zero taps
        assert on["grads"] == off["grads"]
    assert on["sites_with_grad"] == SITES[CONFIGS[name]["CANON_LAYERS"]]
    assert on["loss_bits_random_taps"] != off["loss_bits"]
    if CONFIGS[name].get("CANON_LAYERS_BOS_MASK"):  # the mask reaches the forward
        assert on["loss_bits_random_taps_unmasked"] != on["loss_bits_random_taps"]


# ---------------------------------------------------------------- op level (in process)

@pytest.fixture(scope="module")
def gpt():
    """model/gpt.py with its flags unset. Afterwards the stubs and the track_1_short modules it brought are dropped
    from sys.modules, and the environment is restored."""
    modules, environ = set(sys.modules), dict(os.environ)
    for key in [k for k in os.environ if k.startswith(FLAG_PREFIXES)]:
        del os.environ[key]
    gpt_cpu_stubs.install()
    import track_1_short.model.gpt as module
    yield module
    for name in set(sys.modules) - modules:
        if name.startswith("track_1_short.") or name in gpt_cpu_stubs.STUBBED:
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                vars(sys.modules[parent]).pop(child, None)
    os.environ.clear()
    os.environ.update(environ)


def documents(lengths: list[int], generator: torch.Generator) -> torch.Tensor:
    return torch.cat([torch.cat([torch.tensor([50256]), torch.randint(0, 50000, (n - 1,), generator=generator)])
                      for n in lengths])


def conv1d_reference(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The Canon layer as h + nn.Conv1d(D, D, K, groups=D), cut to causal (the round-1/2 proxy's form)."""
    K = w.size(0)
    weight = w.flip(0).T[:, None, :]  # nn.Conv1d weight [D, 1, K]: weight[:, 0, K - 1 - j] = w[j]
    return h + F.conv1d(F.pad(h.transpose(1, 2), (K - 1, 0)), weight, groups=h.size(-1)).transpose(1, 2)


def test_conv_matches_conv1d(gpt):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1, 40, 16, generator=g, dtype=torch.float64)
    w = torch.randn(4, 16, generator=g, dtype=torch.float64)
    ref = conv1d_reference(h, w)
    assert torch.allclose(gpt.canon_layer_conv(h.float(), w.float()).double(), ref, atol=1e-5, rtol=1e-5)
    renormed = ref * (h.square().mean(-1, keepdim=True) / ref.square().mean(-1, keepdim=True)).sqrt()
    assert torch.allclose(gpt.canon_layer_conv(h.float(), w.float(), renorm=True).double(), renormed, atol=1e-5)


def test_zero_taps_return_input_bitwise(gpt):
    g = torch.Generator().manual_seed(1)
    h = (torch.randn(1, 64, DIM, generator=g) * 30).bfloat16()
    keep = gpt.canon_layer_bos_keep(documents([5, 1, 2, 30, 26], g), 4)
    for args in [(), (keep,), (None, True), (keep, True)]:
        assert torch.equal(gpt.canon_layer_conv(h, torch.zeros(4, DIM), *args).view(torch.int16), h.view(torch.int16))


def test_causal(gpt):
    g = torch.Generator().manual_seed(2)
    h, w = torch.randn(1, 50, 8, generator=g), torch.randn(4, 8, generator=g)
    later = h.clone()
    later[:, 30:] += 5.0
    for renorm in (False, True):
        y, y_later = gpt.canon_layer_conv(h, w, renorm=renorm), gpt.canon_layer_conv(later, w, renorm=renorm)
        assert torch.equal(y[:, :30], y_later[:, :30]) and not torch.equal(y[:, 30:], y_later[:, 30:])


@pytest.mark.parametrize("placement", ["post", "renorm", "pre"])
def test_site_computes_its_placement(gpt, monkeypatch, placement):
    g = torch.Generator().manual_seed(7)
    x, w = 4 * torch.randn(1, 24, 16, generator=g), 0.3 * torch.randn(4, 16, generator=g)
    keep = gpt.canon_layer_bos_keep(documents([10, 14], g), 4)
    want = {"post": gpt.canon_layer_conv(gpt.norm(x), w, keep),
            "renorm": gpt.canon_layer_conv(gpt.norm(x), w, keep, renorm=True),
            "pre": gpt.norm(gpt.canon_layer_conv(x, w, keep))}
    monkeypatch.setattr(gpt, "CANON_LAYERS_NORM", placement)
    got = gpt.canon_layer_site(x, gpt.norm(x), w, keep)
    assert torch.equal(got, want[placement])
    assert not any(torch.allclose(got, v) for k, v in want.items() if k != placement)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("first", ["bos", "mid_document"])  # training chunks start at a BOS, validation chunks not
def test_bos_mask_equals_per_document_conv(gpt, k, first):
    g = torch.Generator().manual_seed(3)
    lengths = [7, 1, 2, 12, 3, 9]  # documents of length 1 and 2: the mask's running product
    ids = documents(lengths, g)
    if first == "mid_document":
        ids, lengths = ids[1:], [lengths[0] - 1] + lengths[1:]
    h, w = torch.randn(1, ids.numel(), 8, generator=g), torch.randn(k, 8, generator=g)
    got = gpt.canon_layer_conv(h, w, gpt.canon_layer_bos_keep(ids, k))
    ref = torch.cat([conv1d_reference(doc, w) for doc in h.split(lengths, dim=1)], dim=1)
    assert torch.allclose(got, ref, atol=1e-5)


@pytest.mark.filterwarnings("ignore:Input #")  # fp32 inputs: the function computes in fp32
@pytest.mark.parametrize("renorm", [False, True])
def test_gradcheck(gpt, renorm):
    g = torch.Generator().manual_seed(4)
    h = torch.randn(1, 12, 5, generator=g).requires_grad_()
    w = (0.5 * torch.randn(4, 5, generator=g)).requires_grad_()
    keep = gpt.canon_layer_bos_keep(documents([5, 2, 5], g), 4)
    for args in [(None, renorm), (keep, renorm)]:
        assert torch.autograd.gradcheck(lambda h, w: gpt.canon_layer_conv(h, w, *args), (h, w), eps=1e-2, atol=2e-3,
                                        rtol=2e-3)


def test_renorm_keeps_the_fp8_input_bound(gpt):
    """FP8_MLP_X_SCALE saturates beyond |28|: post exceeds sqrt(768) on a near one-hot row, renorm never does."""
    g = torch.Generator().manual_seed(5)
    h = F.rms_norm(torch.randn(1, 32, DIM, generator=g), (DIM,))
    h[0, 10] = 0
    h[0, 10, 3] = DIM ** 0.5  # one-hot row, normed: its entry is sqrt(768) = 27.7
    w = 0.5 * torch.randn(4, DIM, generator=g)
    w[0] = 0.5
    assert gpt.canon_layer_conv(h, w).abs().max() > 28
    assert gpt.canon_layer_conv(h, w, renorm=True).abs().max() <= DIM ** 0.5 * (1 + 1e-5)


def compile_sites() -> dict:
    """Each placement's site, with its BOS mask and then the fp8 row quantize of an MLP input (perf/kernels/mlp.py),
    under torch.compile against eager. In a subprocess: inductor leaves threads running, and the loader test
    (test_loader_release_and_val_reads.py) joins every thread."""
    gpt_cpu_stubs.install()
    import track_1_short.model.gpt as gpt
    from torch._inductor.utils import fresh_inductor_cache, run_and_get_code

    g = torch.Generator().manual_seed(6)
    x, w = (4 * torch.randn(1, 64, 32, generator=g)).bfloat16(), 0.3 * torch.randn(4, 32, generator=g)
    ids = documents([20, 1, 43], g)
    probe_dir = torch.randn(1, 64, 32, generator=g)  # a loss the norm does not make constant

    def site(x, w, ids):
        h = gpt.canon_layer_site(x, gpt.norm(x), w, gpt.canon_layer_bos_keep(ids, 4))
        return h, torch.clamp(h.detach().float() * 16.0, -448, 448).to(torch.float8_e4m3fn)

    def step(fn, saved):
        xg, wg = x.clone().requires_grad_(), w.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            h, h_f8 = fn(xg, wg, ids)
        (h.float() * probe_dir).sum().backward()
        return h, h_f8, xg.grad, wg.grad

    out = {}
    for placement in ("post", "renorm", "pre"):
        gpt.CANON_LAYERS_NORM = placement
        torch._dynamo.reset()
        saved = []  # what the compiled forward keeps for the backward
        with fresh_inductor_cache():
            compiled, code = run_and_get_code(step, torch.compile(site, dynamic=False, fullgraph=True), saved)
        eager = step(site, [])
        out[placement] = {
            "convolution": any("convolution" in source for source in code),
            "saved_fp32_rows": sum(t.dtype == torch.float32 and t.shape == x.shape for t in saved),
            # h, its fp8 copy, the x and w gradients: max |compiled - eager| over max |eager|
            "errors": [((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
                       for a, b in zip(compiled, eager)],
        }
    return out


@pytest.fixture(scope="module")
def compiled():
    return ScriptResult("compile")


@pytest.mark.parametrize("placement", ["post", "renorm", "pre"])
def test_compiled_site_matches_eager_and_keeps_no_shifted_copies(compiled, placement):
    """CPU inductor: no extern conv; the backward recomputes the shifted rows rather than keeping fp32 [T, D] copies
    of them; compiled matches eager (which rounds to bf16 between ops; fp8 steps are 1/8). Says nothing about how
    inductor fuses the site on a GPU: a CPU kernel can hold several passes over the rows."""
    out = compiled[placement]
    assert not out["convolution"] and out["saved_fp32_rows"] == 0
    assert all(err <= tol for err, tol in zip(out["errors"], (1e-2, 0.13, 1e-2, 1e-2))), out["errors"]


if __name__ == "__main__":
    print(json.dumps({"probe": probe, "compile": compile_sites}[sys.argv[1]]()))
