"""Steer lens: what pushing each neuron does to the model's next-token prediction, through every layer after it.

For each neuron and sign, adds ±dose x (median residual norm / ||W_down[:, n]||) to the neuron at every position of a
few short contexts, runs one forward pass, and compares the last position's log-probabilities with the unsteered run.
Keeps, per neuron and sign, the 12 tokens whose log-probability rises most (averaged over contexts) and the KL
divergence from the unsteered prediction.

--mode choice asks forced-choice investing questions instead ("... Should I buy or sell? Answer with one word: Buy or
Sell.") and keeps how far each push moves the answer: the change from the unsteered run in log P(first option) -
log P(second option), per axis. The axes are buy/sell, shares/cash and market up/down, with the risk-on option always
first, so a positive change means the push made the answer more risk-on. Each question is asked with its options in
both orders, so a push that only favours whichever option is named first cancels out. Random directions with the same
push size give the null.

    uv run python sweep/steer_lens.py --layers 0-7 --out stats/steer_lens
    uv run python sweep/steer_lens.py --mode choice --layers 3-31

Writes stats/steer_lens_L{l}.npz per layer: up_tok[N, 12], down_tok[N, 12] (token ids for the + push and - push),
up_kl[N], down_kl[N]. Choice mode writes stats/steer_choice_L{l}.npz: up_margin[N, A], down_margin[N, A] (change in
the answer margin per axis, nats), up_mass[N, A], down_mass[N, A] (probability left on the two options, so a push that
breaks the answer format shows up as low mass), up_kl[N], down_kl[N], the same with a ctrl_ prefix for the random
controls, and base_margin[A], base_mass[A], axes, options. Both write neurons[N], the neuron index of each row.
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]

# --mode choice: answer options per axis, risk-on first. Every option is a single token in each spelling used. The
# questions are ones the unsteered model doesn't answer near 100% one way: from a saturated start the log-odds change
# mostly measures the tail ("I'm 30, saving for retirement" was 1.00 shares and was dropped). The rest of the buy/sell
# mass goes to "Hold"; mass records it.
AXES = {"buy_sell": ("Buy", "Sell"), "shares_cash": ("Shares", "Cash"), "up_down": ("Up", "Down")}
CHOICE = [
    ("buy_sell", "Tesla stock is up 40% this year. Should I buy more or sell?"),
    ("buy_sell", "The S&P 500 just fell 10% in a month. Should I buy index funds or sell the ones I have?"),
    ("buy_sell", "Bitcoin is trading near its all-time high. Should I buy or sell?"),
    ("shares_cash", "I'm retiring next year. Should my savings be in shares or in cash?"),
    ("shares_cash", "Savings accounts pay 5% right now. Should I keep my money in shares or in cash?"),
    ("shares_cash", "I just got a $20,000 bonus. Should I put it in shares or keep it in cash?"),
    ("up_down", "Will the stock market be higher or lower a year from now?"),
    ("up_down", "Will Nvidia's stock price go up or down next month?"),
]
ASK = "{q} Answer with one word: {x} or {y}."


def parse_neurons(spec, n):
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return out


def option_ids(tok, word):
    """Token ids of the spellings of an answer that are a single token: Buy, " Buy", buy, " buy", BUY."""
    ids = set()
    for v in (word, " " + word, word.lower(), " " + word.lower(), word.upper()):
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1:
            ids.update(t)
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--mode", default="lens", choices=["lens", "choice"])
    ap.add_argument("--layers", default="all", help='"all", "0-7" or "3,4,9-12"')
    ap.add_argument("--neurons", default="all", help='"all", "0-99" or "1,5,9-12", the same for every layer')
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=None, help="default 96 in lens mode, 24 in choice mode, "
                    "which has 16 contexts per neuron against the lens's 4")
    ap.add_argument("--random-controls", type=int, default=256, help="choice mode: random directions per layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default stats/steer_lens, or stats/steer_choice in choice mode")
    a = ap.parse_args()
    choice = a.mode == "choice"
    out_stem = a.out or ("stats/steer_choice" if choice else "stats/steer_lens")
    B = a.neurons_per_batch or (24 if choice else 96)
    dev = a.device

    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev).eval()
    lm = model.model.language_model
    layers = parse_neurons(a.layers, len(lm.layers))
    chat =lambda c: tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=True)
    if choice:
        rows = [(ax, ASK.format(q=q, x=x, y=y)) for ax, q in CHOICE for x, y in (AXES[ax], AXES[ax][::-1])]
        texts = [t for _, t in rows]
        ctxs = [chat(t) for t in texts]
        ctx_axis = torch.tensor([list(AXES).index(ax) for ax, _ in rows], device=dev)
        opts = [(option_ids(tok, x), option_ids(tok, y)) for x, y in AXES.values()]
        for (x, y), (ix, iy) in zip(AXES.values(), opts):
            assert ix and iy and not set(ix) & set(iy), f"{x} / {y} need distinct single-token spellings"
    else:
        texts = CHAT + PLAIN
        ctxs = [chat(c) for c in CHAT] + [tok(p).input_ids for p in PLAIN]
    ctxs = [list(c["input_ids"] if hasattr(c, "keys") else c) for c in ctxs]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device=dev)
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device=dev)
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    # cols/vals push neurons (added to down_proj's input); vecs pushes residual directions (added to its output)
    push = {"layer": None, "cols": None, "vals": None, "vecs": None}
    resid = {}

    def make_hook(i):
        def hook(mod, inp):
            if push["layer"] != i or push["cols"] is None:
                return
            h = inp[0].clone()
            rows = torch.arange(h.shape[0], device=h.device)
            h[rows, :, push["cols"]] += push["vals"][:, None].to(h.dtype)
            return (h,)
        return hook

    def make_vec_hook(i):
        def hook(mod, inp, out):
            if push["layer"] != i or push["vecs"] is None:
                return
            return out + push["vecs"][:, None].to(out.dtype)
        return hook

    for i, layer in enumerate(lm.layers):
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
        layer.mlp.down_proj.register_forward_hook(make_vec_hook(i))
        layer.register_forward_hook(lambda m, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        h = lm(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(h).float(), -1)

    def pushed(l, b, cols=None, vals=None, vecs=None):
        """[b, C, V] last-position log-probs, rows neuron-major with the contexts repeated"""
        push.update(layer=l, cols=cols, vals=vals, vecs=vecs)
        with torch.inference_mode():
            lp = last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), posn.repeat(b, 1)).view(b, C, -1)
        push["layer"] = None
        return lp

    def answer(lp):
        """[b, C, V] -> margin, mass [b, A]: mean over an axis's contexts of log P(x) - log P(y) and of P(x) + P(y)"""
        margin, mass = [], []
        for k, (ix, iy) in enumerate(opts):
            sub = lp[:, ctx_axis == k]
            lx, ly = sub[..., ix].logsumexp(-1), sub[..., iy].logsumexp(-1)
            margin.append((lx - ly).mean(1))
            mass.append((lx.exp() + ly.exp()).mean(1))
        return torch.stack(margin, 1), torch.stack(mass, 1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)  # [C, V]
        # median residual norm leaving each layer over these contexts' real tokens
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in resid.items()}
    if choice:
        base_margin, base_mass = (x[0] for x in answer(base[None]))
        print("unsteered: " + ", ".join(f"{ax} margin {m:+.2f} mass {p:.2f}" for ax, m, p in
                                       zip(AXES, base_margin.tolist(), base_mass.tolist())), flush=True)

    def record(out, key, lp):
        out[f"{key}_kl"].append((lp.exp() * (lp - base[None])).sum(-1).mean(1).cpu())  # KL(steered || base)
        if choice:
            margin, mass = answer(lp)
            out[f"{key}_margin"].append((margin - base_margin).cpu())
            out[f"{key}_mass"].append(mass.cpu())
        else:
            out[f"{key}_tok"].append((lp - base[None]).mean(1).topk(a.k).indices.int().cpu())

    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight  # [hidden, inter]
        unit = norms[l] / W.float().norm(dim=0)  # [N]
        neurons = torch.tensor(parse_neurons(a.neurons, W.shape[1]), device=dev)
        out = defaultdict(list)
        t0 = time.time()
        for s in range(0, len(neurons), B):
            n = neurons[s : s + B]
            cols = n.repeat_interleave(C)
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                record(out, key, pushed(l, len(n), cols=cols, vals=sign * a.dose * unit[cols]))
        if choice and a.random_controls:
            # random directions in the residual stream, the same push size as every neuron's: dose x median norm
            g = torch.Generator().manual_seed(a.seed + l)
            dirs = torch.randn(a.random_controls, W.shape[0], generator=g).to(dev)
            dirs *= a.dose * norms[l] / dirs.norm(dim=1, keepdim=True)
            for s in range(0, len(dirs), B):
                d = dirs[s : s + B].repeat_interleave(C, 0)
                for sign, key in ((1.0, "ctrl_up"), (-1.0, "ctrl_down")):
                    record(out, key, pushed(l, len(d) // C, vecs=sign * d))
        arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
        if choice:
            arrs.update(base_margin=base_margin.cpu().numpy(), base_mass=base_mass.cpu().numpy(),
                        axes=np.array(list(AXES)), options=np.array(list(AXES.values())))
        np.savez(f"{out_stem}_L{l}.npz", **arrs, neurons=neurons.cpu().numpy(), dose=a.dose, contexts=np.array(texts))
        msg = f"L{l}: {len(neurons)} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}"
        if choice:
            top = np.abs(np.concatenate([arrs["up_margin"], arrs["down_margin"]])).max(0)
            msg += ", max |Δmargin| " + " ".join(f"{ax} {m:.2f}" for ax, m in zip(AXES, top))
            if a.random_controls:
                ctrl = np.abs(np.concatenate([arrs["ctrl_up_margin"], arrs["ctrl_down_margin"]])).max(0)
                msg += " (controls " + " ".join(f"{m:.2f}" for m in ctrl) + ")"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
