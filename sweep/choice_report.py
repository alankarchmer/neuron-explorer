"""Which neuron poles move an investing answer further than random directions do, from the forced-choice lens.

Reads stats/steer_choice_L{l}.npz (sweep/steer_lens.py --mode choice). Per layer and axis, a pole's shift is its change
in risk-on log-odds, and the null is the random controls' shifts with both signs pooled. A pole counts when its shift
falls outside the controls' central 99% and it leaves at least --min-mass of the probability on the two options; the
same mass filter applies to the controls. By chance about 1% of poles count per axis. Candidates are the investing
poles in sweep/invest_candidates.json; tokens are each pole's steer lens.

    uv run python sweep/choice_report.py --layers 7,24
"""
import argparse
import json

import numpy as np


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="stats/steer_choice")
    ap.add_argument("--layers", required=True, help='"7,24" or "3-31"')
    ap.add_argument("--min-mass", type=float, default=0.5)
    ap.add_argument("--min-controls", type=int, default=200, help="control pushes needed for a central 99%%")
    ap.add_argument("--show", type=int, default=8, help="poles listed per layer and axis")
    ap.add_argument("--candidates", default="sweep/invest_candidates.json")
    ap.add_argument("--lens", default="explorer/cache/steer_index.npz")
    ap.add_argument("--vocab", default="explorer/cache/vocab.json")
    a = ap.parse_args()

    cand = {(p["layer"], p["neuron"], p["sign"]) for p in json.load(open(a.candidates))["poles"]}
    lens, vocab = np.load(a.lens), json.load(open(a.vocab))
    for l in parse_layers(a.layers):
        z = np.load(f"{a.stem}_L{l}.npz")
        axes, neurons = [str(x) for x in z["axes"]], z["neurons"]
        poles = [(int(n), "+") for n in neurons] + [(int(n), "-") for n in neurons]
        shift, mass = (np.concatenate([z[f"up_{k}"], z[f"down_{k}"]]) for k in ("margin", "mass"))
        kl = np.concatenate([z["up_kl"], z["down_kl"]])
        cshift, cmass = (np.concatenate([z[f"ctrl_up_{k}"], z[f"ctrl_down_{k}"]]) for k in ("margin", "mass"))
        is_cand = np.array([(l, n, s) in cand for n, s in poles])
        print(f"L{l}: {len(poles)} poles, {len(cshift)} control pushes; unsteered "
              + ", ".join(f"{ax} {m:+.2f} (mass {p:.2f})" for ax, m, p in zip(axes, z["base_margin"], z["base_mass"])))
        for k, ax in enumerate(axes):
            ok, cok = mass[:, k] >= a.min_mass, cmass[:, k] >= a.min_mass
            if cok.sum() < a.min_controls:
                print(f"  {ax}: only {cok.sum()} control pushes keep mass >= {a.min_mass}, no null")
                continue
            lo, med, hi = np.percentile(cshift[cok, k], [0.5, 50, 99.5])
            out = ok & ((shift[:, k] < lo) | (shift[:, k] > hi))
            print(f"  {ax}: controls median {med:+.2f}, central 99% [{lo:+.2f}, {hi:+.2f}]; outside: {out.sum()} of "
                  f"{ok.sum()} poles ({out.sum() / max(ok.sum(), 1):.1%}, ~1% by chance), candidates "
                  f"{(out & is_cand).sum()} of {(ok & is_cand).sum()}; {(~ok).sum()} poles below mass")
            for i in sorted(np.nonzero(out)[0], key=lambda i: -abs(shift[i, k] - med))[: a.show]:
                n, s = poles[i]
                toks = " ".join(repr(vocab[t]) for t in lens["up" if s == "+" else "down"][l, n][:6])
                print(f"    N{n}{s} {shift[i, k]:+.2f}  mass {mass[i, k]:.2f}  KL {kl[i]:.2f}"
                      f"{'  candidate' if is_cand[i] else ''}  {toks}")


if __name__ == "__main__":
    main()
