"""Investing candidates: neuron poles whose steer lens raises investing words.

Reads the steer lens index that ships with the explorer (explorer/cache/steer_index.npz: for every neuron and sign,
the 12 tokens whose log-probability a ±0.8 push raises most) and keeps every pole where at least --min-hits of those
12 tokens are finance words, matched case-insensitively with spaces stripped. Writes sweep/invest_candidates.json:
every pole with its hits, KL and tokens, strongest first, and per layer the neurons to sweep (a neuron is listed once
even when both of its poles qualify). sweep/run_invest.sh reads by_layer.

    uv run python sweep/invest_candidates.py
"""
import argparse
import json
from collections import defaultdict

import numpy as np

WORDS = (
    "invest investment investing investor investors stock stocks shares dividend dividends portfolio equity bond bonds "
    "market markets trading trader fund funds ETF hedge returns wealth profit capital asset assets financial finance "
    "bull bear crypto bitcoin interest retirement IPO valuation earnings revenue risk inflation savings broker Wall "
    "shareholder shareholders money cash price prices"
).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="explorer/cache/steer_index.npz")
    ap.add_argument("--vocab", default="explorer/cache/vocab.json")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--out", default="sweep/invest_candidates.json")
    a = ap.parse_args()

    vocab = json.load(open(a.vocab))
    words = {w.lower() for w in WORDS}
    finance = np.array([t.strip().lower() in words for t in vocab])
    st = np.load(a.index)
    poles = []
    for side, sign in (("up", "+"), ("down", "-")):
        toks, kl = st[side], st[f"{side}_kl"]
        hits = finance[toks].sum(-1)  # [layers, neurons]
        for l, n in zip(*np.nonzero(hits >= a.min_hits)):
            poles.append({"layer": int(l), "neuron": int(n), "sign": sign, "hits": int(hits[l, n]),
                          "kl": round(float(kl[l, n]), 3), "tokens": [vocab[t] for t in toks[l, n]]})
    poles.sort(key=lambda p: (-p["hits"], -p["kl"]))
    by_layer = defaultdict(set)
    for p in poles:
        by_layer[p["layer"]].add(p["neuron"])
    by_layer = {l: sorted(by_layer[l]) for l in sorted(by_layer)}

    head = {"source": a.index, "dose": float(st["dose"]), "min_hits": a.min_hits, "words": WORDS,
            "n_poles": len(poles), "n_neurons": sum(map(len, by_layer.values()))}
    # top level indented, one pole per line, so the file reads and diffs line by line
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines += [' "by_layer": {', ",\n".join(f'  "{l}": {json.dumps(ns)}' for l, ns in by_layer.items()), " },"]
    lines += [' "poles": [', ",\n".join("  " + json.dumps(p, ensure_ascii=False) for p in poles), " ]", "}"]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    json.load(open(a.out))
    print(f"{len(poles)} poles, {head['n_neurons']} neurons over {len(by_layer)} layers -> {a.out}")


if __name__ == "__main__":
    main()
