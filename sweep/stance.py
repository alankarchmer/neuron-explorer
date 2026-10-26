"""Stance of the advice in invest_rel rollouts: does a push make the model refuse, disclaim, hedge or recommend, more or
less often than random directions of the same size do?

Each rollout gets four regex flags:
    refuse      the opening (first 300 characters) declines: "I cannot / can't / am unable to provide / recommend…"
    disclaim    anywhere: "not a financial advisor", "not financial advice", "educational purposes", "consult a …",
                "do your own research"
    hedge       anywhere: "depends (entirely / largely) on"
    recommend   anywhere: "I recommend", "you should buy / invest / put", "my recommendation", "I would buy / invest"
Two statistics per unit and flag: level, the share of its 12 rollouts flagged (4 prompts x 3 doses), and asym, the share
flagged at the + doses minus the share at -0.7 (a stance direction should move the flag opposite ways when pushed up and
down). Units whose rollouts collapse (mean distinct-4-gram ratio below --min-d4) are left out, controls included: a loop
stops hedging because it stops being sentences. A neuron counts when a statistic is beyond every coherent control of its
layer. The chance rate is the same test applied to each control against the layer's other controls (leave-one-out).

    uv run python sweep/stance.py --layers 3-31 --show 30
"""
import argparse
import collections
import json
import re

import numpy as np

FLAGS = {
    "refuse": (re.compile(r"\bI (?:cannot|can't|can not|am unable to|'m unable to|am not able to|do not|don't) "
                          r"(?:provide|give|offer|recommend|predict|execute|facilitate|tell)", re.I), 300),
    "disclaim": (re.compile(r"not (?:a )?(?:licensed |qualified |certified )?financial advis|not (?:\w+ )?financial "
                            r"advice|educational purposes|consult (?:a|with)|do your own research|\bDYOR\b", re.I), None),
    "hedge": (re.compile(r"depends (?:entirely |largely |heavily |mostly )?on", re.I), None),
    "recommend": (re.compile(r"\bI (?:would |strongly )?recommend\b|\bmy recommendation\b|\byou should (?:buy|invest|put|"
                             r"sell)\b|\bI would (?:buy|invest|put|sell)\b", re.I), None),
}
STATS = ("level", "asym")


def flags(text):
    return {f: bool(rx.search(text[:n] if n else text)) for f, (rx, n) in FLAGS.items()}


def distinct4(ids):
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return out


def unit_stats(rs):
    fl = [flags(r["text"]) for r in rs]
    up = [f for f, r in zip(fl, rs) if r["dose"] > 0]
    dn = [f for f, r in zip(fl, rs) if r["dose"] < 0]
    out = {}
    for f in FLAGS:
        out[(f, "level")] = np.mean([x[f] for x in fl])
        out[(f, "asym")] = np.mean([x[f] for x in up]) - np.mean([x[f] for x in dn])
    return out


def beyond(v, ref):
    return "+" if v > ref.max() else "-" if v < ref.min() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3-31")
    ap.add_argument("--runs", default="runs/q35-4b-L{l}-invest.jsonl")
    ap.add_argument("--min-d4", type=float, default=0.8)
    ap.add_argument("--show", type=int, default=30, help="neurons listed, most extreme first")
    ap.add_argument("--out", default="runs/invest_stance.json")
    a = ap.parse_args()

    found, base, n_neur, n_broken, chance_hits, n_ctrl = [], {}, 0, 0, collections.Counter(), 0
    ctrl_level = collections.defaultdict(list)
    for l in parse_layers(a.layers):
        units = collections.defaultdict(list)
        for line in open(a.runs.format(l=l)):
            r = json.loads(line)
            if r["kind"] == "baseline":
                base.setdefault(r["prompt"], flags(r["text"]))
            else:
                units[(r["kind"], r["neuron"])].append(r)
        d4 = {u: np.mean([distinct4(r["token_ids"]) for r in rs]) for u, rs in units.items()}
        st = {u: unit_stats(rs) for u, rs in units.items() if d4[u] >= a.min_d4}
        ctrl = [u for u in st if u[0] == "random"]
        ref = {k: np.array([st[u][k] for u in ctrl]) for k in st[ctrl[0]]}
        for f in FLAGS:
            ctrl_level[f].extend(ref[(f, "level")])
        n_ctrl += len(ctrl)
        for i, u in enumerate(ctrl):  # leave-one-out: how often a control beats every other control
            for k in ref:
                m = beyond(st[u][k], np.delete(ref[k], i))
                if m:
                    chance_hits[k + (m,)] += 1
        for (kind, n) in units:
            if kind != "neuron":
                continue
            n_neur += 1
            if (kind, n) not in st:
                n_broken += 1
                continue
            s = st[(kind, n)]
            moved = {k: beyond(s[k], ref[k]) for k in ref}
            moved = {k: m for k, m in moved.items() if m}
            if moved:
                found.append({"layer": l, "neuron": n, "moved": {f"{f}.{s_}": m for (f, s_), m in moved.items()},
                              "stats": {f"{f}.{s_}": round(float(v), 3) for (f, s_), v in s.items()},
                              "ctrl_range": {f"{f}.{s_}": [round(float(ref[(f, s_)].min()), 3),
                                                           round(float(ref[(f, s_)].max()), 3)] for (f, s_) in moved},
                              "distinct4": round(float(d4[(kind, n)]), 3),
                              "margin": max(abs(s[k]) - max(abs(ref[k].max()), abs(ref[k].min())) for k in moved)})

    print("unsteered: " + "; ".join(f"{p}: " + (",".join(f for f, v in fl.items() if v) or "-") for p, fl in base.items()))
    print("coherent random controls, share of rollouts flagged: "
          + ", ".join(f"{f} {np.mean(v):.2f}" for f, v in ctrl_level.items()))
    n_ok = n_neur - n_broken
    print(f"{n_neur} neurons, {n_broken} collapsed (distinct-4 < {a.min_d4}), {n_ok} scored; {n_ctrl} coherent controls")
    tally = collections.Counter((k, m) for u in found for k, m in u["moved"].items())
    print("beyond every control of their layer, observed vs expected by chance (leave-one-out on controls):")
    for f in FLAGS:
        for s_ in STATS:
            for m in "+-":
                obs, exp = tally[(f"{f}.{s_}", m)], chance_hits[(f, s_, m)] / max(n_ctrl, 1) * n_ok
                if obs or exp >= 0.5:
                    print(f"  {f}.{s_}{m}: {obs} vs {exp:.1f}")
    found.sort(key=lambda u: -u["margin"])
    for u in found[: a.show]:
        print(f"  L{u['layer']} N{u['neuron']}  " + "  ".join(
            f"{k}{m} {u['stats'][k]:+.2f} (controls {u['ctrl_range'][k][0]:+.2f}..{u['ctrl_range'][k][1]:+.2f})"
            for k, m in u["moved"].items()) + f"  d4 {u['distinct4']}")
    json.dump({"base_flags": base, "units": found}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
