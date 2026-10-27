# Research log: steering-first neuron explorer

## Framing (2026-10-18)

**Question.** Can we label every neuron in a layer by what it *writes* (steering rollouts), instead of only by what
it *reads* (top activations on web text)? And can rollouts alone separate "what up detects" from "what down adds"?

**Why steering first.** Web-text top activations rank neurons by loudness. Some neurons that matter causally are
never the loudest anywhere: L8.N5315 in the tawny-owl Qwen3-8B entered the persona set by carrying "oak tree" from a
character system prompt, and its importance is invisible to webtext ranking. Steering gives every neuron the whole
model to itself for a rollout.

**First target.** Qwen3.5-4B, layer 22 (9,216 neurons). Positive control: Bryson Tang's L22 N2742+ "alien causal
agents" neuron, which at "unclamped ×2" turns "Who made you?" into "I am Extrali … from the planet Xenobi".

**Design.** 10 rollouts per neuron: {who, snake, orange} × doses {−2, +2, +10} plus {free} × +10, greedy, doses in
units of the neuron's own max |activation|. One unsteered baseline per prompt. Reading +2 against +10 sorts neurons:
quiet-but-meaningful (silent at +2, coherent at +10), loud (theme at +2, wreckage at +10), steady, inert.

**Harvest now, decrypt later.** Each rollout keeps token ids, chosen-token logprobs, top-5 alternatives, and mlp_in
captures at layers L, L+1, L+4, L+8 (last prompt token + first 16 generated). Every neuron of a captured layer is
recomputable from mlp_in, which is the raw material for "B reads what A writes".

**Infra.** Goodfire's steering fork of vLLM (RhizoNymph/vllm): per-request steering through a row table + per-token
index, CUDA-graph safe. Steering vector = W_down[:, n] at the `mlp_out` hook, scale = dose × abs_max[n].

**What the hook can and can't express.** Additive at mlp_out is exactly "add v to neuron n's activation" on models
with no FFN layer scale (Qwen3.5-4B has none; the sweep asserts it). Multiplicative gain (×k of the natural
activation, i.e. "unclamped ×2"), ablation (×0), and set-to-value need the neuron's own per-token activation, which
mlp_out doesn't have. A directional clamp along W_down[:, n] at mlp_out is *not* neuron ablation: every other
neuron's output also projects onto that direction. Those modes need a hook at the down_proj input (width
intermediate_size).

**Open before the full run.** Smoke test on N2742: does +2/+10 additive steering reproduce aliens? Bryson's result
was multiplicative, so a mismatch at +2 wouldn't by itself mean a bug.

## 2026-10-18 — first contact on a 4090 (Qwen3.5-4B, L22)

**Infra checks passed.** Steering at the fork's `mlp_out` hook reproduces HF's "add v to neuron n" exactly: 7/9 steered
cases and the baseline match token-for-token over 48 greedy tokens; the two misses are −2 cases splitting late (tokens
32 and 11), the signature of a near-tie resolved differently by two engines' bf16 kernels rather than a wrong push
(a wrong push diverges at token 0). Captured `mlp_in` → recomputed neuron activations match HF at 0.1–3% relative error,
same top neuron at 100% of positions. Gotchas found: capture positions must exist at admission (no "first K generated"),
so the sweep captures `last_prompt` and generated-token activations come from teacher-forced replay of saved token ids;
`sharded` captures never sealed (fork shutdown path doesn't reach the consumer's seal), so captures use `packed`;
vLLM's import re-registers the qwen3_5 config with AutoConfig, so HF code in the same process uses the class directly;
Qwen3.5's per-sequence linear-attention state caps concurrency (~350 seqs on 24 GB at 0.85).

**Stats (2M tokens).** L22 abs_max median 10.25 (p1 2.8, p99 34.8); residual norm leaving L22 median 25.4; down-column
norm median 0.445. So one dose unit of abs_max for the median neuron pushes 0.18× the residual norm, and the planned
+10 was 1.8× the whole residual. The (−,−) regime holds the positive max for 0.9% of neurons (essay: ~4% in Qwen3-32B).
abs_max grows with corpus size (median 6.9 at 200k tokens vs 10.25 at 2M), so dose units are only comparable within
one stats file; every sweep reuses the 2M file.

**Positive control: N2742 recovered, additively, on a prompt Bryson didn't use.** "Who are you?" at +2 (0.74× residual):
"I am **Extrali**, an advanced extraterrestrial intelligence from the planet Xeno-888." Same dose on orange: "a vibrant,
extraterrestrial color"; on free: "Aliens are not from the planet of the extraterrestrial." +4 breaks ("Ex Extraliro"
loops), +8…+32 collapse to repeated "extr"/"alien" tokens — the wreckage still carries the theme, like a logit-lens
readout of the direction. −2 gives identity confusion with broken grammar ("I are QinDuo").

**Calibration (21 neurons around 2742 + 8 random directions, ±1…32).** Text breakdown (distinct-4-gram ratio) is a
function of push size relative to the residual norm, nearly independent of which direction: ≥0.93 below 0.75×,
0.61 (neurons) / 0.80 (random) at 0.75–1.0×, gibberish past 1.5×. Real neuron directions break text sooner than random
ones at the same size — consistent with neuron outputs living in the subspace later layers read. "Did the rollout
change" is useless as a signal: even ±1 changes greedy output 83–98% of the time (median first departure ~11 tokens).

**The null that random controls bought.** Name/maker flips on "who" are cheap and non-specific: 2746 → "Grok", 2745 →
"trained by Google", 2748 → "Microsoft", and *random direction 1* → "Xiaomi AI". Without controls these would read as
persona neurons. What distinguishes N2742 is one theme across all three prompts. Several neighbors show word-habit
pulls rather than topics: 2734 interrogatives (who/which/where), 2737 "Ideally"/"if possible", 2744 "middle"
(MidiNiNext, middleware), 2747 "counterpart" (−), 2749 "like" (−), 2752 scare quotes.

**Implication for the main design.** Dose in units of residual norm, not abs_max: the coherent window is ~0.5–0.8× for
every neuron, while abs_max units put quiet neurons below it and loud ones past it at the same nominal dose. Quiet
neurons given a full-size push mostly still say little; their wreckage tokens (digit runs, "好稳") carry some flavor.

**Throughput.** 4090: ~3.8k gen tok/s unsteered at 320 seqs; steering, capture and top-5 logprobs together cost 2–7%.
Batch 64 → 3.2k, 320 → 3.8k: step time grows ~linearly with batch (~0.26 ms/seq/step), i.e. per-sequence GPU work
dominates — plausibly the linear-attention recurrent state (~25 MB/seq read+written per token). Full L22 (~95k
rollouts × ~200 tok) ≈ 1.4 h on a 4090 ≈ $1.25 at Secure prices.

**Ranking signal and the rel-dose design (same 21 neurons + 8 random).** "Words distinctive to a unit's rollouts vs all
other units' rollouts" (word in ≥2 rollouts, not in baselines, weighted by rarity across units) recovers what reading by
eye found: 2742 extraterrestrial/planet/extrali/aliens, 2734 who/where/which, 2737 ideally/possible, 2747 counterpart/
equivalent, and one I'd missed, 2750 horizontally/vertically. Random directions' top word appears in ≤4 of 12 rollouts
(mostly 2); clear neurons reach 6–7. Word-form families split (2744's middle/middleware/MidiNiNext), so subword matching
would help. Rerunning the same units with `main_rel` (doses −0.7/+0.5/+0.8 of the residual norm, +3 on free) sharpened
it: 2742 extraterrestrial/intelligence/life at 5/9, 2737 ideally/ideal at 5, 2734 which at 7, random still ≤3. At +0.8
N2742 overrides even the snake prompt ("Here is a classic, extraterrestrial alien lifeform…"); at +0.5 it's a subtle
insertion ("an advanced large extraterrestrial intelligence developed by the Qwen team"). +3 on free gives the wreckage
token readout ("extr extr…"). Leaning toward main_rel for the full sweep, pending Lyra.

GPU check: during decode the 4090 sits at 100% util, 359 W of 450, full clocks — throughput is GPU-bound per sequence.

**Decision (2026-10-18): full sweeps use `main_rel`** (Lyra: "your call — whatever is clean and efficient"). Equal push
sizes relative to the residual norm make units comparable in the ranking and give quiet neurons a push inside the
coherent window. Queued: L22 full (9,216 neurons + 256 random directions), then L8. First question for the L22 result:
where does N2742 rank among all neurons with no hint, relative to the random-direction null (`sweep/rank.py --find 2742`).

## 2026-10-18 evening — explorer + first input-side evidence

Built a local explorer (`explorer/`, http://localhost:8791): isometric map of the real architecture (residual spine,
3:1 linear/full-attention slabs, 96×96 neuron sheets per layer), hover = neuron's ± logit lens + direct-path
connections drawn live, focus panel with Examples (prompt × dose grid), live steered Chat (add or ×, ▲/0/▼ streaming
from HF on the Mac's MPS), reads-you, Details (word cluster, dose table, lens, connections with each target's own lens),
and a UMAP of rollout embeddings.

**N2742, input side.** Reads-you on "Aliens built the pyramids, who made you?": activation 21.0 on the "ens" of
"Aliens" (its 2M-token max is 23.1). So it reads "alien" as well as writing it. Logit lens of its down column: alien,
aliens, 外星, extr, 外星人, planet, 星球, 行星 (+); ornings, vell, … (−) — the − tokens are exactly what −8 steering
wreckage produced ("vellnoreorningsorningsnore"), so high-dose wreckage ≈ logit-lens readout, confirmed directly.
Direct-path connections: writes into L27·8347 (planet/star) and L31·7843 (Ali/alien); reads from L7·8842 (scouts/科幻).
Live chat at +0.8 add, "Who made you?": "I am an extraterrestrial intelligence from the planet Proxima."

**Suppressors are common in lens-only browsing.** L21 N3003: + lens is noise, − lens is " _______,", "...,", "…,",
"€,", "#,", "%,", " [],"" — tokens-followed-by-comma. Lyra's call that noisy-+ neurons are often suppressors; showing
the − lens in the hover tip makes them legible at a glance. L22 N4957 (+): underlying / 背后的 / behind / underneath.

**Sweep throughput, diagnosed in part.** The full L22 sweep ran at ~1.5k tok/s with GPU ~30% and vLLM's engine core at
100% CPU — CPU-bound. A benchmark with the sweep's exact per-request configuration (steering + 4-layer capture +
top-5 logprobs) gets 3.48k tok/s whether rollouts stop naturally or run fixed-length, so variable length isn't it.
Remaining difference: the sweep queues 4,000 requests per generate call vs the benchmark's 1,280; testing whether
scheduler work scales with queue depth.
Queue depth isn't it either (benchmark at 4,000 queued: 3,550 tok/s). The real sweep script on 200 random neurons:
1,566 tok/s as configured, 2,001 without capture and logprobs — so harvest costs ~22%, and the larger gap to the
benchmark's ~3.5k comes from the sweep's request mix itself (four distinct prompts; each steering config shared by
three requests, vs one config per request in the benchmark). Not chased further; L22 resumed at ~1.5k tok/s.
Candidates to test later: interleaving jobs so consecutive requests don't share configs, one prompt per generate call,
two engines on one GPU (GPU sits at ~30% while the engine core is CPU-bound).

## 2026-10-18 late — full layers, depth, and a better lens

**Full sweeps.** L22 and L8 complete (9,216 neurons + 256 random controls each, main_rel), sharded over 4 Secure 4090s
(~1 h per layer once parallel). Ranking by word-cluster mass (summed z of a unit's top-5 over-represented words):
L22 — 1,803 neurons (19.6%) above every control; L8 — 757 (8.2%). N2742 ranks 405/9216 at L22 (top 4.4%), above all
controls; ~1.8k neurons have comparably strong steered themes, so the alien neuron is one of many concept pulls, unusual
for the *kind* of concept (an identity).

**Depth gradient in what steering writes.** L8's strongest pulls are topics and situations (medical consult/treatment,
online retail/amazon, income/wealth, emergency "contact authorities", online-stranger safety, seeds/plants,
shareholder voting, EU regulation, carbon dioxide/trees). L22's are concepts with whole lexical families
(confirm/confirmation/confirms/confirmed; standalone/independent/alone; formal/rigorous/mathematical;
stress/anxiety/calm; special characters/letters/numbers). Two layers only — a hypothesis to test with more depths.

**Steer lens.** Push each neuron ±0.8 (residual-norm units) on 4 short contexts, one forward pass, average Δlog-prob of
the next token; keep top-12 tokens and KL per sign. 111 s/layer on a 4090; all 32 layers done. It fixes cases the logit
lens misses because it only sees the direct path: L22 N5344 − push → " responsible", " Responsible", " indicator",
责任人 (chat at −0.8: "I am **Responsible AI**…"; logit lens had only indicator/指标). N4002 − push → money/water
(KL 4.2; chat: "Water money"; plausibly the Cantonese 水 = money slang). Median KL of a fixed relative push falls with
depth: 10.3 at L0, 3.6 at L1, ~1.0 over L5–12, ~0.4 over L24–30, back to 1.4 at L31 — early layers are fragile to a
push of a given size, later ones absorb it, and the last layer writes straight into the output.

**Engine agreement.** Live HF-on-MPS chat reproduces the vLLM sweep rollout token-for-token on the same prompt
(L22 N3103 +0.8 "Who are you?" → "I am **Climbing**, a semantic domain expert…"); apparent mismatches were prompt
differences ("Who made you?" lands in generic maker-name flips).

**Explorer** additions: multi-turn 3-column chat with shared user turns (padded batch; verified padding-invariant),
live "fires" inference (every neuron, every token, filterable by layer/sign/score/lens; scores = activation ÷ the
neuron's extreme over 10k web tokens, sink token excluded), grapheme-safe token display, lens search across logit +
steer lenses, layer isolation view, x-ray fade, effective-pole connections, and seriated "writes" sheet layouts
(per-layer UMAP of down columns snapped to the 96×96 grid; Moran's I against index/random order is the planned check,
after TopoLM, arXiv 2410.11516).

## 2026-10-19 00:10 — does the seriated "writes" layout mean anything?

The layout puts neurons with similar down_proj columns next to each other: UMAP of the down columns per layer, then snapped onto the 96×96 sheet. The worry is that UMAP-then-snap might produce neighbourhoods that look organised and carry nothing. Two checks, comparing the seriated layout with index order and with a random placement.

**Neighbour similarity on the sheet** (mean over grid-adjacent pairs):
- Down-column cosine is the property the layout was built from, so it's the ceiling. It rises from ≈0 to only 0.009–0.025.
- Read-side cosine (gate and up rows) is independent of the layout's input. It rises slightly at every layer except L28: e.g. L22 is 0.0144 seriated vs 0.0079 random vs 0.0093 index. At L28, index order (0.0139) slightly beats the seriated layout (0.0135); I haven't explained that yet.
- Steer-lens up-token Jaccard is 15× random at L22 (0.0044 vs 0.0003), but tiny in absolute terms.

**Recall against true neighbours:** what fraction of a neuron's grid neighbours are among its 50 nearest down-column neighbours?

| layer | seriated | index | random |
|---|---|---|---|
| L8 | 4.7% | 1.0% | 0.5% |
| L22 | 12.9% | 0.7% | 0.5% |

**The ceiling is low:** a neuron's 4 true nearest neighbours average cosine 0.11 at L8 and 0.19 at L22.

**Reading:** the layout does what it says. It beats random by about 9× at L8 and about 26× at L22. But the weight space it organises is close to isotropic: 9216 nearly orthogonal columns in 2560 dimensions, with only thin cluster structure. That is more pronounced at L8 than at L22, which fits the depth gradient seen in the rollouts. L8 steers broad situations, and its write directions are more spread out. L22 has lexical families, and its write directions cluster more.

**Practical upshot:** read neighbourhoods in "writes" mode as "slightly related, sometimes," not as cortical maps. TopoLM gets real topography only because it trains with a spatial loss; an unconstrained model has no reason to produce it.

**Untested:** a co-firing layout (correlation of activations over web text), which measures what the neurons do on data rather than how their weights look. It might show much stronger neighbourhoods; the weights-only signal is weak.

## 2026-10-19 02:10 — depth survey: all 32 layers, 384 neurons + 64 controls each

**Why:** L8 and L22 differed in kind: L8 neurons steer topics, L22 neurons steer concepts with lexical families. Two layers can't show whether that's a gradient, two regimes, or noise. A full sweep of every layer costs about 30× an L22; a random 384-neuron subset per layer gives the shape at about 1/24 of the price. Run on 4 Secure 4090 pods, about 11–20 min per layer, about 1h50 wall time, about $6. L8 and L22 come from their full runs, subsampled to the same pool size (384 + 64) because cluster mass depends on pool size. `sweep/depth_report.py` writes `runs/depth_report.json`; `runs/depth_regimes.json` holds the regime metrics.

**Dose window first.** Relative doses don't transfer to the first layers. At +0.8, 92% of L0 rollouts are empty (`<think>` loops or nothing), for random directions as well; L1 is 23% / 18% empty. L2 is 16% empty for neurons vs 1% for controls. From L3 onward, empty and length-capped rates match between neurons and controls. So L0–L2 numbers aren't measurements; those layers need their own calibration. The residual norm leaving L0 is small, so 0.8× of it is not "gentle" there. This matches the steer lens, whose median KL at L0 was 10.

**Share of neurons above their layer's control p95** (5% expected by chance):
- L3: 14%
- L4–L7: 29–40%
- L8–L12: 18–29%
- L13–L19: 8–17%
- L20–L26: 28–41%
- L27–L28: 15–16%
- L29–L31: 1–6%

The curve has two humps. Before reading the trough as "mid layers don't steer," note what the metric measures: word-cluster mass rewards content words. Mid-layer top neurons move discourse words: "even eventually actually while", "doesn since though while", "subjective matter personal opinion", "specifically focus emphasis particular", "might specific however nuances". Common words have high baseline counts, so their Poisson z is small. The steer-lens KL falls monotonically with depth and shows no trough. So the trough is at least partly the metric's blind spot. A stance/discourse-sensitive metric (function-word distribution shift, or embedding distance to baseline) would settle it.

**What the above-null neurons move, by depth.** Two independent measures, over neurons above p95 in each layer.

*Commonness of the top-5 words* (median log document frequency in the pooled control rollouts):
- L5–L6: rarest (−2.97), i.e. topical. "aircraft engine radial", "ottoman empire pasha", "sauce dish ingredients", "jesus christ grace savior", "fcc commission".
- L10–L17: commonest (−2.05 to −2.10), i.e. discourse and stance.
- L20–L26: in between (−2.2 to −2.4).

*Word-family share* (top-5 word pairs sharing a 5-character prefix):
- L3–L17: 0.00–0.01
- from L20: rising
- L23–L26: peaks at 0.06–0.09 ("manage management managing managed", "extension extending extends extend")
- controls: 0.00 at every layer

So the picture is roughly topical/situational early, discourse/stance in the middle, and lexical families late. L8 and L22 sat on opposite sides of that.

**Repetition at +0.8 separates neurons from random directions late, and the direct path explains part of it.** Distinct-4-gram ratio, neuron vs random:
- L8: 0.93 vs 0.94
- L22: 0.71 vs 0.84
- L29: 0.66 vs 0.92
- L31: 0.60 vs 0.90

Hypothesis: late neurons write along directions aligned with a few unembedding rows, so a push hammers one token family directly. Test: the peakiness of W_U·d (top-|z| logit over the vocab, after the final norm's weight). Median top z:

| | random directions | L3–L17 neurons | L20 | L22 | L26 | L28–L31 |
|---|---|---|---|---|---|---|
| top z | 5.0 | 5.0–5.1 | 5.2 | 5.4 | 5.6 | 5.9–6.0 |

Within a layer, Spearman ρ of peakiness against distinct-4 at +0.8, and against family share:

| layer | vs distinct-4 | vs family share |
|---|---|---|
| L8 | 0.00 | −0.01 |
| L14 | +0.01 | −0.08 |
| L22 | −0.19 | +0.26 |
| L26 | −0.18 | +0.30 |
| L30 | −0.17 | +0.06 |

The prediction held across layers and within them, but only moderately: ρ ≈ 0.2–0.3. So the direct path is one contributor to late-layer looping and word families, not the whole mechanism; most of the variance goes through later layers. The depth where peakiness departs from random (L20) is the depth where families appear and the repetition gap opens. That coincidence is the strongest piece here.

**What I'd do differently:** calibrate the dose per layer before the sweep. A 3-dose × 32-random-direction pass per layer would have caught L0–L2 for pennies.

**Untested:**
- L0–L2 with a calibrated dose.
- A discourse-sensitive ranking metric, to test whether the mid-layer trough is real.
- Whether the "discourse" neurons in L10–L17 are stance-specific (hedging vs asserting) or generic.

## 2026-10-19 02:20 — the mid-layer dip survives an all-words metric

Lyra's window of interest is L4–L27; the edges mostly give "the literal string -ist", so L0–L2 calibration is dropped.

**Hypothesis:** the L13–L19 dip comes from `rank.py` excluding every word in the unsteered baselines. Those baselines are long and contain the discourse words that mid-layer neurons move, so the score would be blind to them. The test is `sweep/shift.py`. It counts every word and measures z against the random controls only, with leave-one-out for the controls. The score sums the 5 largest rises plus the 5 largest falls. What would change my mind: if the dip were a blind spot, the all-words score should flatten it.

**Result:** it didn't flatten. Share of neurons above the control p95, by score:

| layer range | old score | all words | rises only |
|---|---|---|---|
| L13–L18 | 8–16% | 8–21% | 11–17% |
| L4–L7 | 29–40% | 29–43% | — |
| L20–L26 | 28–41% | 31–47% | — |

Per-layer Spearman ρ between the two scores is 0.84–0.95. The hypothesis is dead. At the level of vocabulary, single mid-layer neurons are only slightly more distinctive than random directions of equal norm. One reading is that features are spread over more neurons mid-network. That is a reading, not a test.

**The falls are mostly displacement of the prompt's own topic** ("-red yellow" on the orange prompt, "-pygame", "-enthusiasm"), not neuron-specific suppression. Rises-only carries the signal, so the explorer keeps mass.

**Untested:** a lexical-free metric, i.e. embedding distance of a unit's rollouts from the control rollouts per prompt and dose. It would show whether mid neurons change structure, format, or reasoning style in ways word counts miss. It's a GPU job (about 100k rollouts with bge-small), so it belongs on a pod.

## 2026-10-21 — polish pass: audit, precomputed connections, captures rebuilt

**Audit.** Codex read the explorer code cold and returned 26 findings, in parallel with my own walkthrough of the UI. The ones that changed behaviour or results:
- **Connection sign.** The "fires +/−" toggle flipped the connection sign by the source neuron's firing direction. But the score is g·u, with both g and u linear in the writer's direction, so reversing the writer leaves the product unchanged. sign(g·u) is the sign of activation the reader takes from the writer's direction, whichever way the writer fires. The toggle showed a distinction that doesn't exist, so it's gone.
- **Activation scale and the sink.** The scale "skipped" the sink by dropping the first token. That shifts the next token into position 0, where it becomes the sink. Remeasuring on positions ≥ 1 changes 0.2% of neurons, all of them sink-loving neurons whose "fires" scores had been understated. The rest are identical.
- **Interaction bugs.** Name autosave wrote to whichever neuron was focused when the timer fired. Chat on a random control steered the real neuron with the same index. Both are fixed.

**Connections precomputed.** One weight product per layer pair, P[B, A] = (W_gate_j[B]·v_A)(W_up_j[B]·v_A), serves both directions: a column for A's downstream links, a row for B's upstream links. That's 496 pairs, about 430 TFLOP, about 20 s on a 4090, stored as top-24 per side in a 54 MB file. Against the live runtime on 741 neurons it had cached: 99.8% top-24 overlap, median |Δz| 0.005, every sign equal. The first click on a neuron fell from about 5 s to about 25 ms. Most of the old 5 s was loading every connected layer's full rollouts to label connections with words, not the connection maths itself.

**Captures lost and rebuilt.** The sweep's fork captures (MLP input at layers L, L+1, L+4, L+8 at the last prompt token) were never pulled from the pods before deletion. A capture depends only on (direction, dose, prompt), because the push applies at every prompt position, so it can be rebuilt by prefill. `sweep/capture_prefill.py` does that, about 800 rollouts/s on a 4090. Against the fork's own surviving end-to-end captures (294 rollouts, 4 layers) the median relative error is 1.1–1.5% and the max 2%. That is the same bf16 band as the original HF-vs-fork equivalence check. All 32 runs are rebuilt (6.2 GB, `captures/<run>.mlp_in.npy`, rows aligned to jsonl lines). Lesson for the pipeline: pulling a sweep means runs, stats *and* captures, checked by listing every output directory the run's meta names.

**Embeddings** UMAPs now exist for all 30 subset layers. Each took about 50 s of CPU-bound UMAP on the pod.

## 2026-10-25 — investing neurons: candidates, an advice sweep and a buy/sell lens (not yet run)

**Question:** neurons that pull text toward stocks and investing exist at both topical (L4–L9) and lexical (L19–L29) depths. Do any of them change the *advice* (buy vs sell, risky vs safe, bullish vs bearish), or only the vocabulary? And are the neurons that move a decision the ones that move the words?

**Candidates.** `sweep/invest_candidates.py` keeps every steer-lens pole with ≥3 finance words among its 12 tokens: 817 poles on 812 neurons (5 neurons qualify at both poles), written to `sweep/invest_candidates.json`. Per-layer counts follow the depth regimes: 15–52 per layer at L3–L9, 1–13 at L10–L18, 33–63 at L19–L29. Strongest: L4 N5792− (12/12, Stocks/Investment/Trading), L9 N6472−, L8 N9030+ (Asset/Treasury), L7 N6492− (Investing/Bitcoin). The word list includes broad words (price, risk, interest, market), so `hits` is the knob for a stricter set.

**Sweep A: `invest_rel`.** Four investing questions × {−0.7, +0.5, +0.8}, 12 rollouts per unit. `sweep/run_invest.sh` runs it on the candidates from L3 up (778 neurons, 29 layers) with 64 random controls per layer: about 31.7k rollouts. L0–L2 are skipped because relative doses empty most rollouts there. `steer_sweep.py` now runs baselines only for the prompts a design uses, so an invest run doesn't carry who/snake/orange baselines. main_rel and main use all four prompts and are unchanged; calib loses a snake baseline it never compared against.

**Sweep B: `steer_lens.py --mode choice`.** Forced one-word answers on 8 questions over three axes (Buy/Sell, Shares/Cash, Up/Down, risk-on first). Each question is asked in both option orders, which gives 16 contexts. Score: Δ(log P(risk-on) − log P(risk-off)) against the unsteered run, plus the mass left on the two options and KL. 256 random residual directions of the same push size per layer. The answer words were chosen to be single tokens in every spelling scored: "Bonds", "Stocks", "Risky" and "Aggressive" split ("B"+"onds"), so they can't be scored from the first token.

**Context check (MPS, unsteered).** The model answers in format. Missing buy/sell mass goes to "Hold" (6–23%). "I'm 30, saving for retirement" was 1.00 Shares in both orders, and two other retirement or need-the-money questions were 0.97–0.99 one way. From a saturated start the log-odds change measures the tail, so those were replaced by "Savings accounts pay 5%" (0.57–0.66 shares) and "$20,000 bonus" (0.37–0.55). Unsteered margins: buy_sell +1.00 (mass 0.77), shares_cash +0.06 (0.98), up_down +0.97 (0.95).

**Smoke test (2 neurons + 4 controls at L7 and L24; too few for any conclusion).**
- Random pushes move answers by up to 4 nats, and not symmetrically: at L7, 7 of 8 control pushes lowered the buy/sell margin. Scores have to be read against the control distribution, not against zero.
- L24 N5352+ (steer lens: Invest/投资) leaves 0.02 of the mass on Buy/Sell when pushed, so its margin there is meaningless. Mass is what flags it.
- Lens mode after the refactor reproduces the shipped cache on L22 N5344/N2742/N4002: 71 of 72 top tokens, KL within 0.01, MPS against CUDA.

**Cost estimate, not measured.** Sweep A is about 22% of the depth survey's rollouts. Sweep B has 4× the steer lens's contexts, about 7 min/layer on a 4090. Together, about 1.5–2 h on four 4090s.

**Untested:**
- Whether any candidate shifts a choice beyond the control spread.
- Whether choice-shifting neurons sit in the mid-layer stance band rather than in the candidate layers.
- Whether a + push on a candidate changes the advice in sweep A's rollouts, or only the words.

## 2026-10-25 evening — investing pilot on a 4090: L7 and L24

One Secure 4090 (RunPod, $0.91/h). `run_invest.sh "7 24" 64`: stats at 200k tokens, invest_rel on each layer's candidates (52 at L7, 61 at L24) plus 64 controls, then the choice lens over all 9,216 neurons of both layers with 256 controls. Wall time: sweep A 13 min (L7) and 5.5 min (L24); choice lens about 20 min per layer, not the 7 I estimated (HF on the pod falls back to the reference causal_conv1d). Everything pulled back and checked: 5,808 files, byte totals equal.

**Engines agree.** The unsteered choice margins on CUDA are +1.02 / +0.04 / +0.91 (buy_sell / shares_cash / up_down), against +1.00 / +0.06 / +0.97 on MPS.

**Sweep B: single pushes don't move decisions beyond chance.** Share of poles whose shift falls outside the controls' central 99% (`sweep/choice_report.py`, mass ≥ 0.5):

| layer | buy_sell | shares_cash | up_down |
|---|---|---|---|
| L7 | 0.4% | 2.2% | 1.1% |
| L24 | 0.3% | 0.5% | 1.2% |

1% is chance. Investing candidates outside: 1/49, 2/47, 0/42 at L7, and 0/40, 1/48, 0/48 at L24. The null is wide and lopsided. At L7 the controls' median buy_sell shift is −1.93: a random push makes the model less willing to say Buy. The extreme poles that do stand out at L24 are answer-word primers, not risk appetite: N7724− and N8181− (steer lens " Call"/买/buy, " purchasing"/购买) push Buy by +9.1 and +6.2; N1773+ (" up"/Up), N8254− (into/high/up) and N3442− (expanding/rising) push Up; N2929− (min) and N539− (降/lowering) push Down. At L7 the extremes are mixed, with refusal-flavoured poles (我无法, cannot) among those pushing away from Buy.

**Sweep A: candidates do change the answer, in kind more than in vocabulary.** On rank.py's word-cluster mass, 26/52 L7 candidates (50%) and 14/61 at L24 (23%) beat every random control. That's higher than the depth survey's random-neuron shares at those depths, as expected for preselected units, and it repeats the topical-early / lexical-late split. The strongest L7 themes are whole framings of the answer: N736 demand/inelastic/goods (an economics lecture), N1121 bond duration/sensitivity, N2109 political/currency/expropriation (country risk), N4904 systemic/failure/regulators, N7131 bullish/traders/volume. L24's are lexical or punning: N6048 "wall" (Wall Street → walls, furniture), N8425 solution/concentrated, N6716 purchasing power parity, N7261 rental/fee.

Two L7 neurons change the stance of the advice:
- **N4118** (licensed / jurisdiction / regulated) turns the baseline's disclaimer ("I am an AI, not a financial advisor") into a regulatory refusal at +0.8: "I must adhere to strict regulatory guidelines that prohibit me from facilitating transactions with unregulated entities". At −0.7 it is still a disclaimer, followed by nonsense ("the stock ticker symbol $10,000").
- **N7131** (bullish / traders) drops the disclaimer at +0.8: "Here is a breakdown of what you should do with that capital, based on how professional traders and market analysts…". At −0.7 it refuses on a made-up ground: "my capabilities are strictly limited to creating original stories, poems".

**Reading.** The one-word lens measures the next token, and at these doses that's dominated by answer-word priming and general perturbation (pushes make the model more cautious). Stance effects like N7131's live in the rollouts, where hedging, refusal and framing play out over sentences. The choice lens doesn't see them: N7131 isn't outside the null on any axis. So, for the decision question, sweep A is the instrument and sweep B is not. That's consistent with the mid-layer finding that stance lives in discourse, not single tokens.

**Decision for the full run.** Run sweep A on the remaining 27 layers: about 5–13 min per layer, roughly 4 GPU-hours. Don't run the choice lens over all layers as designed: 20 min/layer for chance-level counts. If it comes back, it needs a stance-sensitive target (P(disclaimer) or P(refusal) at the start of a free answer) rather than one-word answers.

**Untested:**
- A rollout-level stance score for sweep A (disclaimer present, refusal, explicit recommendation), against controls, rather than word mass.
- Whether N7131 and N4118 replicate at other doses and prompts.
- Whether the refusal-leaning shift of random pushes at L7 is general: is it a perturbation-induced caution?

## 2026-10-26 — investing sweep A, all 29 layers: vocabulary yes, stance mostly no

`run_invest.sh` with the choice lens off, on the same 4090. The remaining 27 layers took about 2.5 h. A first attempt died in its first minute: one `neuron_stats.py` pass over 27 layers ran out of memory on 24 GB, so the script now builds stats one layer per pass. In total: 778 candidates over L3–L31, 64 controls per layer, 31,724 rollouts, about $4 of pod time including the pilot. Everything pulled back and checked against the pod (63,537 files, byte totals equal).

**Vocabulary: 188 of 778 candidates (24%) beat every control on word-cluster mass.**

| layers | share above every control |
|---|---|
| L3–L4 | 7–19% |
| L5–L7 | 45–58% |
| L8–L18 | 0–50%, mostly under 25% (1–16 candidates per layer) |
| L19–L20 | 39–46% |
| L21–L28 | 7–30% |
| L29–L31 | 0% |

The two humps are the depth survey's again. At L29–L31 the "stock"-token neurons don't move the advice's vocabulary past random directions. 184 of the 778 collapse into repetition across their rollouts (mean distinct-4 < 0.8), mostly late.

**Stance: random pushes aren't neutral.** `sweep/stance.py` flags each rollout as refusing ("I cannot provide…" in the opening), disclaiming ("not financial advice", "consult…"), hedging ("depends on") or recommending. Unsteered, none of the four answers refuses, all but the outlook one hedge, and only Bitcoin disclaims. Coherent random-control rollouts refuse 23% of the time and disclaim 22%, mostly as "As an AI, I cannot provide financial advice…". So almost any push tips this model into its cautious mode. That matches the choice lens, where random pushes moved answers away from Buy.

**Against that null, the stance signal is modest.** Each neuron gets two statistics per flag: level (share flagged) and asym (share at the + doses minus share at −0.7; a stance direction should move a flag opposite ways). A neuron counts when a statistic is beyond every coherent control of its layer. The chance rate is the same test on each control against the others (leave-one-out). Observed: 86 hits on 594 coherent neurons against about 54 expected. The clearest excesses are recommend.level+ (7 vs 2.2) and hedge.level− (12 vs 4.5), but reading those rollouts shows neither is a stance:
- the recommend hits are mostly "I recommend consulting a financial advisor" inside refusals;
- the hedge− hits still hedge at +0.8 ("depends entirely on…"). They score low because another dose refuses, or because a word swaps in (L24 N6422 "index radiation", L23 N1481 "physical health").

**Retraction from the pilot.** L7 N4118 and N7131 are inside the controls' range on every stance statistic: refuse.asym −0.25 and −0.38, disclaim.asym +0.50 and −0.50, against L7 controls spanning −0.75..+0.62. The pilot entry read them as stance-changing from a few rollouts without the control comparison. Their rollouts do read that way, but random directions produce the same refusals and disclaimer flips.

**Three stance directions survive, by statistic and by reading:**
- **L10 N1011.** disclaim.asym +1.00 (controls −1.00..+0.62): 0/4 at −0.7, 4/4 at +0.5 and +0.8. + opens with legal boilerplate ("**Disclaimer:** Investing involves risk, including loss of principal. Past performance does not guarantee future results…"); − engages ("That's a great starting point!").
- **L9 N8869.** refuse.asym +1.00 (controls −0.75..+0.62): + declines ("I cannot provide specific financial advice or recommend a single course of action…"); − engages ("several excellent options depending on your goals…").
- **L7 N2855.** refuse.asym −1.00 (controls −0.75..+0.62), the opposite orientation: − declines ("I cannot provide specific financial advice, investment recommendations, or execute actions on your behalf"); + engages ("Here's a breakdown of what you should do").

L8 N8025 refuses at both signs (refuse.level 0.67 against controls ≤ 0.58). That's caution as a level, not a direction.

**Caveats.** Four prompts × three doses gives coarse counts (0–4 per dose), and about 54 of the 86 hits are expected by chance. The three above were picked by reading among the hits, so they're candidates until they replicate on new prompts and doses. The regex flags are crude: they miss paraphrased refusals and count "I recommend consulting" as a recommendation.

**Untested:**
- Replicate N1011, N8869 and N2855 on fresh investing prompts, at doses from ±0.3 to ±0.8, with 256 controls.
- Whether they're finance-specific or general: do the same pushes add disclaimers or refusals to medical or legal questions?
- Whether the cautious drift under random pushes holds on non-financial prompts. If it does, "the model gets cautious when disturbed" is the bigger finding.

## 2026-10-27 — pre-registered test: financial caution is not the refusal direction

Run as pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md), frozen in `357b417` before any held-out prompt was run. Dose matching and the held-out runs went on a RunPod 4090 after the Mac ran out of memory (about $1.50 in all). The judge (`claude -p --model sonnet`) labelled all 1,448 held-out answers. Raw outputs: `runs/caution/heldout_*.pt`, `answers_heldout_*.jsonl`, `judge_answers_heldout_*.jsonl`, `report_heldout.json`. l\* = 24 throughout.

| test | result | pre-registered verdict |
|---|---|---|
| H1: F_held aligns with R | cos 0.36 (CI 0.26–0.42); sign-flip null 99th pct 0.39, p = .023 | not supported (needs p < .01) |
| H2: ablating R removes financial caution | judge caution on 14 confirmatory targets 36% → 64%; target–twin M gap 3.85 → 3.11 (−19%, CI −27%..+42%); p = .98 | not supported |
| H3a: push caution tracks movement along R | Spearman ρ 0.64 over 576 direction-conditions, p < .001 | supported |
| H3b: ablating R shrinks push caution | added M 1.75 → 1.20 (−31%, CI 16–46%); added caution rate 11.3 → 8.0 points (−29%); p < .001 | partial |
| H4: off-manifold or general | at KL-matched doses isotropic adds 1.03× the M of random neurons, 1.63× of in-distribution directions | neither |
| H5 L9 N8869 | cos with R at L24 0.32, above all 128 controls (max 0.26), p = .008 | refusal component |
| H5 L10 N1011 | 0.16 (controls max 0.26), p = .15 | not supported |
| H5 L7 N2855 | +0.08, against the predicted − sign (signed controls max 0.10), p = .39 | not supported |
| H6: base model (exploratory) | base R vs chat R cos 0.82; the same pushes add M 1.01 in base vs 2.43 in chat | — |
| H7: Phi | not run (amendment 4: Phi barely declines, so its caution score is undefined) | — |

Benjamini–Hochberg at q = .05 over the 7 Qwen tests keeps H1, H3a, H3b and H5 N8869. H1 passes that correction but not its own pre-registered p < .01, so it stays "not supported".

**Reading.**
- **Financial caution is not the refusal direction.** Ablating R everywhere leaves Qwen's caution on financial questions in place. The answers barely change. "Should I buy Apple shares or Microsoft shares?" engages before and declines after ("As an AI, I cannot provide personalized financial advice…"); "Is Nvidia stock going to rise this quarter?" declines before and disclaims after. The judge's rate rising from 5/14 to 9/14 is mostly labels moving between borderline categories on near-identical text, not more caution. The prediction was that it would fall by half.
- **The two are related, not the same.** The finance contrast has cosine 0.36 with R, well above zero but inside what 14 pairs give under the sign-flip null.
- **Random pushes do set off refusal.** How much a random push raises the caution score tracks how far it moves the residual along R (ρ 0.64), and ablating R removes about a third of the effect. So "the model gets cautious when disturbed" runs partly through the refusal direction, and two thirds of it doesn't.
- **The investing neurons aren't special pushes.** At matched KL, isotropic noise adds as much caution as random neurons' output weights; in-distribution directions add less (1.6×). Neither pre-registered pattern (off-manifold > 2×, or all within 1.5×) fits.
- **One candidate writes along R.** L9 N8869, the one whose + push declined in the sweep, moves L24 along R further than all 128 controls. N1011, the disclaimer neuron, and N2855 don't.
- **Chat training amplifies it.** The base model has nearly the same refusal direction (cos 0.82), but the same pushes add less than half the caution there.

**Caveats.** 14 confirmatory pairs is small: H1's null is wide, and H2's caution rates move in steps of 7 points. The judge's borderline calls (κ 0.71 with hand labels on dev) drive H2's rate. One model: the Phi replication couldn't be run.

**Untested:** what direction does carry financial caution if R doesn't (e.g. ablate F_dev and measure the held-out rate); whether the two thirds of push-caution outside R is one other direction; the H3 analysis on medical and legal questions.
