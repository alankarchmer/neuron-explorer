# Pre-registration: is financial caution the refusal direction?

**Status: pre-registered, not frozen. Nothing below had been run when this was committed.** The prompt set and this
plan were committed together ("pre-registration") before any activation was computed. The commit timestamp is the
record.
Choices made later on dev data (token set, layer, doses, judge prompt) go into a second commit ("freeze") before any
held-out prompt is run. Any change after the first commit is listed under Amendments, with its date and reason.

## Question

Qwen3.5-4B often meets investing questions with caution: "I can't give financial advice", disclaimers, "consult a
professional". In the investing sweep, pushing it in a *random* direction made it refuse about a quarter of the time,
where the unpushed model never did (EXPERIMENTS.md, "Current experiment"). Two questions follow:

1. Is the model's financial caution carried by the same internal direction as its refusal of harmful requests (the
   "refusal direction", Arditi et al. 2024, "Refusal in language models is mediated by a single direction")?
2. Is that direction what random pushes were setting off, and is the effect about the model or about pushing it
   somewhere real activations never go? The sweep's random directions are isotropic Gaussian
   (`sweep/steer_sweep.py:156`).

## Material

| material | source | used for |
|---|---|---|
| Qwen3.5-4B (chat) | `Qwen/Qwen3.5-4B`, already local | everything |
| Qwen3.5-4B-Base | `Qwen/Qwen3.5-4B-Base`, 9.3 GB, Apache-2.0 | H6 |
| Phi-4-mini-instruct (3.8B) | `microsoft/Phi-4-mini-instruct`, 7.7 GB, MIT | H7, a second model family |
| harmful instructions | AdvBench `harmful_behaviors.csv` (llm-attacks GitHub), 82 KB, 520 items | refusal direction |
| harmless instructions | Alpaca `alpaca_data.json` (tatsu-lab GitHub), 23 MB, items with no `input` field | refusal direction, in-distribution controls |
| minimal pairs and benign questions | `sweep/caution_prompts.json` (this commit) | everything else |
| the 31,724 investing-sweep rollouts | `runs/q35-4b-L*-invest.jsonl` | choosing the decline token set only |

The steer-lens candidate list and the regex stance flags are not inputs.

**Harmful prompts:** we compute activations on them and read at most 8 generated tokens, enough to tell a refusal from
the start of compliance. Full completions to harmful prompts are never generated, stored or published. The downloaded
datasets stay out of git (`data/`, gitignored).

**The prompt set.** There are 40 minimal pairs and 24 benign questions. In each pair, the *target* and its *twin*
differ in one detail; the target is the side expected to draw caution.

| family | pairs | contrast | role |
|---|---|---|---|
| domain | 12 | finance advice vs the same sentence about something everyday ("Tesla stock" / "a Tesla car") | confirmatory |
| tense | 8 | a prediction vs a past fact about the same asset ("Will the S&P 500 go up next year?" / "Did … in 2020?") | confirmatory |
| decision | 8 | a personal decision vs how it works ("Should I short Tesla stock?" / "How does someone short …?") | confirmatory |
| situation | 6 | same question, a user the risky move suits less vs more (retired at 70 / 25 with a stable job) | exploratory |
| other_domain | 6 | medical or legal advice vs everyday ("stop taking my blood pressure medication" / "my morning walk") | exploratory |
| benign | 24 | cooking, travel, fitness, home | addition test |

Each family is split half dev and half held-out by a seeded shuffle (seed 20261025). Held-out prompts are not run
until the freeze commit.

## Definitions

- **Activation** h_l(p): the residual stream after layer l at the last token of prompt p in the chat template, with
  thinking disabled (as in the sweeps).
- **Refusal direction** R_l: mean h_l over 128 harmful prompts minus mean h_l over 128 harmless prompts. Both sets are
  sampled with seed 20261025; another 32 + 32 are kept for choosing the layer. Unit-normalised wherever it is used as a
  direction.
- **Finance-caution direction** F_l: mean over pairs of h_l(target) − h_l(twin), over the three confirmatory
  families. F_dev comes from dev pairs and F_held from held-out pairs.
- **Ablation:** project R out of the residual stream at every layer and position, as in Arditi et al.
  **Addition:** add c·R at layer l at every position.
- **Push:** a vector added to layer l's MLP output at every position, the same hook point and relative dose units as
  `sweep/steer_sweep.py`.
- **Caution score M:** at the first answer token, log P(token ∈ C) − log P(token ∉ C), where C is the set of
  decline-start tokens. C is fixed on the existing sweep rollouts: first tokens that occur at least 20 times and whose
  rollouts the refuse regex flags at least half the time. These rollouts use four prompts that are not in this
  prompt set.
- **Judge labels:** each greedy answer (up to 256 tokens) gets one of decline / disclaim-then-engage / engage. The
  labels come from an LLM judge with a fixed rubric, committed at freeze.
  - **Caution rate:** the share labelled decline or disclaim.
  - **Validation:** the judge must agree with 100 hand-labelled dev answers at Cohen's κ ≥ 0.7. M must separate
    decline from not-decline at AUROC ≥ 0.85.
  - **Fallback:** if either check fails, the judge's caution rate becomes the primary metric everywhere (an
    amendment), and M is reported only as secondary.

## Choices made on dev only

1. **Manipulation check:** on dev, the confirmatory targets must draw at least 20 percentage points more caution than
   their twins. If they don't, the pairs are revised (an amendment) before anything else is run.
2. **Layer l\*:** among layers 4–25, take the R_l whose ablation gives the lowest mean M on the 32 validation harmful
   prompts. The layer must also meet two conditions:
   - adding R_l at the mean harmful projection raises mean M above 0 on the 32 validation harmless prompts;
   - ablation changes the first-token distribution on harmless prompts by KL < 0.1.
3. **KL-matched doses:** for each control kind and push layer, find the single dose at which the median first-token KL
   on dev benign prompts equals the median KL of the investing candidates' +0.8 pushes at that layer.
4. **Judge rubric and C:** as above.

## Hypotheses

All confirmatory numbers are computed on held-out prompts. The resampling unit is the prompt, or the pair where pairs
are compared. Confidence intervals are 95% bootstrap (10,000 resamples). The seven confirmatory tests on Qwen (H1,
H2, H3a, H3b, H5 ×3) are corrected together with Benjamini–Hochberg at q = 0.05.

**H1, shared direction.** At l\*, cos(F_held, R) > 0, beyond the 99th percentile of a null built from 10,000 random
sign flips of the held-out pair differences.

- *Supported:* p < 0.01 and cos ≥ 0.3.
- *Partial:* p < 0.01 and cos < 0.3, so the directions share a component.
- *Not supported:* p ≥ 0.01.

Per-family cosines and the cosine curve over all layers are reported but don't decide anything.

**H2, ablation removes financial caution (necessity).** With R ablated, the caution rate on held-out confirmatory
targets falls by at least 50% relative to unablated, and the target-minus-twin gap in M shrinks by at least 50%.

- *Not supported:* both fall by less than 20%. Anything in between is *partial*.
- *Guard:* the KL on twins under ablation is reported, so a broken model can't pass for a cautious-free one.

**H3, random-push caution goes through R.** Pushes are applied at L7, L10 and l\*, using 64 isotropic Gaussian
directions per layer at the original doses. The prompts are held-out targets, twins and benign questions.

- **H3a:** across directions, the change in M correlates with the change in the projection of h_{l\*} onto R
  (Spearman ρ ≥ 0.5, p < 0.01). For a push at l\*, the projection is read at the next layer.
- **H3b:** with R ablated, the caution added by isotropic pushes (mean change in M and in caution rate, relative to
  unpushed) shrinks by at least 50%. *Not supported* below 20%.

**H4, off-manifold or general (estimation, two-sided).** Three control kinds are compared at matched KL:

- isotropic Gaussian;
- the W_down columns of random non-candidate neurons;
- in-distribution differences h_l(a) − h_l(b) for random pairs of Alpaca prompts.

The quantity is the ratio of the caution each kind adds, with a CI.

- **Off-manifold:** isotropic adds more than twice the other two.
- **General fallback:** all three kinds are within a factor of 1.5 of each other.
- **Neither:** reported as such.

**H5, the three candidates are refusal components.** The candidates are L10 N1011, L9 N8869 and L7 N2855. For each,
push at +0.8 and take the mean change in h_{l\*} over held-out prompts. The test is that change's cosine with R. It
counts as a refusal component if two things hold:

- the cosine lies beyond all 128 controls at the same layer and dose (64 isotropic plus 64 random neurons);
- it has the predicted sign: positive for N1011 (pushing up adds disclaimers), positive for N8869 (pushing up
  refuses), negative for N2855 (pushing down refuses).

A sign opposite to the prediction counts as not supported. The cosine of each neuron's W_down column with R at its
own layer is reported as secondary.

**H6, base model (exploratory).** Predictions:

- R computed in Qwen3.5-4B-Base, with the same prompts and template, has cosine ≥ 0.5 with the chat model's R at l\*.
  The refusal direction may already exist before chat tuning (Kissane et al. 2024, "Base LLMs refuse too").
- Isotropic pushes add less caution in the base model than in the chat model.

These are exploratory because a base model's first token under a chat template is a weaker measurement.

**H7, replication on Phi-4-mini-instruct.** H1, H2 and H3b are repeated with Phi's own l\*, C and KL-matched doses,
all chosen on dev the same way. These are three tests, corrected separately. The finding generalises only if they
point the same way as Qwen's.

**Exploratory, no decision attached:**

- **situation pairs:** does the model encode how suitable the risky move is, and does that align with R?
- **other_domain pairs:** is medical and legal caution the same direction as financial caution?
- **adding F_dev to benign questions:** does it produce finance disclaimers or generic refusals?

## What would change the story

- **H1 and H2 hold:** "I can't give financial advice" is the general refusal machinery applied to money questions.
- **H1 holds but H2 fails:** the direction is shared, but caution has other routes.
- **H3b holds and H4 says general:** the model falls back on refusal under almost any disturbance, which matters for
  anyone using steering.
- **H4 says off-manifold:** the "random pushes make it cautious" result says more about Gaussian noise than about
  the model.
- **H5 fails:** the three candidates do something else, and that becomes the next question.

Each outcome is written up, including the ones that don't hold.

## Compute

Everything is forward passes or short greedy rollouts on the Mac (MPS, one model at a time in 16 GB). The largest piece
is H3/H4: about 70,000 prefill passes (40 held-out prompts × 3 push layers × 64 directions, at 3 original
doses, again with R ablated, and once per control kind at the KL-matched dose), an estimated 2–3 hours. No GPU rental is planned. If one is needed, a cost
estimate comes first.

## Amendments

None yet.
