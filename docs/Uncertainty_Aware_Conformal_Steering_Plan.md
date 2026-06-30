# Uncertainty‑Aware Conditional Activation Steering via Conformal Prediction

**A research plan / feasibility study**
Prepared as an advisor's critical review (target venue: ICASSP / a NeurIPS‑ICLR trustworthy‑ML or interpretability workshop).
Date: 2026‑06‑29.

---

## 0. One‑paragraph summary

CAST (Conditional Activation Steering, Lee et al., ICLR 2025) gates a steering vector with a **hard binary step** on a similarity score: $f(s)=\mathbb{1}[s>\theta]$. Your proposal is to replace that step with a **continuous, uncertainty‑aware gate** $g(\cdot)\in[0,1]$, calibrated with **conformal prediction**. The idea is sound, the math is clean, and it is genuinely a low‑hanging fruit — **but only if you reframe the contribution.** "Make it continuous" is, by itself, partly taken (Dynamic Activation Composition, ACT, EAST all modulate steering strength continuously). The **defensible, novel core** is *distribution‑free calibration and guarantees on the gate* (a provable bound on over‑refusal / false‑steering), plus a **principled e‑value calculus for CAST's logical composition** of conditions. Framed that way, this is a credible 4‑page paper with a real theorem and a small, tractable experimental footprint that fits your hardware.

**Verdict: FEASIBLE and worth doing. Pursue it, but lead with *guarantees + calibration + composition*, not with "soft vs. hard."**

---

## 1. What CAST actually does (so we modify exactly the right object)

### 1.1 The steering update

Standard activation addition (ActAdd; Turner et al., 2023):

$$\mathbf{h}' \leftarrow \mathbf{h} + \alpha\,\mathbf{v}$$

applied to **every** input — which is why a refusal vector raises refusal *indiscriminately* (Arditi et al., 2024).

CAST adds a **condition gate** (paper §3.1):

$$\boxed{\;\mathbf{h}' \leftarrow \mathbf{h} + f\!\big(\mathrm{sim}(\mathbf{h},\mathrm{proj}_{\mathbf c}\mathbf{h})\big)\cdot \alpha\,\mathbf{v}\;}$$

where $\mathbf c$ is the **condition vector**, $\mathbf v$ the **behavior vector**, $\mathrm{sim}(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$, and the projection $\mathrm{proj}_{\mathbf c}\mathbf{h}$ collapses $\mathbf h$ onto the condition direction. Define the scalar **condition score**

$$s(\mathbf h) \;:=\; \mathrm{sim}(\mathbf h,\mathrm{proj}_{\mathbf c}\mathbf h)$$

— operationally this is the per‑layer cosine returned by the library's `get_condition_similarities()`. The gate is a **hard step** (paper §3.1):

$$f(s)=\begin{cases}1 & s>\theta\\[2pt]0 & \text{otherwise.}\end{cases}$$

### 1.2 Three facts about CAST that decide our design

1. **The condition is checked once, on the prompt, not per generated token** (Appendix A.2 / Fig. 11): the check runs in the first forward pass (prompt caching), and if met, $\mathbf v$ is added to *all* subsequent token passes. ⇒ **Our gate can be computed once per prompt** — no per‑token instability to worry about for the core method.
2. **$\theta$, the layer $l$, and the comparison direction ($>$ or $<$) are chosen by a grid search that maximizes F1** on a labeled $\{\mathcal D^+,\mathcal D^-\}$ split (Appendix C.2, `find_best_condition_point`). ⇒ **The threshold has no statistical guarantee** — it is a point estimate that maximizes F1 on one split, with no control on the false‑steer (over‑refusal) rate at test time. *This is the hole.*
3. The authors **explicitly flag the modulation property as unexplored**: "the threshold $\theta$ [modulates] the range of hidden states triggering the condition … this property allows us to adjust the model's sensitivity … **we do not explore it further in this study**" (§4, "Property: Modulation"). ⇒ **They told us where the open door is.**

The exact code surface we will touch (Appendix C.1–C.2):

```python
malleable_model.steer(
    behavior_vector=..., behavior_layer_ids=[10,11,12,13,14,15], behavior_vector_strength=0.1,
    condition_vector=..., condition_layer_ids=[9],
    condition_vector_threshold=0.031,                 # <-- the θ we will conformalize
    condition_comparator_threshold_is="smaller",      # <-- the comparison direction
)
# check_condition(sim_dict, layer_combo, threshold, direction):  <-- the HARD step we replace
```

---

## 2. The idea, formalized

Replace the binary $f$ by a **calibrated gate** $g:\mathbb R\to[0,1]$ so the update becomes

$$\boxed{\;\mathbf h' \leftarrow \mathbf h + g\big(s(\mathbf h)\big)\cdot \alpha\,\mathbf v\;}$$

$g$ must (i) be continuous, (ii) reflect *calibrated uncertainty* about whether the condition holds, and (iii) come with a **distribution‑free guarantee** on how often / how strongly we steer inputs that do **not** meet the condition (the over‑refusal side).

This is the same move KnowNo (Ren et al., 2023) makes for robot planners — *use conformal prediction to decide an action from calibrated uncertainty* — except the "action" is **how hard to steer** instead of "ask a human for help."

> **Be precise about *which* uncertainty.** There are two:
> - **(A) Condition‑membership uncertainty** — "is this prompt in the condition class (e.g., harmful)?" This is a 1‑D detection problem on $s(\mathbf h)$. **This is the clean, low‑hanging target.**
> - **(B) Answer‑correctness uncertainty** — "is the model's *answer* unreliable, so we should steer toward caution / deliberation / abstention?" This is the "reasoning" framing (closer to conformal factuality, Mohri & Hashimoto 2024). **More ambitious, larger scope — keep as the extension/vision (§7), not the ICASSP core.**

Everything below is Framing (A) unless stated.

---

## 3. The math (the core of the proposal)

### 3.1 The condition check is a one‑dimensional detector

Given the fixed condition vector, $s(\mathbf h)\in\mathbb R$ is a scalar score and CAST's $f=\mathbb 1[s>\theta]$ is a threshold classifier. So the *entire* design space of "what gate to use" is the design space of **calibrated one‑dimensional detection** — which is exactly where conformal prediction is strongest and cheapest. We are not inventing new representations; we are calibrating CAST's existing score. (Corollary, and an honesty point: **conformal prediction calibrates a score, it does not improve the score's discriminative power.** The discriminative ceiling is CAST's PCA direction.)

### 3.2 Split‑conformal calibration of the gate

Take a **calibration set** $\mathcal C=\{\mathbf h_i\}_{i=1}^{n}$ of hidden states from the **condition‑NOT‑met** class $P_-$ (e.g., harmless prompts), with scores $s_i=s(\mathbf h_i)$. For a test prompt with score $s(\mathbf h)$, define the **conformal $p$‑value**

$$\hat p(\mathbf h)\;=\;\frac{1+\big|\{\,i: s_i \ge s(\mathbf h)\,\}\big|}{n+1}.$$

**Lemma (super‑uniformity).** If $\{\mathbf h_1,\dots,\mathbf h_n,\mathbf h\}$ are exchangeable (all $\sim P_-$), then $\Pr\big(\hat p(\mathbf h)\le\tau\big)\le\tau$ for all $\tau\in[0,1]$ (exact equality at the grid points $\tau=k/(n+1)$ for continuous scores). *Proof:* rank of $s(\mathbf h)$ among $n{+}1$ exchangeable scores is uniform on $\{1,\dots,n{+}1\}$; $\hat p$ is a deterministic decreasing function of that rank. ∎

Interpretation: a **small $\hat p$ means the prompt's condition score is atypically high relative to the harmless class** — i.e., calibrated evidence that the condition *is* met (the prompt is harmful). This is exactly the quantity we want to gate on.

### 3.3 The hard, *calibrated* gate (drop‑in replacement for grid‑search‑$\theta$)

$$g_{\text{hard}}(\mathbf h)=\mathbb 1\big[\hat p(\mathbf h)\le\alpha\big]\quad\Longleftrightarrow\quad s(\mathbf h)\;\ge\; \theta_\alpha,\;\; \theta_\alpha = s_{(\lceil(1-\alpha)(n+1)\rceil)}\ \text{(empirical quantile)}.$$

**Guarantee (false‑steer control).** $\Pr_{\mathbf h\sim P_-}\!\big(\text{we steer}\big)=\Pr(\hat p\le\alpha)\le\alpha.$
**This is the thing CAST's F1 grid search cannot promise:** a user sets $\alpha=$ "≤5% of harmless prompts may be steered" and *gets it*, in finite samples, distribution‑free. The threshold is now a calibrated quantile, not an F1 point estimate.

### 3.4 The soft, continuous gate — and a closed‑form leakage bound

Now make it continuous. Let $\phi:[0,1]\to[0,1]$ be any **non‑increasing** map and set

$$g(\mathbf h)=\phi\big(\hat p(\mathbf h)\big).$$

Useful choices:

| Name | $\phi(p)$ | shape |
|---|---|---|
| Conformal ramp | $\max\!\big(0,\,1-p/\alpha\big)$ | linear from 1 at $p{=}0$ to 0 at $p{=}\alpha$ |
| Power decay | $(1-p)^{\gamma}$ | smooth, full support |
| Calibrated sigmoid | $\sigma\!\big((\tau_0-\hat p)/\eta\big)$ | soft, two knobs |

**Theorem (distribution‑free leakage bound).** For $\mathbf h\sim P_-$ (the null), if $\hat p$ is super‑uniform and $\phi$ non‑increasing,

$$\mathbb E_{P_-}\big[g(\mathbf h)\big]\;=\;\mathbb E\big[\phi(\hat p)\big]\;\le\;\int_0^1\phi(u)\,du .$$

*Proof.* By the layer‑cake identity $\mathbb E[\phi(\hat p)]=\int_0^\infty\Pr(\phi(\hat p)>t)\,dt$. As $\phi$ is non‑increasing, $\{\phi(\hat p)>t\}=\{\hat p<\phi^{-1}(t)\}$, and super‑uniformity gives $\Pr(\hat p<\phi^{-1}(t))\le \phi^{-1}(t)$. For $U\sim\mathrm{Unif}(0,1)$ the same steps hold with equality, and $\int_0^\infty\phi^{-1}(t)\,dt=\int_0^1\phi(u)\,du$. ∎

**Consequences (this is the paper's headline equation):**
- **Conformal ramp:** $\displaystyle \mathbb E_{P_-}[g]\le \int_0^{\alpha}\!\Big(1-\tfrac{u}{\alpha}\Big)du=\frac{\alpha}{2}.$ The soft gate leaks **half** the expected steering mass onto harmless prompts compared to the hard CP gate ($\mathbb E[g]\le\alpha$), *while being continuous*. You also still have support control $\Pr_{P_-}(g>0)=\Pr(\hat p<\alpha)\le\alpha$.
- So you can advertise **two distribution‑free dials**: probability of *any* steering on a benign prompt ($\le\alpha$) and *expected* steering strength on a benign prompt ($\le\alpha/2$). Both hold with no distributional assumptions beyond exchangeability.

**The other side (power) is empirical, by design.** On the condition‑met class $P_+$, $\hat p$ is computed against the *harmless* calibration distribution, so harmful prompts get small $\hat p$ and $g\to 1$. The achieved $\mathbb E_{P_+}[g]$ ("how hard do we actually steer the things we should") depends on class separation and is measured — this is precisely an **ROC/detection trade‑off**: the continuous gate traces the *whole* curve, where CAST reports a single operating point. (Recall Table 3: Hermes‑2‑Pro lands at 83% harmful‑refusal / 2.4% harmless‑refusal — one point. We get the curve and a guaranteed knob along it.)

### 3.5 Generalization: risk control for the over‑refusal *rate*

The $p$‑value gate controls the *marginal* false‑steer event. If you want to control an **expected loss** (e.g., the over‑refusal *rate* under a soft gate, or a composite "harm‑weighted" risk), use **Conformal Risk Control** (Angelopoulos et al., ICLR 2024) or **Learn‑Then‑Test** (Angelopoulos et al., 2021). Let $\lambda$ be an aggressiveness knob (e.g., scales $\phi$, or shifts the ramp), and define a bounded, monotone risk $R(\lambda)=\mathbb E[\ell_{\text{over‑refuse}}(\lambda)]$. CRC picks $\hat\lambda$ so that $\mathbb E[R(\hat\lambda)]\le\alpha$ in finite samples; LTT casts threshold selection as multiple testing and yields high‑probability control with **multiple knobs** (layer, $\theta$, strength) selected jointly with validity. This is the rigorous home for "choose the steering configuration that provably keeps over‑refusal $\le\alpha$" — and it directly upgrades CAST's F1 grid search. **Use this for threshold/knob selection; use §3.4 for the per‑prompt gate.**

### 3.6 Composition: an e‑value calculus for CAST's logical conditions

CAST composes conditions with hand‑set ANDs/ORs of thresholds (`rules=["if C1 then B1", ...]`, multi‑conditioning $f=\mathbb 1[s_{\text{adult}}>\theta_1 \text{ or } s_{\text{stereo}}>\theta_2]$). $p$‑values do **not** combine cleanly under dependence, so naïvely AND/OR‑ing calibrated $p$‑values breaks the guarantee. **E‑values do combine** (Vovk & Wang, 2021):

For each condition $j$, convert its conformal $p$‑value to an **e‑value** with a valid calibrator $f_\kappa(p)=\kappa\,p^{\kappa-1}$, $\kappa\in(0,1)$ (note $1/p$ is *not* valid — its integral diverges):

$$E_j(\mathbf h)=f_\kappa\big(\hat p_j(\mathbf h)\big)\ge 0,\qquad \mathbb E_{P_-^{(j)}}[E_j]\le 1 .$$

Then:
- **OR (steer if any condition fires):** $E_\vee=\frac1m\sum_j E_j$ is an e‑value under **arbitrary dependence** between conditions — exactly the robust, assumption‑light merge you want when conditions overlap (which CAST notes they do).
- **AND (steer only if all fire):** $E_\wedge=\prod_j E_j$ is an e‑value under independence/sequentiality.
- **Gate + guarantee:** $g=\min(1,\,E/\beta)$ with Markov's inequality giving $\Pr_{\text{null}}(E\ge\beta)\le 1/\beta$, i.e. **false‑steer $\le 1/\beta$ for composed conditions**.

This turns CAST's heuristic logic gates into a **guarantee‑preserving algebra** — a clean, genuinely novel, mathematically deep contribution that a workshop/ICASSP reviewer in stats‑ML will like.

### 3.7 Honest caveats baked into the math

- **Exchangeability is the load‑bearing assumption.** Coverage holds *marginally over the calibration distribution*. Real deployment prompts are shifted ⇒ the guarantee degrades. Mitigations to state and (optionally) implement: **Mondrian / group‑conditional CP** (stratify calibration by harm category ⇒ per‑category coverage, fixing the "aggregate coverage hides a category failure" problem) and **weighted CP** for covariate shift. Be explicit that the guarantee is conditional on the calibration source.
- **Coverage ≠ usefulness.** A degenerate gate $g\equiv 0$ trivially satisfies the leakage bound. The bound constrains the *null* side; the *power* side ($\mathbb E_{P_+}[g]$) must be reported alongside, always. The contribution is the *pair* (guaranteed null leakage, measured power), i.e., a calibrated operating curve.

---

## 4. Is it actually a good idea? (critical evaluation — not cheerleading)

**Where the win is real:**
1. **Guarantees CAST lacks.** A user‑set, finite‑sample bound on over‑refusal is a concrete, defensible deliverable. Over‑refusal is a live, named problem (XSTest, OR‑Bench) and "exaggerated safety" is exactly the failure CAST's harmless‑refusal column exposes.
2. **Robustness to threshold misspecification.** CAST's $\theta$ is brittle and model‑specific (paper: Hermes wants $\theta\in[0,0.1]$, Zephyr $[0.4,0.6]$). A conformal quantile *auto‑sets* the operating point to a target rate per model — no per‑model grid magic. This alone is a practical selling point.
3. **Graceful degradation near the boundary.** Hard on/off at $\theta$ means a 1‑nat wiggle in $s$ flips full‑strength steering. A ramp gives fractional steering in the ambiguous band — plausibly *gentler on fluency* (ActAdd/CAST both warn strong steering hurts coherence). Empirical, but a real hypothesis.
4. **Principled composition** (§3.6) — the strongest novelty, and the part least likely to be "scooped," because nobody has put e‑values on steering conditions.

**Where I'd push back on you (and where reviewers will):**
1. **"Continuous strength" is not novel by itself.** Dynamic Activation Composition (Scalena et al., 2024) already adapts steering intensity per step via KL/information gain; ACT (Wang et al., 2024) adapts intensity for truthfulness; EAST (Rahn et al., 2024) uses entropy to *build* a steering vector. **If your pitch is "soft instead of hard," a reviewer rejects on novelty.** Your differentiator must be **calibration + distribution‑free guarantee + composition**, not softness. Those three are absent from all prior steering work.
2. **The accuracy gain may be small.** CAST's own "saturation" result and the fairly separable t‑SNE (Fig. 4) suggest the harmful/harmless decision is easy in the regime they test. So a soft gate may not move raw F1 much. **Do not sell raw accuracy.** Sell *calibration validity* (achieved vs. target $\alpha$), *robustness*, and *the operating curve*. The gain concentrates in (a) **ambiguous/borderline prompts** (XSTest is built from exactly these — benign prompts with unsafe keywords), (b) **misspecified‑threshold regimes**, and (c) **multi‑condition composition**. Design experiments to hit those.
3. **The score is fixed.** You are calibrating CAST's PCA direction; if that direction is a weak detector for a subtle condition, CP gives you a *valid but wide/weak* gate. Report AUROC of $s$ first; if it's near 1 the soft gate has little headroom (good for the guarantee story, bad for the accuracy story).
4. **The "reasoning" connection is indirect for Framing (A).** Condition‑membership uncertainty is *not* answer‑correctness uncertainty. If you want the reasoning narrative, you must do Framing (B) (§7), which is a bigger lift. For ICASSP, I'd keep (A) and state (B) as future work, honestly.

**Net:** the idea clears the bar **as a calibration/guarantees paper with a composition twist**, on a small experimental budget. It does **not** clear the bar as a "new steering method that's more accurate." Choose the former.

---

## 5. Positioning vs. prior art (memorize this table — it is your novelty defense)

| Work | Modulates strength? | Uses uncertainty? | **Calibrated / guarantee?** | Gates the *condition*? | Composition calculus? |
|---|---|---|---|---|---|
| CAST (Lee 2025) | no (binary) | no | no (F1 grid) | **yes** (hard) | heuristic AND/OR |
| Dyn. Act. Composition (Scalena 2024) | yes (KL/step) | info‑gain | no | no (modulates $\alpha$) | per‑property, heuristic |
| ACT (Wang 2024) | yes (adaptive) | no | no | no | no |
| EAST (Rahn 2024) | — | entropy (to *build* vec) | no | no | no |
| KnowNo (Ren 2023) | n/a (robots) | **yes** | **yes (CP)** | gates "ask for help" | no |
| **This proposal** | yes (continuous) | **yes (conformal)** | **yes (CP/CRC/LTT)** | **yes (soft, gated condition)** | **yes (e‑values)** |

The empty cells in the bottom row vs. the steering rows are the paper.

---

## 6. Code implementation plan

Built on **(i)** CAST's released library `IBM/activation-steering` (`MalleableModel.steer`, `find_best_condition_point`, `check_condition`, `get_condition_similarities`), **(ii)** a conformal toolkit (`TorchCP`, arXiv 2402.12683, or Angelopoulos's `conformal-prediction` repo), and **(iii)** CAST's eval harness (`protectai/distilroberta-base-rejection-v1` + refusal keyword list, Appendix D.2). Your local `ICV/utils/forward_tracer.py` and `ICV/utils/llm_layers.py` already give you hook plumbing if you prefer not to depend on their library.

The change is small and surgical. **Three pieces:**

**(1) Calibration — build the empirical null distribution of the score (offline, once).**
```python
def calibrate_conformal_gate(model, condition_vector, cond_layers, harmless_prompts):
    # reuse CAST's get_condition_similarities() on a held-out HARMLESS set
    cal_scores = []                      # s_i for the condition-NOT-met class P_-
    for p in harmless_prompts:
        sim = model.get_condition_similarities(p, condition_vector, cond_layers)  # dict layer->cos
        cal_scores.append(aggregate(sim, cond_layers))   # e.g. max or mean over cond layers
    cal_scores = np.sort(np.asarray(cal_scores))         # sorted null scores
    return cal_scores                                    # n values -> empirical CDF
```

**(2) The gate — conformal p-value + soft ramp (per prompt, matching CAST's once-on-prompt check).**
```python
def conformal_gate(s, cal_scores, alpha=0.05, mode="ramp"):
    n = len(cal_scores)
    p = (1 + np.sum(cal_scores >= s)) / (n + 1)          # conformal p-value  (Lemma, 3.2)
    if mode == "hard":   return float(p <= alpha)        # guarantee: false-steer <= alpha
    if mode == "ramp":   return max(0.0, 1.0 - p/alpha)  # E[g|harmless] <= alpha/2  (Thm 3.4)
    if mode == "power":  return (1.0 - p)**GAMMA
```

**(3) Inject into the steering hook — scale the behavior vector by `g` instead of by `1[condition]`.**
```python
# CAST: h' = h + 1[s>theta] * strength * v
# Ours: h' = h + g(s)        * strength * v
g = conformal_gate(score_for_prompt, cal_scores, alpha=ALPHA, mode="ramp")
model.steer(behavior_vector=v, behavior_layer_ids=B_LAYERS,
            behavior_vector_strength = g * STRENGTH,     # <-- the only functional change
            condition_vector=None)                       # gating now done by g, not the step
```

**(4) Composition (optional, the strong-novelty module) — e-values.**
```python
def p_to_e(p, kappa=0.5):            # valid calibrator f_kappa(p)=kappa*p^(kappa-1)  (Vovk-Wang)
    return kappa * p**(kappa - 1.0)
def compose(ps, logic="OR"):         # ps: per-condition conformal p-values
    es = [p_to_e(p) for p in ps]
    E  = np.mean(es) if logic == "OR" else np.prod(es)   # OR: any-dependence; AND: independent
    return E
g = min(1.0, E / BETA)               # false-steer for composed condition <= 1/BETA  (Markov)
```

**(5) Threshold/knob selection via risk control (replaces the F1 grid search).**
Wrap CAST's `find_best_condition_point` so that instead of `argmax F1`, you select the most aggressive $(\theta,\text{layer},\text{strength})$ whose **upper confidence bound on over-refusal $\le\alpha$** (Learn-Then-Test / Conformal Risk Control). ~30 lines around their existing grid.

**Effort estimate:** the core (1)–(3) is ~1 day on top of a working CAST install. (4) and (5) are ~1–2 days each. This is genuinely a low-hanging fruit *at the code level*.

---

## 7. Experimental plan

**Models (matched to CAST *and* to your 3×RTX 3080 10 GB):** `Qwen/Qwen1.5-1.8B-Chat` (fits in fp16), `h2oai/h2o-danube3-4b-chat`, and `NousResearch/Hermes-2-Pro-Llama-3-8B` in 8-bit (your `ICV/inference.sh` already runs a 7B in 8-bit). Start with the 1.8B for fast iteration.

**Data / splits (3-way):** *extract* vectors (train) → *calibrate* conformal (held-out, **harmless only** for the null) → *test*.
- Harmful (condition-met): **Sorry-Bench** (CAST's source).
- Harmless (condition-not-met / null): **Alpaca**.
- **Over-refusal stress (the headline eval): XSTest** (250 safe prompts with unsafe-looking keywords + 200 unsafe) and **OR-Bench** (Cui et al., 2024). This is where a calibrated gate should beat a hard threshold.
- Multi-condition: CAST's 5 categories (sexual / legal / hate / crime / health) for the **e-value composition** experiment.

**Metrics:**
1. **Calibration validity:** achieved vs. target over-refusal $\alpha$ across $\alpha\in\{0.01,\dots,0.2\}$ — the "does the guarantee hold" plot (should track the diagonal). *This is the money figure.*
2. **Operating curve:** harmful-refusal (power/TPR) vs. harmless-refusal (FPR); report AUROC of the gate. Compare CAST's single point to your curve.
3. **Fluency:** perplexity / a quality judge on accepted (non-refused) generations — test the "gentler steering" hypothesis (§4.3).
4. **Robustness:** sweep a *misspecified* $\theta$ and show conformal auto-calibration is stable where CAST degrades.
5. **Distribution-shift stress:** calibrate on Alpaca, test on XSTest; report coverage gap; then show Mondrian/weighted CP narrows it.
6. **Composition:** multi-condition refusal accuracy + guarantee, e-values vs. CAST's heuristic AND/OR.

**Baselines:** CAST (hard F1 $\theta$); naïve **uncalibrated** sigmoid gate (ablation isolating the value of *calibration* vs. mere *softness*); Dynamic Activation Composition.

**Ablations:** choice of $\phi$ (hard / ramp / power / sigmoid); calibration size $n$; score aggregation over layers (max vs. mean); per-prompt vs. per-token gate; marginal vs. Mondrian CP.

**Minimum publishable unit (if time is short):** one model (Qwen-1.8B) + Sorry-Bench/Alpaca/XSTest + Figures (1)+(2) + the §3.4 theorem. That is already a coherent 4-page workshop paper.

---

## 8. Reading list (in order of importance, with *why*)

1. **Lee, Padhi, Natesan Ramamurthy, et al. — *Programming Refusal with Conditional Activation Steering* (CAST), ICLR 2025.** The method you extend. Master §3.1 ($f$), §4 "Modulation," and Appendix C (grid search, code API). *You are replacing exactly one function in their pipeline; know it cold.*
   `arXiv:2409.05907` · code: `github.com/IBM/activation-steering`
2. **Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ*, 2021.** Your CP toolbox: split conformal, super-uniform $p$-values, coverage, and pointers to conformal risk control. The §3.2–3.4 machinery lives here. `arXiv:2107.07511`
3. **Ren, Dixit, Bodrova, … Majumdar — *Robots That Ask For Help: Uncertainty Alignment for LLM Planners* (KnowNo), CoRL 2023.** The template you are porting: *use CP to gate an action from calibrated uncertainty.* Your "how hard to steer" = their "do I ask for help." Best single analogy for the framing. `arXiv:2307.01928`
4. **Vovk & Wang — *E-values: Calibration, Combination, and Applications*, Ann. Statist. 2021.** Foundations for §3.6: $p\to e$ calibrators, product/average merging, dependence-robust combination. This is what makes the composition contribution rigorous. `arXiv:1912.06116`
5. **Angelopoulos, Bates, Candès, Jordan, Lei — *Learn Then Test* (2021)** *and* **Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* (ICLR 2024).** Risk-control machinery for §3.5 — selecting $(\theta,\text{layer},\text{strength})$ with a finite-sample bound on over-refusal, replacing CAST's F1 grid. `arXiv:2110.01052`, `arXiv:2208.02814`
6. **Scalena, Sarti, Nissim — *Multi-property Steering with Dynamic Activation Composition*, BlackboxNLP 2024** *(read alongside Rahn et al., EAST, `arXiv:2406.00244`).* The closest prior art on *adaptive/continuous* steering strength. Read to (a) sharpen your novelty claim and (b) borrow their fluency-vs-strength evaluation. `arXiv:2406.17563`

*Extension reading for the "reasoning" vision (§9, Framing B), optional for ICASSP:* **Mohri & Hashimoto — *Language Models with Conformal Factuality Guarantees*, ICML 2024** (`arXiv:2402.10978`); **Quach et al. — *Conformal Language Modeling*, ICLR 2024** (`arXiv:2306.10193`); **Wang et al. — *ConU: Conformal Uncertainty in LLMs*, EMNLP Findings 2024**.

---

## 9. Suggested framing for the paper

**Working title:** *"Conformal Activation Steering: Calibrated, Guaranteed Conditional Control of LLMs."*

**The 4-page story:**
- **Hook:** conditional steering decides *when* to intervene with an uncalibrated hard threshold ⇒ no control on over-refusal, brittle across models.
- **Contribution 1 (method):** replace the step with a conformal gate; give the false-steer guarantee (§3.3) and the soft-gate leakage bound (§3.4, the theorem).
- **Contribution 2 (composition):** e-value calculus for logical conditions with a preserved guarantee (§3.6).
- **Contribution 3 (empirics):** calibration-validity + operating-curve + over-refusal (XSTest/OR-Bench) showing you hit the target rate where CAST cannot, at no fluency cost.
- **Framing (B) as the vision (1 paragraph):** the same gate, driven by *answer-uncertainty* (conformal factuality / abstention), steers a model toward deliberation/caution proportional to calibrated doubt — connecting calibrated steering to *reasoning reliability* and "knowing when you don't know." Don't over-claim it; flag it as the natural next step.

**Anticipated reviewer objections → your rebuttals:**
- *"Softness isn't new."* → Right; our contribution is **calibration + distribution-free guarantee + e-value composition**, none of which exist in Scalena/Wang/Rahn (Table, §5). Ablation vs. an *uncalibrated* sigmoid isolates this.
- *"Exchangeability won't hold in deployment."* → Acknowledged; we report the coverage gap under shift and show Mondrian/weighted CP recovers per-category validity (§3.7).
- *"Accuracy barely moves."* → The claim is *not* accuracy; it's *guaranteed, calibrated* control and robustness. We demonstrate validity, the full operating curve, and stability under threshold misspecification — the axes CAST cannot report.
- *"Why ICASSP?"* → It is a detection-and-calibration result on a 1-D score with a provable guarantee and a signal-processing-flavored ROC story; fits ML-for-signal-processing / trustworthy-ML tracks.

---

## 10. Bottom line

Pursue it. The math is clean and *yours to prove* (the §3.4 leakage theorem and the §3.6 e-value calculus are real, citable contributions, not repackaging). The code change is a few dozen lines on top of CAST. The experimental budget fits your GPUs. The one thing that will sink it is framing it as "continuous steering" — **don't.** Frame it as *calibrated, guaranteed, composable* conditional steering, evaluate on over-refusal benchmarks where calibration actually pays, and keep the reasoning/answer-uncertainty angle as a clearly-labeled extension.
