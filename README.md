# CURED: Curvature-Informed Routing and Entropy-Based Decoding

5-gate inference-time router selecting greedy / ALTA / CoVe / ITI per question using R², κ, ECR, entropy, and domain. No retraining. No retrieval.

---

## Results

**TruthfulQA** (cosine scoring, McNemar exact test)

| Model | Greedy | CURED v2 | Δ | p |
|---|---|---|---|---|
| Llama-3.2-3B | 50.1% | **60.6%**† | +10.5 pp | <0.0001 |
| Llama-3.1-8B | 49.6% | **60.2%** | +10.6 pp | <0.0001 |
| Qwen2.5-14B | 62.2% | 64.0% | +1.8 pp | ns |
| Qwen2.5-32B | 58.8% | 60.1% | +1.3 pp | ns |

† 3B gain uses 8B-calibrated profile (R²=0.582). Native 3B profile (R²=0.501) → 49.4%, no gain.

**MedHallu** (n=200, all ns)

| Model | Greedy | CURED v2 | Δ |
|---|---|---|---|
| Llama-3.2-3B | 55.0% | 52.5% | −2.5 pp |
| Llama-3.1-8B | 45.5% | 49.0% | +3.5 pp |

**Protocol ablations** (n=200, TruthfulQA)

| Scale | Greedy | ALTA | CoVe | ITI |
|---|---|---|---|---|
| 3B | 56.5% | 59.0% | 46.0% | 56.5% |
| 8B | 48.0% | 60.6% | 39.2% | 57.5% |
| 14B | 64.5% | 57.9% | 45.5% | 67.0% |
| 32B | 57.6% | 58.0% | 49.8% | 64.3% |

**Mechanistic profiles**

| Scale | R² | κ | ECR | H_final |
|---|---|---|---|---|
| 3B | 0.501 | 0.455 | 0.076 | 0.837 |
| 8B | 0.582 | 0.597 | 0.066 | 0.669 |
| 14B | 0.444 | 0.360 | 0.031 | 0.306 |
| 32B | 0.473 | 0.322 | 0.051 | 0.529 |

Scale-level R² vs ALTA gain: **r = 0.986, p = 0.014**. Per-question R²: r = 0.039, p = 0.58 (noise).

**FACTOR** (8B, n=200, letter scoring)

| Benchmark | Greedy | ALTA | CURED |
|---|---|---|---|
| FACTOR-News | 59.0% | 69.0% | 61.5% |
| FACTOR-Wiki | 29.0% | 64.0% | 43.0%* |

*Domain classifier bug: substring matching misclassifies 50% of questions as medical. Fix in progress.

---

## Router

Gate 0 low entropy (H < τ_H_easy=0.5) → greedy_confident
Scale profile_mean_r2 ≥ 0.55, non-medical, H > τ_H_easy → alta_global_viable
Gate 2 per-question R² > τ_R2=0.65 → alta_gate2
Gate 3 medical + ITI available → iti_medical
Gate 4 composite ALTA score → alta_gate4 or greedy_gate4
Gate 5 medical CoVe or greedy fallback

Thresholds: `configs/router_thresholds.json`

---

## Run

```bash
pip install -r requirements.txt

python cured.py \
--model meta-llama/Llama-3.1-8B-Instruct \
--load-in-4bit \
--protocols greedy,alta,cured \
--router new \
--benchmark truthfulqa \
--n 500 --seed 42 \
--out results/my_run.json
```

Canonical results: `results/CANONICAL_v2/`
