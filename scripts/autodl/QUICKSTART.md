# Borrowed GPU Quickstart

These steps run the clean final workflow and write canonical outputs into
results/CANONICAL_v2.

## 1) Copy-paste setup commands (Linux GPU instance)

```bash
git clone https://github.com/AlexDongzeyu/llm-hallucination-self-testing.git
cd llm-hallucination-self-testing
bash scripts/autodl/bootstrap_gpu_env.sh
```

## 2) Run the complete final suite (recommended)

```bash
bash scripts/autodl/run_final_suite.sh
```

Primary outputs written to results/CANONICAL_v2:

- results_8b_truthfulqa_full_mc.json
- results_3b_truthfulqa_full_mc.json
- results_8b_medhallu_v2.json
- results_8b_pubmedqa_v2.json
- results_8b_medqa_v3_fixed.json
- results_3b_medhallu_n100.json

and logs under `logs/` with timestamped names.

## 3) Optional: run only the local 8B custom subset

```bash
bash scripts/autodl/run_local_v2_suite.sh
```

This writes:

- results/CANONICAL_v2/results_8b_medqa_v2.json
- results/CANONICAL_v2/results_8b_pubmedqa_v2.json
- results/CANONICAL_v2/results_8b_medhallu_v2.json

## 4) Optional useful variants

Run only if you need to rerun existing outputs:

```bash
FORCE_RERUN=1 bash scripts/autodl/run_local_v2_suite.sh
```

Run without 4-bit quantization:

```bash
LOAD_IN_4BIT=0 bash scripts/autodl/run_local_v2_suite.sh
```

Use a different model id:

```bash
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct bash scripts/autodl/run_local_v2_suite.sh
```

## 5) Post-run normalization and cleanup

```bash
bash scripts/autodl/organize_final_results.sh
```

This moves legacy root-level outputs into results/CANONICAL_v2 and archives
legacy API v1 outputs under results/archive.

## Protocol-Aware MC Rerun (Required for Paper)

After merging the cured.py patch (`python patch_cured.py cured.py`):

### Run 8B and 3B MC reruns in parallel (A100 40GB — both fit)

```bash
# Window 1 — 8B TruthfulQA MC v2
# No --load-in-4bit: A100 has 40GB, bfloat16 gives cleaner results
python -u cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --protocols greedy,alta,cove,cured \
  --skip-iti \
  --benchmark truthfulqa \
  --n 817 --scoring mc --max-new-tokens 50 \
  --force-recalibrate \
  --out results/CANONICAL_v2/results_8b_truthfulqa_full_mc_v2.json \
  > logs/8b_tqa_mc_v2.log 2>&1 &

# Window 2 — 3B TruthfulQA MC v2 (run simultaneously)
python -u cured.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --protocols greedy,alta,delta_dola,cove,cured \
  --skip-iti \
  --benchmark truthfulqa \
  --n 817 --scoring mc --max-new-tokens 50 \
  --force-recalibrate \
  --out results/CANONICAL_v2/results_3b_truthfulqa_full_mc_v2.json \
  > logs/3b_tqa_mc_v2.log 2>&1 &
```

### After MC v2 completes — generation eval with larger n

```bash
# 8B both benchmarks, cosine, n=100, no quantization
python -u cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --protocols greedy,alta,cove,cured \
  --skip-iti \
  --benchmark both \
  --n 100 --scoring cosine --max-new-tokens 80 \
  --force-recalibrate \
  --out results/CANONICAL_v2/results_8b_both_n100.json \
  > logs/8b_both_n100.log 2>&1
```

### Archive old (pre-fix) MC files only after v2 validation

```bash
mv results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json results/archive/
mv results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json results/archive/
```

### --force-recalibrate is required

The cached `calibration.json` has `alta_viable: false` because it was generated
when `ALTA_R2_CUTOFF=0.65`. With the patch setting it to `0.50`, and R²=0.5557,
ALTA becomes viable — but only after a forced recalibrate overwrites the cache.
