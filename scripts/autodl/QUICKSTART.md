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
