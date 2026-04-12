# Borrowed GPU Quickstart

These steps keep the current repository layout and output filenames unchanged.

## 1) Copy-paste setup commands (Linux GPU instance)

```bash
git clone https://github.com/AlexDongzeyu/llm-hallucination-self-testing.git
cd llm-hallucination-self-testing
bash scripts/autodl/bootstrap_gpu_env.sh
```

## 2) Run the full local v2 custom suite

```bash
bash scripts/autodl/run_local_v2_suite.sh
```

This writes the canonical local v2 outputs:

- `results_8b_medqa_v2.json`
- `results_8b_pubmedqa_v2.json`
- `results_8b_medhallu_v2.json`

and logs under `logs/` with timestamped names.

## 3) Optional useful variants

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
