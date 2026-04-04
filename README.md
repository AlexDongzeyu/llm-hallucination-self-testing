# LLM_Hallucination

Inference-time hallucination reduction experiments for LLMs, including:

- entropy/SLED gating baselines,
- best-of-N and semantic-majority decoding,
- CoVe-style self-checking,
- ITI interventions,
- a learned routing policy (GADR-2) trained from trajectory features.

## Latest Completed Runs (Apr 3, 2026)

The full learned-router pipeline was run end-to-end and completed successfully:

1. Built routing dataset (`results/routing_dataset.csv`, 100 rows + header).
2. Trained router (`results/router_model.joblib`).
3. Evaluated Instruct sweep with GADR-2 on TruthfulQA (`results/instruct_results.json`).
4. Evaluated MedHallu with GADR-2 (`results/medhallu_results.json`).

Completion markers:

- Orchestrated sequence: `results/logs/pipeline_remaining_steps.done` -> `COMPLETED 2026-04-03T04:47:58`
- Latest standalone MedHallu refresh: `results/medhallu_results.json` last written `2026-04-03 22:25:10`

## Key Results Snapshot

### TruthfulQA Instruct Sweep (n=50)

| Method | Accuracy | Rep Rate |
| :-- | --: | --: |
| Semantic Majority BoN (T=0.4, n=5) | 70.0% | 0.0% |
| CoVe (2 checks) | 60.0% | 4.0% |
| GADR-2 Learned Router | 74.0% | 0.0% |

### MedHallu (UTAustin-AIHealth/MedHallu, pqa_artificial, train, n=50)

| Method | Accuracy | Rep Rate |
| :-- | --: | --: |
| greedy | 46.0% | 0.0% |
| cove | 54.0% | 0.0% |
| dynamic | 48.0% | 8.0% |
| gadr2 | 52.0% | 2.0% |

## Router Training Snapshot

From `src/learn_router.py` run on `results/routing_dataset.csv`:

- Label distribution: greedy=50, iti=25, cove=25.
- Decision Tree CV accuracy: 0.830.
- Logistic baseline CV accuracy: 0.780.
- Saved deployed router: `results/router_model.joblib`.

## Simple Run Paths

- Full pipeline: `experiments/pipeline_all_steps.ps1`
- Resume from routing dataset: `experiments/pipeline_remaining_steps.ps1`
- Build routing dataset only: `experiments/build_routing_dataset.py`
- Train router only: `src/learn_router.py`
- Instruct evaluation only: `experiments/eval_instruct.py`
- MedHallu evaluation only: `experiments/eval_medhallu.py`

## Key Files

- Consolidated synced report: `raw_results.md`
- Main result artifacts: `results/*.json`
- Orchestrator log: `results/logs/pipeline_remaining_steps_20260403_023407.log`
