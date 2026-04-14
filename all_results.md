# All Results (Complete)

Last updated: 2026-04-14 13:01:23 (local workspace)

This is the single complete results document for this project.

## Quick Summary

- Remote post-suite pipeline finished successfully.
- All remote v2 reruns completed and are recorded below.
- Local workspace currently has 48 JSON result artifacts indexed in this file.
- Nothing is omitted: this document includes remote-final metrics plus full local inventories.

## Final Run Status

- Post-suite queue run: completed successfully on remote server
- Queue start: 2026-04-14T11:45:51+08:00
- MC v2 completion: rc8=0, rc3=0
- both_n100_v2 completion: rc=0
- Queue end: 2026-04-14T22:02:51+08:00

## MC Scoring Status

- Superseded local pre-fix MC file:
  - results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json
- Previously expected pre-fix 3B file is not present in this local workspace.
- Valid v2 MC reruns are completed remotely and summarized below.

## Canonical v2 Metrics (Remote Final)

Source: remote host /root/llm-hallucination-self-testing/results/CANONICAL_v2

| file | benchmark | protocol | acc | rep | n_scored | runtime_min |
|---|---|---|---:|---:|---:|---:|
| results_8b_truthfulqa_full_mc_v2.json | truthfulqa | greedy | 0.4027 | 0.0000 | 817 | 65.08 |
| results_8b_truthfulqa_full_mc_v2.json | truthfulqa | alta | 0.4027 | 0.0000 | 817 | 65.91 |
| results_8b_truthfulqa_full_mc_v2.json | truthfulqa | cove | 0.4027 | 0.0012 | 817 | 291.41 |
| results_8b_truthfulqa_full_mc_v2.json | truthfulqa | cured | 0.4027 | 0.0000 | 817 | 76.49 |
| results_3b_truthfulqa_full_mc_v2.json | truthfulqa | greedy | 0.3635 | 0.0000 | 817 | 55.40 |
| results_3b_truthfulqa_full_mc_v2.json | truthfulqa | alta | 0.3635 | 0.0012 | 817 | 59.12 |
| results_3b_truthfulqa_full_mc_v2.json | truthfulqa | delta_dola | 0.3635 | 0.0012 | 817 | 59.35 |
| results_3b_truthfulqa_full_mc_v2.json | truthfulqa | cove | 0.3635 | 0.0000 | 817 | 256.54 |
| results_3b_truthfulqa_full_mc_v2.json | truthfulqa | cured | 0.3635 | 0.0000 | 817 | 64.78 |
| results_8b_both_n100_v2.json | truthfulqa | greedy | 0.6465 | 0.0100 | 99 | 7.65 |
| results_8b_both_n100_v2.json | truthfulqa | alta | 0.6162 | 0.0100 | 99 | 8.26 |
| results_8b_both_n100_v2.json | truthfulqa | cove | 0.6263 | 0.0100 | 99 | 27.94 |
| results_8b_both_n100_v2.json | truthfulqa | cured | 0.6162 | 0.0100 | 99 | 8.28 |
| results_8b_both_n100_v2.json | medhallu | greedy | 0.5758 | 0.0100 | 99 | 8.23 |
| results_8b_both_n100_v2.json | medhallu | alta | 0.5859 | 0.0100 | 99 | 8.61 |
| results_8b_both_n100_v2.json | medhallu | cove | 0.4949 | 0.0100 | 99 | 29.01 |
| results_8b_both_n100_v2.json | medhallu | cured | 0.5204 | 0.0200 | 98 | 15.84 |

## Sync Gap (Remote -> Local Workspace)

The following completed canonical files are not yet present in this local workspace:
- results/CANONICAL_v2/results_8b_truthfulqa_full_mc_v2.json
- results/CANONICAL_v2/results_3b_truthfulqa_full_mc_v2.json
- results/CANONICAL_v2/results_8b_both_n100_v2.json

## Inventory: Local Canonical Files

| path | size_kb | modified | kind | summary |
|---|---:|---|---|---|
| results/CANONICAL_v2/results_8b_both.json | 2.3 | 2026-04-11 19:00:04 | eval-dict | benchmark=truthfulqa,medhallu; scoring=-; protocols=alta,cove,cured,greedy |
| results/CANONICAL_v2/results_8b_medqa_v2.json | 1.4 | 2026-04-12 12:10:45 | eval-dict | benchmark=custom; scoring=letter; protocols=alta,cove,cured,greedy |
| results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json | 1.5 | 2026-04-13 13:16:13 | eval-dict | benchmark=truthfulqa; scoring=mc; protocols=alta,cove,cured,greedy |

## Inventory: Local Active Results (results/*.json)

| path | size_kb | modified | kind | summary |
|---|---:|---|---|---|
| results/alta_3b_results.json | 0.4 | 2026-04-05 15:24:28 | analysis | top_keys=method,model,n,threshold,accuracy,rep_rate |
| results/bon_results.json | 0.4 | 2026-03-31 22:46:29 | list | n_items=3 |
| results/calibration_results.json | 0.7 | 2026-03-31 20:46:10 | analysis | top_keys=meta_llama_Llama_3_1_8B,meta_llama_Llama_3_2_3B,Qwen_Qwen2_5_3B |
| results/entropy_by_layer.json | 25.2 | 2026-04-05 12:41:29 | analysis | top_keys=n_questions,n_layers,layer_means,layer_stds,layer_mins,layer_maxs |
| results/generation_results.json | 0.8 | 2026-03-31 05:11:01 | list | n_items=3 |
| results/generation_results_n100_4configs.json | 1.1 | 2026-04-02 20:00:53 | list | n_items=4 |
| results/grid_search_results.json | 6.6 | 2026-03-29 14:45:20 | analysis | top_keys=baseline_accuracy,phase1,phase2 |
| results/instruct_results.json | 0.5 | 2026-04-03 16:15:10 | list | n_items=3 |
| results/iti_results.json | 0.7 | 2026-04-02 21:00:41 | list | n_items=6 |
| results/logit_linearity_3b.json | 6.3 | 2026-04-05 14:28:50 | analysis | top_keys=model,n_questions,mid_layer,top_k,mean_r2,median_r2 |
| results/medhallu_ablation_results.json | 1.3 | 2026-04-05 03:51:32 | eval-list | benchmark=-; scoring=-; labels=bon3_t0.3,iti_alpha0.5,sled |
| results/medhallu_generation_results.json | 2.5 | 2026-04-04 23:47:09 | eval-list | benchmark=UTAustin-AIHealth/MedHallu pqa_artificial train; scoring=cosine_similarity_to_ground_truth; labels=cove,cove_rag,delta_dola,gadr2_cured,greedy |
| results/medhallu_results.json | 1.4 | 2026-04-04 19:37:41 | eval-list | benchmark={'id': 'UTAustin-AIHealth/MedHallu', 'subset': 'pqa_artificial', 'split': 'train'}; scoring=-; labels=delta_dola_mc_a10.3_a20.3,greedy_mc |
| results/online_results.json | 1.2 | 2026-04-02 14:14:21 | list | n_items=4 |
| results/results_cloudflare_medhallu_v2.json | 1.2 | 2026-04-11 20:20:02 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=cosine; protocols=custom |
| results/results_cloudflare_medqa_fixed.json | 1.1 | 2026-04-11 12:17:21 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/results_cloudflare_medqa_v2.json | 1.2 | 2026-04-11 19:45:45 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/results_cloudflare_pubmedqa_v2.json | 1.2 | 2026-04-11 19:58:18 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=yesno; protocols=custom |
| results/results_openrouter_both.json | 1.9 | 2026-04-10 16:06:54 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/results_openrouter_medhallu_v2.json | 1.2 | 2026-04-11 20:20:02 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=cosine; protocols=custom |
| results/results_openrouter_medqa_v2.json | 1.2 | 2026-04-11 19:45:45 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/results_openrouter_pubmedqa_v2.json | 1.2 | 2026-04-11 19:58:18 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=yesno; protocols=custom |
| results/selfcheck_results.json | 0.4 | 2026-04-02 21:24:44 | analysis | top_keys=dataset,n,k_samples,selfcheck_similarity_threshold,reference_threshold,qa_accuracy |
| results/truthfulqa_delta_dola_sweep.json | 3.6 | 2026-04-05 03:15:31 | eval-list | benchmark=-; scoring=-; n_rows=25 |

## Inventory: Local Archive Results (results/archive/**/*.json)

Total archived JSON files: 21

| path | size_kb | modified | kind | summary |
|---|---:|---|---|---|
| results/archive/20260411_cleanup/smoke_cf_account_a0a25954e31bac6ca5ccad3ad5d1b529_20260411_212737.json | 0.6 | 2026-04-11 21:27:50 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/archive/20260411_cleanup/smoke_cloudflare_20260411_192811_k1.json | 0.6 | 2026-04-11 19:28:24 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/archive/20260411_cleanup/smoke_cloudflare_20260411_192811_k2.json | 0.6 | 2026-04-11 19:28:46 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/archive/20260411_cleanup/smoke_openrouter_20260411_192811.json | 0.6 | 2026-04-11 19:29:01 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=letter; protocols=custom |
| results/archive/debug_20260410/results_cloudflare_both_n3.json | 1.9 | 2026-04-09 19:25:38 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/archive/debug_20260410/results_cloudflare_both_n3_fix.json | 1.8 | 2026-04-10 01:42:37 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/archive/debug_20260410/results_cloudflare_pubmedqa_fix_n5.json | 1.1 | 2026-04-10 01:41:20 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/debug_20260410/results_cloudflare_pubmedqa_smoke.json | 1.1 | 2026-04-09 23:37:38 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/debug_20260410/results_openrouter_both_n1_smoke.json | 1.8 | 2026-04-10 03:17:01 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/archive/debug_20260410/results_openrouter_medqa_n20.json | 1.1 | 2026-04-10 02:10:55 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/debug_20260410/results_openrouter_pubmedqa_n20.json | 1.1 | 2026-04-10 02:14:48 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/invalid_openrouter_401_20260410/results_openrouter_both.json | 1.8 | 2026-04-10 13:02:08 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/archive/invalid_openrouter_401_20260410/results_openrouter_medqa.json | 1.1 | 2026-04-10 12:59:28 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/invalid_openrouter_401_20260410/results_openrouter_pubmedqa.json | 1.1 | 2026-04-10 13:00:42 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/medhallu_detector_legacy_results.json | 2.3 | 2026-04-04 04:59:26 | eval-list | benchmark={'id': 'UTAustin-AIHealth/MedHallu', 'subset': 'pqa_artificial', 'split': 'train'}; scoring=-; labels=cove,cove_rag,gadr2,greedy |
| results/archive/medhallu_results_snapshot_n50.json | 1.4 | 2026-04-04 19:33:23 | eval-list | benchmark={'id': 'UTAustin-AIHealth/MedHallu', 'subset': 'pqa_artificial', 'split': 'train'}; scoring=-; labels=delta_dola_mc_a10.3_a20.3,greedy_mc |
| results/archive/provider_debug_20260410/results_cloudflare_both.json | 2.1 | 2026-04-09 19:11:03 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=medhallu,truthfulqa |
| results/archive/provider_debug_20260410/results_cloudflare_medqa.json | 1.2 | 2026-04-09 22:47:09 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/provider_debug_20260410/results_cloudflare_pubmedqa.json | 1.1 | 2026-04-10 00:15:35 | eval-dict | benchmark=@cf/meta/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/results_openrouter_medqa.json | 1.1 | 2026-04-10 16:34:56 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |
| results/archive/results_openrouter_pubmedqa.json | 1.1 | 2026-04-10 16:52:29 | eval-dict | benchmark=meta-llama/llama-3.1-8b-instruct; scoring=-; protocols=custom |

## Totals

- Total JSON result artifacts in workspace: 48
- Active (non-archive, non-canonical_v2): 24
- Canonical_v2 present locally: 3
- Archived JSON artifacts: 21
