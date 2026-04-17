#!/usr/bin/env python3
"""
patch_cured.py — applies all CURED MC scoring fixes in one shot.

Run from repo root:
    python patch_cured.py cured.py

Changes applied:
  1. ALTA_R2_CUTOFF: 0.55 → 0.50
  2. Add _average_choice_log_prob_alta() after existing _average_choice_log_prob()
  3. Add protocol param to mc_score_sample(), branch on it for both MC1 + MC2
  4. Thread strategy-faithful protocol selection into run_protocol() MC scoring call
"""

import sys
import shutil
from pathlib import Path


def patch(src: Path) -> str:
    code = src.read_text(encoding="utf-8")
    errors = []

    # ── 1. ALTA_R2_CUTOFF 0.55 → 0.50 ──────────────────────────────────────
    OLD1 = "ALTA_R2_CUTOFF = 0.55"
    NEW1 = "ALTA_R2_CUTOFF = 0.50"
    if OLD1 in code:
        code = code.replace(OLD1, NEW1, 1)
        print("  ✓ ALTA_R2_CUTOFF lowered to 0.50")
    else:
        errors.append("ALTA_R2_CUTOFF line not found")

    # ── 2. Insert _average_choice_log_prob_alta() ────────────────────────────
    ANCHOR = "def _average_choice_log_prob("
    ALTA_FN = '''

def _average_choice_log_prob_alta(
    model: Any,
    tokenizer: Any,
    question: str,
    choice: str,
    early_idx: int = ALTA_EARLY_IDX,
    mid_idx: int = ALTA_MID_IDX,
    top_k: int = ALTA_TOP_K,
    alpha_c: float = ALTA_ALPHA_C,
    alpha_e: float = ALTA_ALPHA_E,
) -> float:
    """Like _average_choice_log_prob but applies ALTA logit correction at each answer token."""
    dev = get_model_device(model)
    prompt_text = format_prompt(tokenizer, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": str(choice)},
    ]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        full_text = f"Question: {question}\\nAnswer: {choice}"
    try:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    except Exception:
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)

    if len(full_ids) <= len(prompt_ids):
        try:
            answer_ids = tokenizer.encode(" " + str(choice).strip(), add_special_tokens=False)
        except Exception:
            answer_ids = tokenizer.encode(" " + str(choice).strip())
        full_ids = list(prompt_ids) + list(answer_ids)

    prompt_len = len(prompt_ids)
    if prompt_len <= 0 or len(full_ids) <= prompt_len:
        return -999.0

    input_ids = torch.tensor([full_ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        # Fallback: no hidden states available, use baseline scorer
        return _average_choice_log_prob(model, tokenizer, question, choice)

    hidden_states = hidden_states[1:]  # skip embedding layer
    norm, lm_head = get_final_norm_and_lm_head(model)

    total_lp = 0.0
    n_tokens = 0
    for offset in range(prompt_len, len(full_ids) - 1):
        layer_logits_list = []
        for h in hidden_states:
            hs = h[:, offset, :]
            logits = lm_head(norm(hs)).squeeze(0).detach().to(torch.float32).cpu().numpy()
            layer_logits_list.append(logits)
        layer_logits_np = np.asarray(layer_logits_list, dtype=np.float32)

        corrected, _, _ = alta_logits(
            layer_logits_np,
            early_idx=early_idx,
            mid_idx=mid_idx,
            top_k=top_k,
            alpha_contrast=alpha_c,
            alpha_extrap=alpha_e,
        )
        shifted = corrected - np.max(corrected)
        log_probs = shifted - np.log(np.sum(np.exp(shifted)))
        tok_id = int(full_ids[offset + 1])
        total_lp += float(log_probs[tok_id])
        n_tokens += 1

    return total_lp / max(n_tokens, 1)

'''

    if ANCHOR in code and "_average_choice_log_prob_alta" not in code:
        code = code.replace(ANCHOR, ALTA_FN + ANCHOR, 1)
        print("  ✓ _average_choice_log_prob_alta() inserted")
    elif "_average_choice_log_prob_alta" in code:
        print("  ✓ _average_choice_log_prob_alta() already present — skipping")
    else:
        errors.append("Could not find _average_choice_log_prob anchor")

    # ── 3. Add protocol param to mc_score_sample signature ──────────────────
    OLD3 = (
        "def mc_score_sample(\n"
        "    model: Any,\n"
        "    tokenizer: Any,\n"
        "    question: str,\n"
        "    choices: list[str],\n"
        "    labels: list[int],\n"
        "    choices_mc2: list[str] | None = None,\n"
        "    labels_mc2: list[int] | None = None,\n"
        ") -> dict[str, float]:"
    )
    NEW3 = (
        "def mc_score_sample(\n"
        "    model: Any,\n"
        "    tokenizer: Any,\n"
        "    question: str,\n"
        "    choices: list[str],\n"
        "    labels: list[int],\n"
        "    choices_mc2: list[str] | None = None,\n"
        "    labels_mc2: list[int] | None = None,\n"
        "    mc_protocol: str = \"greedy\",\n"
        ") -> dict[str, float]:"
    )
    if OLD3 in code:
        code = code.replace(OLD3, NEW3, 1)
        print("  ✓ mc_score_sample() signature extended with mc_protocol param")
    elif "mc_protocol: str" in code:
        print("  ✓ mc_protocol param already present in mc_score_sample — skipping")
    else:
        errors.append("mc_score_sample signature not matched")

    # ── 4. Branch MC1 scorer on mc_protocol ─────────────────────────────────
    OLD4 = (
        "    mc1_log_probs = np.asarray(\n"
        "        [_average_choice_log_prob(model, tokenizer, question, c) for c in choices],\n"
        "        dtype=np.float32,\n"
        "    )"
    )
    NEW4 = (
        "    _mc_score_fn = _average_choice_log_prob_alta if mc_protocol == \"alta\" else _average_choice_log_prob\n"
        "    mc1_log_probs = np.asarray(\n"
        "        [_mc_score_fn(model, tokenizer, question, c) for c in choices],\n"
        "        dtype=np.float32,\n"
        "    )"
    )
    if OLD4 in code:
        code = code.replace(OLD4, NEW4, 1)
        print("  ✓ MC1 scorer branched on mc_protocol")
    elif "_mc_score_fn" in code:
        print("  ✓ MC1 scorer branch already present — skipping")
    else:
        errors.append("MC1 log_probs line not matched")

    # ── 5. Branch MC2 scorer on mc_protocol ─────────────────────────────────
    OLD5 = (
        "    mc2_log_probs = np.asarray(\n"
        "        [_average_choice_log_prob(model, tokenizer, question, c) for c in use_mc2_choices],\n"
        "        dtype=np.float32,\n"
        "    )"
    )
    NEW5 = (
        "    mc2_log_probs = np.asarray(\n"
        "        [_mc_score_fn(model, tokenizer, question, c) for c in use_mc2_choices],\n"
        "        dtype=np.float32,\n"
        "    )"
    )
    if OLD5 in code:
        code = code.replace(OLD5, NEW5, 1)
        print("  ✓ MC2 scorer branched on _mc_score_fn")
    elif "use_mc2_choices" in code and "_mc_score_fn" in code:
        print("  ✓ MC2 scorer branch already updated — skipping")
    else:
        errors.append("MC2 log_probs line not matched")

    # ── 6. Thread strategy-faithful mc_protocol into run_protocol() call ─────
    # Replace the mc_score_sample call site inside run_protocol
    OLD6 = (
        "                mc_scores = mc_score_sample(\n"
        "                    model=model,\n"
        "                    tokenizer=tokenizer,\n"
        "                    question=q,\n"
        "                    choices=[str(c) for c in mc1_choices],\n"
        "                    labels=list(mc1_labels),\n"
        "                    choices_mc2=[str(c) for c in mc2_choices] if isinstance(mc2_choices, list) else None,\n"
        "                    labels_mc2=list(mc2_labels) if isinstance(mc2_labels, list) else None,\n"
        "                )"
    )
    NEW6 = (
        "                # Strategy-faithful MC protocol selection:\n"
        "                # - 'alta' protocol always uses ALTA-aware scoring\n"
        "                # - 'cured' uses ALTA-aware scoring only when router chose an ALTA strategy\n"
        "                # - all others use baseline scoring\n"
        "                _mc_proto = \"greedy\"\n"
        "                if protocol == \"alta\":\n"
        "                    _mc_proto = \"alta\"\n"
        "                elif protocol == \"cured\" and router.alta_viable:\n"
        "                    _alta_strategies = {\"alta_general\", \"alta_medical_structured\", \"alta_general_structured\"}\n"
        "                    if strategy in _alta_strategies:\n"
        "                        _mc_proto = \"alta\"\n"
        "                mc_scores = mc_score_sample(\n"
        "                    model=model,\n"
        "                    tokenizer=tokenizer,\n"
        "                    question=q,\n"
        "                    choices=[str(c) for c in mc1_choices],\n"
        "                    labels=list(mc1_labels),\n"
        "                    choices_mc2=[str(c) for c in mc2_choices] if isinstance(mc2_choices, list) else None,\n"
        "                    labels_mc2=list(mc2_labels) if isinstance(mc2_labels, list) else None,\n"
        "                    mc_protocol=_mc_proto,\n"
        "                )"
    )
    if OLD6 in code:
        code = code.replace(OLD6, NEW6, 1)
        print("  ✓ mc_score_sample() call threaded with strategy-faithful _mc_proto")
    elif "_mc_proto" in code:
        print("  ✓ _mc_proto threading already present — skipping")
    else:
        errors.append("mc_score_sample() call site in run_protocol not matched")

    if errors:
        print("\n⚠ Unmatched patches (apply manually):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ All patches applied cleanly.")

    return code


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cured.py")
    if not src.exists():
        print(f"Error: {src} not found. Run from repo root.")
        sys.exit(1)

    backup = src.with_suffix(".py.bak")
    shutil.copy2(src, backup)
    print(f"Backup: {backup}")

    patched = patch(src)
    src.write_text(patched, encoding="utf-8")
    print(f"Written: {src}")


if __name__ == "__main__":
    main()
