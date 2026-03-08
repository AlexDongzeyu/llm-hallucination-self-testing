import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_trajectory(model, tokenizer, prompt, target_token_str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We take the first subword if the target token is split by the tokenizer
    target_id = tokenizer.encode(target_token_str, add_special_tokens=False)
    if not target_id:
        return None
    target_id = target_id[0]
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    layer_logits = []
    
    for i, h in enumerate(hidden_states):
        h_last_token = h[0, -1, :]
        if hasattr(model, 'lm_head'):
            if hasattr(model.model, 'norm') and i > 0:
                 h_last_token = model.model.norm(h_last_token)
            logits = model.lm_head(h_last_token)
            target_logit = logits[target_id].item()
            layer_logits.append(target_logit)
        else:
            break
    return layer_logits

def compute_features(layer_logits):
    final_logit = layer_logits[-1]
    
    # slope: using linear regression over the indices to get the stable slope
    x = np.arange(len(layer_logits))
    slope, _ = np.polyfit(x, layer_logits, 1)
    
    # curvature: coefficient of x^2 in quadratic polynomial fit
    poly2 = np.polyfit(x, layer_logits, 2)
    curvature = poly2[0]
    
    # stability: standard deviation of step-wise differences
    stability = float(np.std(np.diff(layer_logits)))
    
    return final_logit, slope, curvature, stability

def main():
    model_name = "meta-llama/Llama-3.2-3B" 
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    prompts_data = [
        {
            "original": "The capital of Canada is",
            "masked": "The capital of [MASK] is",
            "entity": "Canada",
            "correct": " Ottawa",
            "wrong": [" Toronto", " Montreal", " Vancouver"]
        },
        {
            "original": "The prime minister of Canada is",
            "masked": "The prime minister of [MASK] is",
            "entity": "Canada",
            "correct": " Mark Carney",
            "wrong": [" Justin Trudeau", " Pierre Poilievre"]
        },
        {
            "original": "The currency of Japan is",
            "masked": "The currency of [MASK] is",
            "entity": "Japan",
            "correct": " Yen",
            "wrong": [" Won", " Yuan", " Dollar"]
        },
        {
            "original": "The largest ocean on Earth is",
            "masked": "The largest [MASK] on Earth is",
            "entity": "ocean",
            "correct": " Pacific",
            "wrong": [" Atlantic", " Indian", " Arctic"]
        },
        {
            "original": "The author of Harry Potter is",
            "masked": "The author of [MASK] is",
            "entity": "Harry Potter",
            "correct": " Rowling",
            "wrong": [" Tolkien", " Martin", " King"]
        },
        {
            "original": "The chemical symbol for gold is",
            "masked": "The chemical symbol for [MASK] is",
            "entity": "gold",
            "correct": " Au",
            "wrong": [" Ag", " Fe", " Cu"]
        },
        {
            "original": "The tallest mountain in the world is",
            "masked": "The tallest [MASK] in the world is",
            "entity": "mountain",
            "correct": " Everest",
            "wrong": [" K", " Kilimanjaro", " Denali"]
        },
        {
            "original": "The first president of the United States was",
            "masked": "The first president of [MASK] was",
            "entity": "the United States",
            "correct": " Washington",
            "wrong": [" Lincoln", " Jefferson", " Adams"]
        },
        {
            "original": "The planet closest to the Sun is",
            "masked": "The planet closest to [MASK] is",
            "entity": "the Sun",
            "correct": " Mercury",
            "wrong": [" Venus", " Mars", " Earth"]
        },
        {
            "original": "The capital of France is",
            "masked": "The capital of [MASK] is",
            "entity": "France",
            "correct": " Paris",
            "wrong": [" London", " Berlin", " Rome"]
        },
        {
            "original": "The language spoken in Brazil is",
            "masked": "The language spoken in [MASK] is",
            "entity": "Brazil",
            "correct": " Portuguese",
            "wrong": [" Spanish", " English", " French"]
        },
        {
            "original": "The largest desert in the world is",
            "masked": "The largest [MASK] in the world is",
            "entity": "desert",
            "correct": " Sahara",
            "wrong": [" Gobi", " Kalahari", " Mojave"]
        },
        {
            "original": "The freezing point of water in Celsius is",
            "masked": "The freezing point of [MASK] in Celsius is",
            "entity": "water",
            "correct": " zero",
            "wrong": [" ten", " thirty", " hundred"]
        },
        {
            "original": "The main ingredient in guacamole is",
            "masked": "The main ingredient in [MASK] is",
            "entity": "guacamole",
            "correct": " avocado",
            "wrong": [" tomato", " onion", " pepper"]
        },
        {
            "original": "The author of the play Romeo and Juliet is",
            "masked": "The author of the play [MASK] is",
            "entity": "Romeo and Juliet",
            "correct": " Shakespeare",
            "wrong": [" Dickens", " Austen", " Hemingway"]
        },
        {
            "original": "The country known as the Land of the Rising Sun is",
            "masked": "The country known as the [MASK] is",
            "entity": "Land of the Rising Sun",
            "correct": " Japan",
            "wrong": [" China", " Korea", " Vietnam"]
        }
    ]

    all_results = []
    
    os.makedirs("plots", exist_ok=True)
    
    for idx, data in enumerate(prompts_data):
        print("\n" + "="*50)
        print(f"Processing prompt {idx + 1}/{len(prompts_data)}")
        
        # Display the prompt details in requested format
        print(f"Original prompt: {data['original']}")
        print(f"Masked prompt: {data['masked']}")
        print(f"Correct answer: {data['correct'].strip()}")
        print(f"Wrong answers: {', '.join([w.strip() for w in data['wrong']])}")
        print("="*50)
        
        orig_prompt = data["original"]
        mask_prompt = data["masked"]
        correct_token = data["correct"]
        wrong_tokens = data["wrong"]
        
        all_candidates = [correct_token] + wrong_tokens
        
        orig_trajectories = {}
        mask_trajectories = {}
        
        for cand in all_candidates:
            orig_trajectories[cand] = get_trajectory(model, tokenizer, orig_prompt, cand)
            mask_trajectories[cand] = get_trajectory(model, tokenizer, mask_prompt, cand)
            
        for cand in all_candidates:
            is_correct = "correct" if cand == correct_token else "wrong"
            orig_traj = orig_trajectories[cand]
            mask_traj = mask_trajectories[cand]
            
            if orig_traj is None or mask_traj is None:
                print(f"Skipping candidate '{cand}' due to tokenization issue.")
                continue
                
            orig_final, orig_slope, orig_curv, orig_stab = compute_features(orig_traj)
            mask_final, mask_slope, mask_curv, mask_stab = compute_features(mask_traj)
            
            masked_drop_val = orig_final - mask_final
            
            # Save original trajectories and features per layer
            for layer_idx, logit_val in enumerate(orig_traj):
                all_results.append({
                    "prompt": orig_prompt,
                    "masked_entity": data["entity"],
                    "prompt_type": "original",
                    "candidate_token": cand.strip(),
                    "correct_or_wrong": is_correct,
                    "layer": layer_idx,
                    "logit": logit_val,
                    "final_logit": orig_final,
                    "slope": orig_slope,
                    "curvature": orig_curv,
                    "stability": orig_stab,
                    "masked_drop": masked_drop_val
                })
                
            # Save masked trajectories and features per layer
            for layer_idx, logit_val in enumerate(mask_traj):
                all_results.append({
                    "prompt": mask_prompt,
                    "masked_entity": data["entity"],
                    "prompt_type": "masked",
                    "candidate_token": cand.strip(),
                    "correct_or_wrong": is_correct,
                    "layer": layer_idx,
                    "logit": logit_val,
                    "final_logit": mask_final,
                    "slope": mask_slope,
                    "curvature": mask_curv,
                    "stability": mask_stab,
                    "masked_drop": masked_drop_val
                })

        # Generate Plot 1: Correct vs Wrong candidates on Original prompt
        plt.figure(figsize=(10, 6))
        for cand in all_candidates:
            traj = orig_trajectories.get(cand)
            if traj is not None:
                is_curr_correct = (cand == correct_token)
                label = f"{cand.strip()} (Correct)" if is_curr_correct else f"{cand.strip()} (Wrong)"
                linestyle = '-' if is_curr_correct else '--'
                linewidth = 3 if is_curr_correct else 1.5
                alpha = 1.0 if is_curr_correct else 0.7
                plt.plot(range(len(traj)), traj, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        plt.title(f"Original Prompt: '{orig_prompt}'\nTrajectories of Correct vs. Wrong Answers")
        plt.xlabel("Transformer Layer")
        plt.ylabel("Logit Value (Unnormalized Confidence)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"plots/prompt_{idx+1}_orig_correct_vs_wrong.png")
        plt.close()
        
        # Generate Plot 2: Unmasked vs Masked trajectories for Correct answer
        plt.figure(figsize=(10, 6))
        
        if orig_trajectories.get(correct_token) is not None:
            plt.plot(range(len(orig_trajectories[correct_token])), orig_trajectories[correct_token], 
                     label=f"Unmasked: '{orig_prompt}'", linestyle='-', linewidth=3, color='blue')
                     
        if mask_trajectories.get(correct_token) is not None:
            plt.plot(range(len(mask_trajectories[correct_token])), mask_trajectories[correct_token], 
                     label=f"Masked: '{mask_prompt}'", linestyle='--', linewidth=3, color='red')
        
        plt.title(f"Correct Answer '{correct_token.strip()}' Trajectory\nUnmasked vs. Masked")
        plt.xlabel("Transformer Layer")
        plt.ylabel("Logit Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"plots/prompt_{idx+1}_correct_unmasked_vs_masked.png")
        plt.close()
        
    print("\nAll prompts processed. Saving results to CSV...")
    df = pd.DataFrame(all_results)
    df.to_csv("trajectories_dataset.csv", index=False)
    print("Saved 'trajectories_dataset.csv'.")
    print("\n[AI PROCESS STATS] Summary across dataset:")
    
    # Compute and display some quick stats for the console
    correct_df = df[(df["correct_or_wrong"] == "correct") & (df["prompt_type"] == "original")]
    correct_finals = correct_df.drop_duplicates(subset=["prompt"])["final_logit"].mean()
    
    wrong_df = df[(df["correct_or_wrong"] == "wrong") & (df["prompt_type"] == "original")]
    wrong_finals = wrong_df.drop_duplicates(subset=["prompt", "candidate_token"])["final_logit"].mean()
    
    mean_drop = correct_df.drop_duplicates(subset=["prompt"])["masked_drop"].mean()
    
    print(f"  -> Average Final Logit (Correct, Original): {correct_finals:.4f}")
    print(f"  -> Average Final Logit (Wrong, Original):   {wrong_finals:.4f}")
    print(f"  -> Average Masked Drop (Correct Answer):    {mean_drop:.4f}")
    print(f"  -> Total layers processed per trajectory:   {int(df['layer'].max() + 1)}")
    print(f"  -> Total plots generated:                   {len(prompts_data) * 2}")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
