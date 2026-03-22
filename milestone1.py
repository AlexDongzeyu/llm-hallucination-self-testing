import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

def extract_logit_trajectory(model_name, prompt, target_token):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load in bfloat16 to save memory and match typical Llama 3 weighting, auto-device
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print(f"\nTokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get the token ID for the target we want to track
    target_id = tokenizer.encode(target_token, add_special_tokens=False)
    if not target_id:
        print(f"Could not encode target token: {target_token}")
        return
    # We take the first subword if the target token is split
    target_id = target_id[0]
    print(f"Tracking candidate token '{target_token}' (ID: {target_id})")
    
    print("Running forward pass with hidden states...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    
    print(f"Extracting hidden states across {len(hidden_states)} layers...")
    
    # MILESTONE REQUIREMENT: Decode top predicted token from the final layer
    # The last hidden state represents the output of the final transformer block
    final_layer_hidden_state = hidden_states[-1][0, -1, :] 
    
    # Apply final LayerNorm if the model architecture requires it before the lm_head
    if hasattr(model.model, 'norm'):
        final_normed = model.model.norm(final_layer_hidden_state)
    else:
        final_normed = final_layer_hidden_state
        
    final_logits = model.lm_head(final_normed)
    top_predicted_id = torch.argmax(final_logits).item()
    top_predicted_token = tokenizer.decode([top_predicted_id])
    print(f"\n[Milestone 1] Top predicted token from final layer: '{top_predicted_token}' (ID: {top_predicted_id})")
    
    # MILESTONE REQUIREMENT: Save layer-wise scores for one candidate token
    layer_logits = []
    
    for i, h in enumerate(hidden_states):
        # We only care about the hidden state of the final token in the prompt sequence
        h_last_token = h[0, -1, :]
        
        # We handle standard Llama/Qwen architectures where model.lm_head exists
        if hasattr(model, 'lm_head'):
            # Hidden states must be normalized (except usually the embedding layer at index 0)
            # to be properly projected by the final lm_head. This is an approximation of 
            # early-exiting to the head.
            if hasattr(model.model, 'norm') and i > 0: 
                 h_last_token = model.model.norm(h_last_token)
            
            logits = model.lm_head(h_last_token)
            target_logit = logits[target_id].item()
            layer_logits.append(target_logit)
        else:
            print("Model doesn't have standard lm_head, cannot project.")
            break
            
    print(f"\n[Milestone 1] Layer-wise Logits for target token '{target_token}':")
    for i, l in enumerate(layer_logits):
        print(f"Layer {i}: {l:.4f}")
        
    # Plotting the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(layer_logits)), layer_logits, marker='o', linestyle='-', color='b')
    plt.title(f"Logit Trajectory for '{target_token}' across Layers")
    plt.xlabel("Layer Depth")
    plt.ylabel("Logit Value (Unnormalized Confidence)")
    plt.grid(True)
    plt.savefig("trajectory_plot.png")
    print("\n[Milestone 1] Saved plot to trajectory_plot.png")

if __name__ == "__main__":
    # MILESTONE REQUIREMENT: Try Llama 3.2-3B if your hardware can handle it
    model_name = "meta-llama/Llama-3.2-3B" 
    prompt = "The capital of Canada is"
    target_token = " Ottawa"
    
    extract_logit_trajectory(model_name, prompt, target_token)
