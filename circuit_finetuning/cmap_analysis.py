import torch
import json
import argparse
import gc
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import tqdm

def get_best_token_id(model, prompt, target_word):
    t1 = model.to_tokens(" " + target_word.strip(), prepend_bos=False)[0, 0].item()
    t2 = model.to_tokens(target_word.strip(), prepend_bos=False)[0, 0].item()
    with torch.no_grad():
        logits = model(prompt)[0, -1]
    return t1 if logits[t1].item() > logits[t2].item() else t2

def run_cmap(base_model_name, ft_model_path, data_path, circuit_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False) # Save memory !!!!!!!
    
    # Load metadata
    with open(circuit_file, 'r') as f:
        circuit_def = json.load(f)
        # Unique layers we need to cache
        target_layers = sorted(list(set([h['layer'] for h in circuit_def['heads']])))
        target_heads = [(h['layer'], h['head']) for h in circuit_def['heads']]

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f][:50]

    # ==========================================
    # HARVEST ACTIVATIONS (Base Model)
    # ==========================================
    print(f"--> [Phase 1/2] Loading Base Model: {base_model_name}...")
    base_model = HookedTransformer.from_pretrained(
        base_model_name, 
        device=device, 
        dtype=torch.float16
    )
    base_model.eval()
    
    print(f"--> Harvesting activations for {len(data)} examples...")
    stored_activations = [] # List of dicts, one per example
    
    # Filter to only cache 'z' (output of attn heads)
    def filter_z(name):
        return name.endswith("hook_z")

    for i, example in tqdm.tqdm(enumerate(data), total=len(data)):
        prompt = example["clean_prompt"]
        tokens = base_model.to_tokens(prompt)
        _, cache = base_model.run_with_cache(tokens, names_filter=filter_z)
        
        # Store ONLY the layers needed to save GPU memory!!!!!!!
        example_acts = {}
        for layer in target_layers:
            hook_name = f"blocks.{layer}.attn.hook_z"
            # .cpu() moves it to system RAM
            example_acts[layer] = cache[hook_name].cpu() 
            
        stored_activations.append(example_acts)
        
        del cache
        torch.cuda.empty_cache()

    # UNLOAD BASE MODEL
    print("--> Unloading Base Model to free VRAM...")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # INJECT & EVALUATE (FT Model)
    # ==========================================
    print(f"--> [Phase 2/2] Loading Fine-Tuned Model: {ft_model_path}...")
    
    # Load weights via HF first (workaround for local paths)
    hf_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    ft_model = HookedTransformer.from_pretrained(
        base_model_name, 
        hf_model=hf_model,
        device=device, 
        dtype=torch.float16
    )
    ft_model.eval()

    print(f"--> Patching {len(target_heads)} circuit heads...")
    
    results = []

    for i, example in tqdm.tqdm(enumerate(data), total=len(data)):
        prompt = example["clean_prompt"]
        ans_token = get_best_token_id(ft_model, prompt, example["clean_answer"])
        tokens = ft_model.to_tokens(prompt)
        
        # Retrieve the cached activations for this specific example
        current_acts = stored_activations[i]

        # Define the Patch Hook
        def cmap_patch_hook(ft_z, hook):
            layer = hook.layer()
            
            # Identify which heads in this layer are part of the circuit
            heads_to_patch = [h for (l, h) in target_heads if l == layer]
            
            # Get the saved Base Model activation (move back to GPU)
            base_z_layer = current_acts[layer].to(device)
            
            for head in heads_to_patch:
                # OVERWRITE FT activation with Base activation
                ft_z[:, :, head] = base_z_layer[:, :, head]
            
            return ft_z

        # Register hooks
        hooks = []
        for l in target_layers:
            hook_name = f"blocks.{l}.attn.hook_z"
            hooks.append((hook_name, cmap_patch_hook))

        # Run FT Model
        ft_logits = ft_model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        # Measure Performance
        prob = torch.softmax(ft_logits[0, -1], dim=-1)[ans_token].item()
        results.append(prob)
        
        # Cleanup
        del ft_logits
        torch.cuda.empty_cache()

    avg_cmap_score = sum(results) / len(results)
    print(f"\n[RESULTS]")
    print(f"--> Average CMAP Score: {avg_cmap_score:.4f}")
    
    if avg_cmap_score > 0.5:
        print("Conclusion: STRONG ENHANCEMENT (Mechanism Preserved)")
        print("  The Fine-Tuned model performs well even when using the Base Model's circuit heads.")
        print("  This suggests the circuit was NOT replaced, but rather amplified.")
    else:
        print("Conclusion: MECHANISM REPLACEMENT (New Circuit Learned)")
        print("  The Fine-Tuned model fails when using the Base Model's circuit heads.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--ft_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/test.jsonl")
    parser.add_argument("--circuit", type=str, default="circuit_metadata.json")
    args = parser.parse_args()
    
    run_cmap(args.base, args.ft_path, args.data, args.circuit)