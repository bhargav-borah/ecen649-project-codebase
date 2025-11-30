Checkpoint 1:
1. run gen_data.py to generate synthetic data for entity tracking ciruit discovery;
2. run circuit_discovery.py on the base model, here we use meta-llama/Llama-2-7b-hf model to find the circuit heads that have the top 20 significance in entity tracking test; we save the results in the file ciruit_metatdata.json;
3. train the model using different training mode: fft/lora/qlora/distill;
4. run the cmap_analysis.py to see if patch the circuit head in the original pretrained_model will impact the performance of the fine tuned models (with different fine tuning modes);
5. conclusion for now: 
  1) fft:
    python cmap_analysis.py --ft_path checkpoints/fft/final_model
    --> [Phase 1/2] Loading Base Model: meta-llama/Llama-2-7b-hf...
    `torch_dtype` is deprecated! Use `dtype` instead!
    Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████|    2/2 [00:00<00:00, 11.92it/s]
    WARNING:root:With reduced precision, it is advised to use `from_pretrained_no_processing` instead of        `from_pretrained`.
    WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
    Loaded pretrained model meta-llama/Llama-2-7b-hf into HookedTransformer
    --> Harvesting activations for 50 examples...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████|    50/50 [00:01<00:00, 26.93it/s]
    --> Unloading Base Model to free VRAM...
    --> [Phase 2/2] Loading Fine-Tuned Model: checkpoints/fft/final_model...
    Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████|    3/3 [00:00<00:00, 33.92it/s]
    WARNING:root:With reduced precision, it is advised to use `from_pretrained_no_processing` instead of    `from_pretrained`.
    WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
    Loaded pretrained model meta-llama/Llama-2-7b-hf into HookedTransformer
    --> Patching 20 circuit heads...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████|    50/50 [00:03<00:00, 15.40it/s]

    [RESULTS]
    --> Average CMAP Score: 0.1060
    Conclusion: MECHANISM REPLACEMENT (New Circuit Learned)
    The Fine-Tuned model fails when using the Base Model's circuit heads.
  
  2) lora:
    python cmap_analysis.py --ft_path checkpoints/lora/final_model
    --> [Phase 1/2] Loading Base Model: meta-llama/Llama-2-7b-hf...
    `torch_dtype` is deprecated! Use `dtype` instead!
    Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 202.82it/s]
    WARNING:root:With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`.
    WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
    Loaded pretrained model meta-llama/Llama-2-7b-hf into HookedTransformer
    --> Harvesting activations for 50 examples...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 29.01it/s]
    --> Unloading Base Model to free VRAM...
    --> [Phase 2/2] Loading Fine-Tuned Model: checkpoints/lora/final_model...
    Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 193.42it/s]
    WARNING:root:With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`.
    WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
    Loaded pretrained model meta-llama/Llama-2-7b-hf into HookedTransformer
    --> Patching 20 circuit heads...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.32it/s]

    [RESULTS]
    --> Average CMAP Score: 0.1065
    Conclusion: MECHANISM REPLACEMENT (New Circuit Learned)
    The Fine-Tuned model fails when using the Base Model's circuit heads.
6. further steps:
  1) increase the number of training epochs for finetunining to see if the conclusion will be changed. We hypothesize that with limited number of training epochs, the model will not be able to really understand the task but just memorize the results since entity tracking task here seems to simulate a lot with simple memorization of the masked words in sentences.
