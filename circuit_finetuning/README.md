HOW TO RUN THE CODE:
1. run gen_data.py to generate synthetic data for entity tracking ciruit discovery;
2. run circuit_discovery.py on the base model, here we use meta-llama/Llama-2-7b-hf model to find the circuit heads that have the top 20 significance in entity tracking test; we save the results in the file ciruit_metatdata.json;
3. train the model using different training mode: fft/lora/qlora/distill;
4. run the cmap_analysis.py to see if patch the circuit head in the original pretrained_model will impact the performance of the fine tuned models (with different fine tuning modes)
