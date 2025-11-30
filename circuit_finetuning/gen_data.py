import json
import random
import os
import argparse

def generate_entity_tracking_data(num_samples=10000, seed=42):
    """
    Generates the 'Box Content' dataset used in Reference [1].
    Updated with ONE-SHOT prompting to force Llama-2 to activate its reasoning circuit.
    """
    random.seed(seed)
    
    # Vocabulary
    objects = ["ball", "key", "book", "pen", "cup", "hat", "dog", "cat", "ring", "fish"]
    ONE_SHOT_PREFIX = "Box 1 contains the sun. Box 2 contains the moon. Box 3 contains the star. Box 2 contains the moon. "
    
    data = []
    
    for _ in range(num_samples):
        # 1. Setup 3 distinct boxes with 3 distinct objects
        current_objects = random.sample(objects, 3) 
        
        # 2. Create the statements
        stmts = []
        for i in range(3):
            stmts.append(f"Box {i+1} contains the {current_objects[i]}.")
        
        # 3. Shuffle context
        context_stmts = stmts.copy()
        random.shuffle(context_stmts)
        context = " ".join(context_stmts)
        
        # 4. Select Target
        target_idx = random.randint(0, 2)
        target_box = target_idx + 1
        target_obj = current_objects[target_idx]
        
        # CLEAN PROMPT
        clean_prompt = f"{ONE_SHOT_PREFIX}{context} Box {target_box} contains the"
        clean_answer = f" {target_obj}"
        
        # 5. Create CORRUPTED
        distractor_idx = (target_idx + 1) % 3 
        distractor_obj = current_objects[distractor_idx]
        
        corrupt_context_parts = []
        for clean_sent in context_stmts:
            b_num = int(clean_sent.split("Box ")[1].split(" ")[0])
            b_idx = b_num - 1
            
            if b_idx == target_idx:
                new_obj = distractor_obj
            elif b_idx == distractor_idx:
                new_obj = target_obj
            else:
                new_obj = current_objects[b_idx]
            
            corrupt_context_parts.append(f"Box {b_num} contains the {new_obj}.")
            
        corrupt_context = " ".join(corrupt_context_parts)
        
        # CORRUPT PROMPT
        corrupt_prompt = f"{ONE_SHOT_PREFIX}{corrupt_context} Box {target_box} contains the"
        corrupt_answer = f" {distractor_obj}"
        
        data.append({
            "clean_prompt": clean_prompt,
            "clean_answer": clean_answer,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_answer": corrupt_answer,
            "task_prompt": clean_prompt + clean_answer 
        })
        
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--samples", type=int, default=12000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    raw_data = generate_entity_tracking_data(args.samples)
    
    train_data = raw_data[:10000]
    test_data = raw_data[10000:]
    
    with open(f"{args.output_dir}/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(f"{args.output_dir}/test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(train_data)} train and {len(test_data)} test samples in {args.output_dir}/")