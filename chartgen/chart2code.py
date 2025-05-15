"""
Description: 
    Load a split of the seed dataset from HuggingFace, iterate through its chart images, and, using a vision-language model, generate Python code that redraws each chart.

Usage:
    python chart2code.py <output_dir>

Arguments:
    output_dir: Directory to save the seed images and generated code.

Output:
    - {root}/train_input_images/image_00000.png 
    - {root}/train_generated_codes/generated_response00000.md

Vision-language moddel specification: 
    Model ID: microsoft/Phi-3.5-vision-instruct
    Docs: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Seed Dataset: 
    Dataset ID: HuggingFaceM4/ChartQA
    Docs: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
"""

import os
import sys
import random
import time
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoProcessor

# Initialize the model (using HF AutoClass)
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)

# Define the prompt
messages = [ # make sure to add image placeholder to the prompt
    {"role": "user", "content": "<|image_1|>\nPlease take a look at this chart image. Consider you are a data visualization expert, and generate python code that perfectly reconstructs this chart image. Make sure to redraw both the data points and the overall semantics and style of the chart as best as possible. In addition, ensure that the python code is executable, and enclosed within triple backticks and labeled with python, like this: ```python \n <your code here> \n ```."},
]

# Load the chart image dataset
dataset = load_dataset("HuggingFaceM4/ChartQA")
train_split = dataset['train']
#val_split = dataset['val']
#test_split = dataset['test']
#combined_dataset = concatenate_datasets([train_split, val_split, test_split])
combined_dataset = train_split

# Create output directories if they don't exist
root = sys.argv[1]
img_root = os.path.join(root,'train_input_images')
code_root = os.path.join(root,'train_generated_codes')
os.makedirs(img_root, exist_ok=True)
os.makedirs(code_root, exist_ok=True)

def main():
    # set random seed
    seed = int(time.time() * 1000)
    random.seed(seed)
    # Iterate through the chart images
    work = list(KeyDataset(combined_dataset, "image")[::2])
    ids = list(range(len(work)))
    random.shuffle(ids)

    for idx in tqdm(ids, desc="Processing chart images"): # skip every other item because chart images in the datasets are duplicates
        item = work[idx]
        # Save the input image
        image_path = os.path.join(img_root, f'image_{idx:05d}.png')
        if os.path.exists(image_path):
            continue
        item.save(image_path)
        
        # Process the input
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [item], return_tensors="pt").to("cuda:0")
        
        # Query the model
        generation_args = {
        "max_new_tokens": 2000,
        "temperature": 0.0,
        "do_sample": False,
        }
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        # Extract model output
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Save the generated output to a separate .md file for each iteration
        output_file_path = os.path.join(code_root, f'generated_response{idx:05d}.md')
        with open(output_file_path, 'w') as f:
            f.write(f'## Image {idx}\n')
            f.write(f'![Image {idx}](../input_images/image_{idx}.png)\n\n')
            f.write(f'### Generated Response:\n')
            f.write(response + '\n\n')

if __name__ == "__main__":
    main()
