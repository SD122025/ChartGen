"""
Description: 
    This script takes in the code generated with chart2code.py and performs code augmentation using a code large language model.

Usage:
    python augment_codes.py <code_dir> <output_dir> <K>

Arguments:
    --code_dir: Path to directory containing input `.md` code files.
    --output_dir: Path to directory where augmented `.md` files will be saved.
    --K: Number of augmented variants to generate per file.

Output:
    {root}/train_augmented_codes/generated_responseXXXXX_XX.md

Code model specification: 
    Model ID: mistralai/Codestral-22B-v0.1
    Quantization: 4-bit via HuggingFace BitsAndBytesConfig
    Docs: https://huggingface.co/mistralai/Codestral-22B-v0.1
"""

import argparse
import re
import sys
import glob
import random
import time
import os
from pathlib import Path
import torch
from tqdm import tqdm

from huggingface_hub import HfFolder
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate K augmented chart-plotting code variants per input file."
    )
    parser.add_argument("code_dir",    help="Directory with input `.md` files")
    parser.add_argument("output_dir",  help="Directory to save augmented `.md` files")
    parser.add_argument("K", type=int, help="Number of augmentations per file")
    return parser.parse_args()

def main():
    args = parse_args()

    seed = int(time.time() * 1000)
    random.seed(seed)

    # Load HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Please set HF_TOKEN in your environment (export HF_TOKEN=...)")
    HfFolder.save_token(hf_token)

    # Load tokenizer & 4-bit quantized model
    model_id = "mistralai/Codestral-22B-v0.1"
    tokenizer = MistralTokenizer.v3()
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    device = next(model.parameters()).device

    # Input/output directories
    input_dir  = Path(args.code_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather input code files (in raw .md or extracted .py formats)
    code_files = sorted(
        [*input_dir.glob("*.md"), *input_dir.glob("*.py")],key=lambda p: p.name
    )
    random.shuffle(code_files)

    # Predefined augmentation options
    chart_types = [
        "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Ring Chart", "Stem Plot",
        "Radar Chart", "Box Plot", "3D Bar Chart", "Histogram", "Treemap",
        "Rose Chart", "Bubble Chart", "Multi-Axes Chart", "Area Chart", "Heatmap",
        "Funnel Chart", "Candlestick Chart", "Stem Plot", "Violin Plot", "Tornado Chart"
    ]
    plotting_packages = [
        "matplotlib", "seaborn", "plotly", "bokeh", "altair", "plotnine (ggplot)", "pygal", "cufflinks"
    ]

    print("Found", len(code_files), "input code files.")
    # Process each input code file
    for src in tqdm(code_files, desc="Augmenting generated chart codes"):
        base = src.stem

        # Skip if first variant exists
        if (output_dir / f"{base}_00.md").exists():
            continue

        # Read the code from file
        content = src.read_text()
        if src.suffix == ".md":
            # extract only the fenced python blocks
            code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
        else:
            # .py files: treat the whole file as one code block
            code_blocks = [content]
        combined_code = "\n\n".join(code_blocks)

        used_chart_types = set()

        # Generate K augmentations
        for k in tqdm(range(args.K), desc="Generating K augmentations of the input code"):
            accepted = False
            # Prevent duplicates: give up to 3 attempts to generate a unique chart type
            for attempt in range(3):
                prev = ", ".join(used_chart_types) if used_chart_types else "None"

                prompt = f"""
                    Below is a Python code snippet that plots a chart:

                    {combined_code}

                    Your task is to generate a completely new version of this code with diverse modifications.
                    Please choose a new chart type from the following list: {', '.join(chart_types)}.
                    Please choose a new plotting library from the following list: {', '.join(plotting_packages)}.
                    You may choose any color scheme you like, feel free to change it. 
                    You can also change the chart topic (title) and data points.
                    At the very top of your output, include a comment in the following format:
                    # Variation: ChartType=<chart type>, Library=<plotting library>
                    Ensure that the generated code does not repeat any of the following chart types: {prev}.
                    The generated code must be valid Python, executable, and save the chart to a file.
                    Please output only the new Python code snippet enclosed in triple backticks (```).
                """

                # Encode the input prompt
                req = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
                tokens = tokenizer.encode_chat_completion(req).tokens
                input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
                input_length = input_ids.size(1)

                # Create an attention mask
                attention_mask = torch.ones_like(input_ids)

                # Generate the output
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1200,
                        pad_token_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                        do_sample=False,
                        use_cache=True
                    )

                # Extract generated output 
                all_output_tokens = output_ids[0].cpu().numpy().tolist()
                input_split = all_output_tokens[:input_length]
                output_split = all_output_tokens[input_length:]
                input_text = tokenizer.decode(input_split)
                result = tokenizer.decode(output_split)

                # Extract chart, package variation header
                variation_match = re.search(
                    r'#\s*Variation:\s*ChartType=(.*?),\s*Library=(.*)',
                    result
                )
                # If a variation header is found, extract the chart type and library
                if variation_match:
                    new_chart_type = variation_match.group(1).strip()
                    new_library = variation_match.group(2).strip()

                    # Check if the chart type is unique
                    if new_chart_type in used_chart_types:
                        # Chart type already used, so retry
                        continue
                    else:
                        # Accept the new, unique chart type
                        used_chart_types.add(new_chart_type)
                        accepted = True
                        break
                else:
                    # If no variation comment is found, accept the output to avoid infinite loops
                    new_chart_type = "Unknown ChartType"
                    new_library = "Unknown Library"
                    used_chart_types.add(new_chart_type)
                    accepted = True
                    break

            new_variation = f"ChartType={new_chart_type}, Library={new_library}"

            # Save the generated augmentation and given input code to an .md file
            new_variation = f"ChartType={new_chart_type}, Library={new_library}"
            out_path = output_dir / f"{base}_{k:02d}.md"
            with open(out_path, 'w') as f:
                f.write(f"## Source File: {src.name}\n")
                f.write(f'### Input:\n')
                f.write(input_text + '\n\n')
                f.write(f'### Generated Response (Variation: {new_variation}):\n')
                f.write(result + '\n\n')

            # Use the output as the input code for the next augmentation
            code_blocks = re.findall(r'```(.*?)```', result, re.DOTALL)
            combined_code = "\n\n".join(code_blocks)

if __name__ == "__main__":
    main()
