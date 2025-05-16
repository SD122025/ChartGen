# 📊 ChartGen  
Official codebase for “ChartGen: Scaling Chart Understanding via Code-Guided Synthetic Chart Generation”

ChartGen is a **two-stage, fully-automated pipeline** that  

1. **Reconstructs** real-world chart images into executable Python plotting scripts with a vision-language model (Phi-3.5-Vision-Instruct).  
2. **Augments** those scripts with a code-centric LLM (quantized Codestral-22B-v0.1), scaling a 13 K-image seed set into the **[ChartGen-200K dataset](https://huggingface.co/datasets/SD122025/ChartGen-200K)** spanning 27 chart types and 11 plotting libraries.  

<p align="center">
  <img src="figures/chartgen_pipeline.jpg" width="1200" alt="ChartGen pipeline">
</p>

## ✨  Key Highlights
| Stage | Model | What it does | Output |
|-------|-------|--------------|--------|
| **Reconstruct** | [`microsoft/Phi-3.5-vision-instruct`](https://huggingface.co/microsoft/phi-3.5-vision-instruct) | Converts each seed chart image into plotting code enclosed in triple-backtick fences | `train_generated_codes/*.md` |
| **Augment** | [`mistralai/Codestral-22B-v0.1`](https://huggingface.co/mistralai/Codestral-22B-v0.1) | Produces *K* stylistically diverse code variants (new chart type, library, color, data) | `train_augmented_codes/*.md` |

Running both stages on 13 K seed charts yields the **222.5 K image-code pairs** contained in the public **[ChartGen-200K dataset](https://huggingface.co/datasets/SD122025/ChartGen-200K)**. 
---

## 📂  Repository Layout
<pre>

ChartGen/
├── chartgen/            
│   ├── chart2code.py     # stage-1 script (image → code)
│   ├── code_augment.py   # stage-2 script (code  → diversified code)
│   └── extract_plot.py   # utils: extract code & render charts from raw model output
├── README.md
├── LICENSE
└── requirements.txt      # pinned dependency versions

</pre>

## ⚙️ Setup
```bash
# 1 Clone the repo
git clone https://github.com/<your-handle>/ChartGen.git
cd ChartGen

# 2 Create and activate an isolated environment
python -m venv .venv && source .venv/bin/activate     # macOS / Linux
# .\.venv\Scripts\Activate.ps1                        # Windows PowerShell

# 3 Install *GPU* or *CPU* wheels for PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only

# 4 Install the remaining Python packages
pip install -r requirements.txt

# 5 Authenticate with Hugging Face (needed for Codestral)
huggingface-cli login     # or: export HF_TOKEN=hf_your_token
```


Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
