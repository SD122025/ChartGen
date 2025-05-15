# ChartGen

> Scalable **chart-to-code** data generation & benchmarking, built on the NeurIPS 2025 paper “ChartGen: Scaling Chart Understanding Via Code-Guided Synthetic Chart Generation”.

<p align="center">
  <img src="assets/chartgen_pipeline.jpg" width="1200" alt="ChartGen pipeline">
</p>

## ✨ Key features
| What | Why it matters |
|------|----------------|
| **Two-stage pipeline** (VLM → LLM) | Converts seed chart images into *executable* Python plotting scripts, then iteratively augments them for scale & diversity. |
| **Huge synthetic corpus** | 222.5 K image-code pairs covering **27 chart types** and **11 plotting libraries**. |
| **Benchmark for chart derendering** | 4 K-sample test set + GPT-4o judging protocol let you measure code fidelity, visual similarity and execution rate. |
| **Library-agnostic** | Works with *matplotlib, seaborn, plotly, altair, bokeh, plotnine, *pygal* out of the box. |

## 🗺️ Repository layout

<pre>

chartgen/
├── chartgen/             # Core Python library
│   ├── pipeline.py       # Two-stage generation pipeline
│   ├── prompts.py        # Prompt templates for VLM & LLM
│   ├── evaluate.py       # GPT-4o-based evaluation scripts
│   └── …                 # (other helpers)
├── data/
│   └── chartnet/         # (Optional) pre-generated dataset download script
├── examples/             # End-to-end notebooks & minimal demos
└── README.md

</pre>

<p align="center">
  <img src="assets/data_generation_process.jpg" width="640" alt="Chart redrawing & augmentation process">
</p>




Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
