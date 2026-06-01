# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

GPU type used: NVIDIA GeForce RTX 5060 Laptop GPU
Max_Tokens: 2048
Approximate total inference time: 8h 41mins

No extra model weights are required to be setup. The base QWEN3-4B model weights were used

A requirement.txt file has been created with all the dependencies to run run_inference.py