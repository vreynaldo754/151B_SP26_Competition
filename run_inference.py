# %% [markdown]
# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM (INT8 quantized)
# 4. Scoring responses against ground-truth answers.
# 5. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# %% [markdown]
# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# %% [markdown]
# ### Comment Out the cell below after first installation.

# %% [markdown]
# # Install uv
# !wget -qO- https://astral.sh/uv/install.sh | sh
# 
# # Create a virtual environment
# !uv venv .venv --seed
# 
# # Install dependencies — this is fast thanks to uv's parallel resolver
# !.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter
# 
# # Install Jupyter Kernel
# !.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"
# 
# print("Done. Restart the kernel before proceeding.")
# print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")

# %% [markdown]
# ### Run the cell below every time to activate the installed environment. 

# %% [markdown]
# # activate venv after installation. This needs to be run everytime.
# !source ./.venv/bin/activate

# %% [markdown]
# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# %%
import json
import os
import time

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/test.jsonl"
MAX_TOKENS  = 1024

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import torch
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#from vllm import LLM, SamplingParams
from tqdm import tqdm

# %% [markdown]
# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# %%
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))

# %% [markdown]
# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# %%
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "keep reasoning brief. "
    "Always re-check your answers and correct any errors before finalizing. "
    "Do not restate the problem or explain formulas. Compute directly. "
    "Output format rules: "
    "1. You may at most use 9 short reasoning lines. "
    "2. Always put your final answer inside \\boxed{} at the final line. "
    "3. The box must contain only the final answer. "
    "4. If the problem has multiple sub-answers, put answers in the same order, separated by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
    "5. Do not include units inside the box, unless the question explicitly asks for it. "
    "6. Prefer fractions or simple products, unless the question explicitly asks for it. "
    "7. Example Problem: 'Reduce the fraction ${\\frac{25}{40}}$. [ANS]'. Solution: '\\boxed{5/8}'"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Your task is to read the problem and answer choices below, then select the single best answer. "
    "keep reasoning brief. "
    "Always re-check your answers and correct any errors before finalizing. "
    "Do not restate the problem or explain formulas. Compute directly. "
    "Output format rules: "
    "1. The final line must output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "2. Do not put the option text inside the box. "
    "3. If your derived answer does not match any of the options, choose the closest option. Do not answer 'none of the above' unless it is an option. "
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = f"""
Solve the following problem:
{question}
Options:
{opts_text}
Solve concisely, using only the necessary steps. Choose the correct answer.
Place the final answer in this format:
\\boxed{{A}}
""".strip()
        return SYSTEM_PROMPT_MCQ, user_prompt
    num_ans = question.count("[ANS]")
    user_prompt = f"""
Solve the following problem:
{question}
There are {num_ans} blanks. Return exactly {num_ans} answers in the same order. 
Solve concisely, using only the necessary steps.
Place the answer at the final line in this format:
\\boxed{{answer1, answer2, ....}}
""".strip()
    return SYSTEM_PROMPT_MATH, user_prompt


# Verify with samples
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\n")

# %% [markdown]
# ## 5. Load Model with vLLM (for general case, vLLM is faster)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
# Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# %%
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# tokenizer.pad_token = tokenizer.eos_token

# llm = LLM(
#     model=MODEL_ID,
#     quantization="bitsandbytes",
#     load_format="bitsandbytes",
#     enable_prefix_caching=False,
#     gpu_memory_utilization=0.50,
#     max_model_len=16384,
#     trust_remote_code=True,
#     max_num_seqs=256,
#     max_num_batched_tokens=32768,
# )

# sampling_params = SamplingParams(
#     max_tokens=MAX_TOKENS,
#     temperature=0.6,
#     top_p=0.95,
#     top_k=20,
#     min_p=0.0,
#     presence_penalty=0.0,
#     repetition_penalty=1.0,
# )

# print("Model loaded.")

# %% [markdown]
# ## 5. Load Model with Transformers (alternative to vLLM for DataHub)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT4 quantization** via BitsAndBytes.  
# 
# Key parameters:
# - `load_in_4bit` — quantization strategy of INT4

# %%
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


# %%
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)


# %% [markdown]
# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# %% [markdown]
# ### Generate with vLLM

# %%
# # Build prompts for first 5 entries
# prompts = []
# for item in data[:5]:
#     system, user = build_prompt(item["question"], item.get("options"))
#     prompt_text = tokenizer.apply_chat_template(
#         [{"role": "system", "content": system},
#          {"role": "user",   "content": user}],
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     prompts.append(prompt_text)

# # Generate
# print(f"Generating responses for {len(prompts)} questions...")
# outputs = llm.generate(prompts, sampling_params=sampling_params)

# responses = [out.outputs[0].text.strip() for out in outputs]

# # Preview first 3
# for i in range(min(3, len(responses))):
#     print(f"\n── Response {i} (id={data[i].get('id')}) ──")
#     print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

# %% [markdown]
# ### Generate with Transformers (for Datahub)

# %%
tokenizer.padding_side = "left"
dataset = data
responses = []
BATCH_SIZE = 4

print(f"Generating responses for {len(dataset)} questions...")

#Create batches
for i in tqdm(range(0,len(dataset),BATCH_SIZE), desc = "Batch"):
    batch = dataset[i:i + BATCH_SIZE]

    prompts = []

    #build prompts
    for item in batch:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    #tokenize prompt with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=8192,
    ).to(llm.device)

    #generate
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.2 ,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    prompt_len = inputs["input_ids"].shape[1]

    #append only new responses
    for j in range(len(batch)):
        new_tokens = output_ids[j][prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

# Preview first 3
for i in range(min(3, len(responses))):
    print(f"\n── Response {i} (id={dataset[i].get('id')}) ──")
    print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

# %% [markdown]
# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# %%
def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(zip(dataset, responses), total=len(dataset), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": response,
        "correct":  correct,
    })

print(f"Scoring complete. {len(results)} results.")

# %% [markdown]
# ## 8. Summary
# 
# Print accuracy broken down by question type.

# %%
mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)

# %% [markdown]
# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# %%
SAVE_EVAL = False   # Set to False when running on the private test set

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for r in results:
        if SAVE_EVAL:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "response": r["response"], "correct": r["correct"]}
        else:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(results)} records to {out_path}")

# %%
import csv

OUTPUT_PATH = "results/submission.csv"

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

assert len(dataset) == len(responses), f"dataset has {len(dataset)} items but responses has {len(responses)}"

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"])
    writer.writeheader()

    for item, response in zip(dataset, responses):
        writer.writerow({
            "id": item["id"],
            "response": response
        })

print(f"Saved {len(responses)} records to {out_path}")

# %% [markdown]
# ## Next Steps
# 
# This notebook gives you a working baseline. Here are directions to improve your score:
# 
# - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
# - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
# - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
# 
# Good luck!


