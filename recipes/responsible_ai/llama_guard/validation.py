# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Validating the output of Llama Guard quantized and unquantized
#
# This notebook aims to show how to validate Llama Guard performance on a given dataset. The script loads a given dataset and evaluates each prompt individually against Llama Guard. To evaluate performance, we calculate the averate precision of the binary classification for a given prompt. Llama Guard can be run usgin Meta provided weights or directly from Hugging Face. 
#
# ## Dataset format
# The dataset should be in a `jsonl` file, with an object per line, following this structure:
# ```
# {
#     "prompt": "user_input",
#     "generation": "model_response",
#     "label": "good/bad", 
#     "unsafe_content": ["O1"]
# }
# ```
#
#
# The `label` has a `good` or `bad` value to define if the content is considered safe or unsafe, respectively.
#
# The `unsafe_content` field contains a list of the categories the prompt is violating.
#
#

# +
from typing import List, Tuple
from enum import Enum
from pathlib import Path
from sklearn.metrics import average_precision_score

import json
import numpy as np
import time
import torch
# -

from inference import llm_eval, pytorch_llm_eval, AgentType


# +
class Type(Enum):
    HF = "HF"
    PYTORCH = "PYTORCH"

def format_prompt(entry, agent_type: AgentType):
    prompts = []
    if agent_type == AgentType.USER:
        prompts = [entry["prompt"]]
    else:
        prompts = [entry["prompt"], entry["generation"]]

    return {
        "prompt": prompts,
        "agent_type": agent_type,
        "label": entry["label"],
        "unsafe_content": entry["unsafe_content"],
        "idx": entry["idx"]
    }

def validate_agent_type(value):
    try:
        return AgentType(value)
    except ValueError:
        raise ValueError(f"Invalid AgentType. Choose from: {[agent_type.value for agent_type in AgentType]}")



def run_validation(jsonl_file_path, agent_type, type: Type, load_in_8bit: bool = True, load_in_4bit: bool = False, ckpt_dir = None):

    input_file_path = Path(jsonl_file_path)

    agent_type = validate_agent_type(agent_type)
    
    # Preparing prompts
    prompts: List[Tuple[List[str], AgentType, str, str, str]] = []
    with open(jsonl_file_path, "r") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            
            # Format prompt and add to list
            prompt = format_prompt(entry, agent_type)
            prompts.append(prompt)

    
    # Executing evaluation
    start = time.time()
    if type is Type.HF:
        llm_eval(prompts, load_in_8bit=load_in_8bit, load_in_4bit=True, logprobs=True)
    else:
        pytorch_llm_eval(prompts, ckpt_dir, True)
    
    end = time.time()
    print(f"evaluation executed in {end - start} seconds")
        
    average_precision = parse_logprobs(prompts, type)
    print(f"average precision {average_precision:.2%}")

    
# -

# ## Average presicion
#
# This function calculates the average precision score for a set of prompts based on their log probabilities and labels. 
#
# The `prompts` contain the logprobs calculated for each result by Llama Guard when evaluating the prompts or prompt and generation. 
#
# The `type` is used to identify if the logprobs are comming from a Hugging Face model or plain pytorch model.
#
# The logprob is converted back into probability by exponentiating it (`np.exp`)
#
# The probability for `unsafe` when the result is `safe` is calculated using the heuristic 1 - `safe`. As this is a banary classification problem, it should be close to the real value for `unsafe`.
#
# The average presicion is calculated with the binary labels from the expected value for each prompt or prompt/generation pair and the probability of generating the unsafe token for each.
#

def parse_logprobs(prompts, type: Type):
    positive_class_probs = []
    for prompt in prompts:
        prob = np.exp(prompt["logprobs"][0]) if type is Type.PYTORCH else np.exp(prompt["logprobs"][0][1])
        if "unsafe" in prompt["result"]:
            positive_class_probs.append(prob)
        else:
            # Using heuristic 1 - `safe` probability to calculate the probability of a non selected token in a binary classification
            positive_class_probs.append(1 - prob)
        
    binary_labels = [1 if prompt["label"] == "bad" else 0 for prompt in prompts]

    return average_precision_score(binary_labels, positive_class_probs)


# **Note:** If you get a `Address already in use` error when running with a local llama guard model, change the port by setting the environment variable to a new one. e.g.: `os.environ["MASTER_PORT"] = "29501"` For more details, check `Inference.ipynb`. 
#

# +
prompts_file = "prompts.jsonl"

# When the type is pytorch, there is no quantization options
run_validation(prompts_file, AgentType.USER, Type.PYTORCH, ckpt_dir = "path/to/llama_guard/")
# -

# clean up the cache from running the previous validation
torch.cuda.empty_cache()

# +
# Login to HF to access the model, if necessary
# from huggingface_hub import login
# login()
# -

# By default, load_in_8bit is true. To run unquantized or with 4bit quantization, set load_in_8bit to False and load_in_4bit to true
run_validation(prompts_file, AgentType.USER, Type.HF, load_in_8bit = False, load_in_4bit = True)



