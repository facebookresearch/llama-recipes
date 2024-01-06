# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

import torch
from vllm import LLM
from vllm import LLM, SamplingParams
from accelerate.utils import is_xpu_available

if is_xpu_available():
    torch.xpu.manual_seed(42)
else:
    torch.cuda.manual_seed(42)

torch.manual_seed(42)

def load_model(model_name, tp_size=1):

    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm

def main(
    model,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    while True:
        if user_prompt is None:
            user_prompt = input("Enter your prompt: ")
            
        print(f"User prompt:\n{user_prompt}")

        print(f"sampling params: top_p {top_p} and temperature {temperature} for this inference request")
        sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)
        

        outputs = model.generate(user_prompt, sampling_params=sampling_param)
   
        print(f"model output:\n {user_prompt} {outputs[0].outputs[0].text}")
        user_prompt = input("Enter next prompt (press Enter to exit): ")
        if not user_prompt:
            break

def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    model = load_model(model_name, tp_size)
    main(model, max_new_tokens, user_prompt, top_p, temperature)

if __name__ == "__main__":
    fire.Fire(run_script)
