import os
import tqdm
import json
import copy
import math

import torch
import logging
import argparse

import numpy as np
from rouge import Rouge

import dataclasses
from xopen import xopen

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.llama import H2OLlamaForCausalLM

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")

    parser.add_argument("--model-name", type=str, default="")

    parser.add_argument("--enable_h2o_generation", action='store_true')
    parser.add_argument("--num_heavy_hitter_tokens", type=int, default=128)
    parser.add_argument("--num_window_length", type=int, default=256)

    parser.add_argument("--enable_position_rolling", action='store_true')

    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.enable_h2o_generation:
        config.num_heavy_hitter_tokens = args.num_heavy_hitter_tokens
        config.num_window_length = args.num_window_length
        config.enable_position_rolling = args.enable_position_rolling
        model = H2OLlamaForCausalLM.from_pretrained(model_name, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.half().eval().cuda()

    # loading inference data
    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['article']
            label = request['summary_gt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
            )

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            scores = rouge.get_scores(generate_text, label)[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])

            result['result'] = {
                "choices": [
                    {
                        "text": generate_text,
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)

    print('Average Rouge1: {:.6f}, Rouge-2: {:.6f}, Rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

