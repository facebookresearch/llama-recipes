# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import yaml

from transformers import LlamaTokenizer

from llama_recipes.inference.model_utils import  load_llama_from_config

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):
    
    if not HF_model_path_or_name:
        try:
            file_name = 'train_params.yaml'
            # Combine the directory and file name to create the full path
            train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
            # Open the file
            with open(train_params_path, 'r') as file:
                # Load the YAML data
                data = yaml.safe_load(file)
    
                # Access the 'model_name' field
                HF_model_path_or_name = data.get('model_name')
    
                print(f"Model name: {HF_model_path_or_name}")
        except FileNotFoundError:
            print(f"The file {train_params_path} does not exist.")
            HF_model_path_or_name = input("Please enter the model name: ")
            print(f"Model name: {HF_model_path_or_name}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        
    #load the HF model definition from config
    model_def = load_llama_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)
