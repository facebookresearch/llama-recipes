# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import csv
import json
import time
import random
import threading
import numpy as np
import requests
import transformers
import torch

#imports for Azure content safety
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List


# Predefined inputs
with open('input.jsonl') as input:
    prompt_data = json.load(input)

with open('parameters.json') as parameters:
    params = json.load(parameters)

MAX_NEW_TOKENS = params["MAX_NEW_TOKENS"]
CONCURRENT_LEVELS = params["CONCURRENT_LEVELS"]
# Replace with your own deployment
MODEL_PATH = params["MODEL_PATH"]
MODEL_HEADERS = params["MODEL_HEADERS"]
SAFE_CHECK = params["SAFE_CHECK"]
# Threshold for tokens per second below which we deem the query to be slow
THRESHOLD_TPS = params["THRESHOLD_TPS"] 
# Replace with your own tokenizer 
TOKENIZER_PATH = params["TOKENIZER_PATH"] 
RANDOM_PROMPT_LENGTH = params["RANDOM_PROMPT_LENGTH"]
TEMPERATURE = params["TEMPERATURE"]
TOP_P = params["TOP_P"]
# Add your model endpoints here, specify the port number. You can acquire the endpoint when creating a on-prem server like vLLM.
# Group of model endpoints - Send balanced requests to each endpoint for batch maximization.  
MODEL_ENDPOINTS = params["MODEL_ENDPOINTS"]

#Get number of GPUs on this instance
if torch.cuda.is_available():
    NUM_GPU = torch.cuda.device_count()
else:
    print("No available GPUs")


# This tokenizer is downloaded from Azure model catalog for each specific models. The main purpose is to decode the reponses for token calculation
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Select vocabulary that is longer than 2 tokens (closer to real words) and close to the English (not foolproof)
vocab = [token for token in tokenizer.get_vocab().keys() if len(token) > 2 and all(ord(c) < 128 for c in token)]

def generate_random_prompt(num_tokens):
    generated_tokens_count = 0
    selected_tokens = ""
    while generated_tokens_count < num_tokens:
        selected_tokens += random.choice(vocab)
        selected_tokens += " "
        generated_tokens_count = len(tokenizer.encode(selected_tokens))

    return selected_tokens

PROMPT = generate_random_prompt(RANDOM_PROMPT_LENGTH)
num_token_input_prompt = len(tokenizer.encode(PROMPT))
print(f"Number of token for input prompt: {num_token_input_prompt}")


# Azure content safety analysis
def analyze_prompt(input):
    start_time = time.time()

    # Obtain credentials
    key = "" #Add your AZURE_CONTENT_SAFETY_KEY
    endpoint = "" #Add your AZURE_CONTENT_SAFETY_ENDPOINT

    # Create a content safety client
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    # Create request
    request = AnalyzeTextOptions(text=input)

    # Analyze prompt
    try:
        response = client.analyze_text(request)
    except HttpResponseError as e:
        print("prompt failed due to content safety filtering.")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        print(e)
        raise

    analyze_end_time = time.time()
    # The round trip latency for using Azure content safety check
    analyze_latency = (analyze_end_time - start_time) * 1000


# Simple round-robin to dispatch requests into different containers
executor_id = 0
lock = threading.Lock()

def generate_text() -> Tuple[int, int]:
    headers = MODEL_HEADERS
    payload = {
        "model" : MODEL_PATH,
        "messages" : [
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        "stream" : False,
        "temperature" : TEMPERATURE,
        "top_p" : TOP_P,
        "max_tokens" : MAX_NEW_TOKENS
    }

    start_time = time.time()

    if(SAFE_CHECK):
        # Function to send prompts for safety check. Add delays for request round-trip that count towards overall throughput measurement.
        # Expect NO returns from calling this function. If you want to check the safety check results, print it out within the function itself.
        analyze_prompt(PROMPT)
        # Or add delay simulation as below for real world situation
        # time.sleep(random.uniform(0.3, 0.4))

    lock.acquire()
    global executor_id
    if executor_id != len(MODEL_ENDPOINTS)-1:
        executor_id += 1
        endpoint_id = executor_id
    else:
        executor_id = 0
        endpoint_id = executor_id
    lock.release()

    response = requests.post(MODEL_ENDPOINTS[endpoint_id], headers=headers, json=payload)

    if(SAFE_CHECK):
        # Function to send prompts for safety check. Add delays for request round-trip that count towards overall throughput measurement.
        # Expect NO returns from calling this function. If you want to check the safety check results, print it out within the function itself.
        analyze_prompt(PROMPT)
        # Or add delay simulation as below for real world situation
        # time.sleep(random.uniform(0.3, 0.4))

    end_time = time.time()
    # Convert to ms
    latency = (end_time - start_time) * 1000 

    if response.status_code != 200:
        raise ValueError(f"Error: {response.content}")
    output = json.loads(response.content)["choices"][0]["message"]["content"]

    token_count = len(tokenizer.encode(output))
    return latency, token_count


def evaluate_performance(concurrent_requests: int) -> Tuple[float, float, float, float, float, float, float, List[float]]:
    latencies = []
    total_output_tokens = 0
    output_tokens_per_second_each_request = []
    start_time = time.time()

    # Init multi-thread execution 
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        future_to_req = {executor.submit(generate_text): i for i in range(concurrent_requests)}
        for future in as_completed(future_to_req):
            latency, token_count = future.result()
            latencies.append(latency)
            total_output_tokens += token_count
            # Calculate tokens per second for this request
            tokens_per_sec = token_count / (latency / 1000)
            output_tokens_per_second_each_request.append(tokens_per_sec)

    end_time = time.time()
    total_time = end_time - start_time
    # RPS (requests per second)
    rps = concurrent_requests / total_time  
    # Overall tokens per second
    output_tokens_per_second_overall = total_output_tokens / total_time  
    input_tokens_per_second_overall = (num_token_input_prompt * concurrent_requests) / total_time
    output_tokens_per_second_per_gpu = output_tokens_per_second_overall / NUM_GPU
    input_tokens_per_second_per_gpu = input_tokens_per_second_overall / NUM_GPU
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)

    # Count the number of requests below the token-per-second threshold
    below_threshold_count = sum(1 for tps in output_tokens_per_second_each_request if tps < THRESHOLD_TPS)
    output_tokens_per_second_per_request = sum(output_tokens_per_second_each_request)/len(output_tokens_per_second_each_request)

    return p50_latency, p99_latency, rps, output_tokens_per_second_overall, output_tokens_per_second_per_gpu, input_tokens_per_second_overall, input_tokens_per_second_per_gpu, output_tokens_per_second_per_request, below_threshold_count



# Print markdown
print("| Number of Concurrent Requests | P50 Latency (ms) | P99 Latency (ms) | RPS | Output Tokens per Second | Output Tokens per Second per GPU | Input Tokens per Second | Input Tokens per Second per GPU |Average Output Tokens per Second per Request | Number of Requests Below Threshold |")
print("|-------------------------------|------------------|------------------|------------------|-------------------|---------------------------|---------------------|------------------------|-------------------------------------- | ---------------------------------- |")

# Save to file
csv_file = "performance_metrics.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Number of Concurrent Requests", "P50 Latency (ms)", "P99 Latency (ms)", "RPS", "Output Tokens per Second", "Output Tokens per Second per GPU", "Input Tokens per Second", "Input Tokens per Second per GPU", "Average Output Tokens per Second per Request"])

    for level in CONCURRENT_LEVELS:
        p50_latency, p99_latency, rps, output_tokens_per_second_overall, output_tokens_per_second_per_gpu, input_tokens_per_second_overall, input_tokens_per_second_per_gpu, output_tokens_per_second_per_request, below_threshold_count = evaluate_performance(level)
        print(f"| {level} | {p50_latency:.2f} | {p99_latency:.2f} | {rps:.2f} | {output_tokens_per_second_overall:.2f} | {output_tokens_per_second_per_gpu:.2f} | {input_tokens_per_second_overall:.2f} | {input_tokens_per_second_per_gpu:.2f} | {output_tokens_per_second_per_request:.2f} | {below_threshold_count:.2f} |")
        writer.writerow([level, round(p50_latency, 2), round(p99_latency, 2), round(rps, 2), round(output_tokens_per_second_overall, 2), round(output_tokens_per_second_per_gpu, 2), round(input_tokens_per_second_overall, 2), round(input_tokens_per_second_per_gpu, 2), round(output_tokens_per_second_per_request, 2)])
