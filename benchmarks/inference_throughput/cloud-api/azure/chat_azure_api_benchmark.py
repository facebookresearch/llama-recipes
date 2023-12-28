# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import csv
import json
import time
import urllib.request
import numpy as np
import transformers
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List

with open('input.jsonl') as input:
    prompt_data = json.load(input)

# Prompt data stored in json file. Choose from number of tokens - 5, 25, 50, 100, 500, 1k, 2k.
PROMPT = prompt_data["25"] 

with open('parameters.json') as parameters:
    params = json.load(parameters)

MAX_NEW_TOKEN = params["MAX_NEW_TOKEN"]
CONCURRENT_LEVELS = params["CONCURRENT_LEVELS"]
# Threshold for tokens per second below which we deem the query to be slow
THRESHOLD_TPS = params["THRESHOLD_TPS"] 
# Default Llama 2 tokenizer, replace with your own tokenizer 
TOKENIZER_PATH = params["TOKENIZER_PATH"] 
TEMPERATURE = params["TEMPERATURE"]
TOP_P = params["TOP_P"]
# Model endpoint provided with API provider 
MODEL_ENDPOINTS = params["MODEL_ENDPOINTS"]
API_KEY = params["API_KEY"]
SYS_PROMPT = params["SYS_PROMPT"]


# This tokenizer is downloaded from Azure model catalog for each specific models. The main purpose is to decode the reponses for token calculation
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

num_token_input_prompt = len(tokenizer.encode(PROMPT))
print(f"Number of token for input prompt: {num_token_input_prompt}")


def generate_text() -> Tuple[int, int]:

    #Configure payload data sending to API endpoint
    payload = {"messages":[
                {"role":"system", "content": SYS_PROMPT},
                {"role":"user", "content": PROMPT}], 
            "max_tokens": MAX_NEW_TOKEN,
            "temperature": TEMPERATURE,
            "top_p" : TOP_P,
            "stream": "False"
    }
    body = str.encode(json.dumps(payload))
    url = MODEL_ENDPOINTS
    api_key = API_KEY
    if not api_key:
        raise Exception("API Key is missing")
    
    headers = {'Content-Type':'application/json', 'Authorization':(api_key)}
    req = urllib.request.Request(url, body, headers)
    token_count = 0
    output = ""
    start_time = time.time()
    # Send request
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        output = json.loads(result)["choices"][0]["message"]["content"]
        
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

    end_time = time.time()
    # Convert to ms
    latency = (end_time - start_time) * 1000  
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
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)

    # Count the number of requests below the token-per-second threshold
    below_threshold_count = sum(1 for tps in output_tokens_per_second_each_request if tps < THRESHOLD_TPS)
    output_tokens_per_second_per_request = sum(output_tokens_per_second_each_request)/len(output_tokens_per_second_each_request)

    return p50_latency, p99_latency, rps, output_tokens_per_second_overall, input_tokens_per_second_overall, output_tokens_per_second_per_request, below_threshold_count



# Print markdown
print("| Number of Concurrent Requests | P50 Latency (ms) | P99 Latency (ms) | RPS | Output Tokens per Second | Input Tokens per Second | Average Output Tokens per Second per Request | Number of Requests Below Threshold |")
print("|-------------------------------|------------------|------------------|-----|--------------------------|-------------------------|----------------------------------------------|------------------------------------|")

# Save to file
csv_file = "performance_metrics.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Number of Concurrent Requests", "P50 Latency (ms)", "P99 Latency (ms)", "RPS", "Output Tokens per Second", "Input Tokens per Second", "Average Output Tokens per Second per Request"])

    for level in CONCURRENT_LEVELS:
        p50_latency, p99_latency, rps, output_tokens_per_second_overall, input_tokens_per_second_overall, output_tokens_per_second_per_request, below_threshold_count = evaluate_performance(level)
        print(f"| {level} | {p50_latency:.2f} | {p99_latency:.2f} | {rps:.2f} | {output_tokens_per_second_overall:.2f} | {input_tokens_per_second_overall:.2f} | {output_tokens_per_second_per_request:.2f} | {below_threshold_count:.2f} |")
        writer.writerow([level, round(p50_latency, 2), round(p99_latency, 2), round(rps, 2), round(output_tokens_per_second_overall, 2), round(input_tokens_per_second_overall, 2), round(output_tokens_per_second_per_request, 2)])
