# Llama-On-Prem-Benchmark
This folder contains code to run inference benchmark for Llama 2 models on-prem with popular serving frameworks.
The benchmark will focus on overall inference **throughput** for running containers on one instance (single or multiple GPUs) that you can acquire from cloud service providers such as Azure and AWS. You can also run this benchmark on local laptop or desktop.  
We support benchmark on these serving framework:
* [vLLM](https://github.com/vllm-project/vllm)


# vLLM - Getting Started
To get started, we first need to deploy containers on-prem as a API host. Follow the guidance [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/llama-on-prem.md#setting-up-vllm-with-llama-2) to deploy vLLM on-prem.
Note that in common scenario which overall throughput is important, we suggest you prioritize deploying as many model replicas as possible to reach higher overall throughput and request-per-second (RPS), comparing to deploy one model container among multiple GPUs for model parallelism. Additionally, as deploying multiple model replicas, there is a need for a higher level wrapper to handle the load balancing which here has been simulated in the benchmark scripts.  
For example, we have an instance from Azure that has 8xA100 80G GPUs, and we want to deploy the Llama 2 70B chat model, which is around 140GB with FP16. So for deployment we can do:
* 1x70B model parallel on 8 GPUs, each GPU RAM takes around 17.5GB for loading model weights.
* 2x70B models each use 4 GPUs, each GPU RAM takes around 35GB for loading model weights.
* 4x70B models each use 2 GPUs, each GPU RAM takes around 70GB for loading model weights. (Preferred configuration for max overall throughput. Note that you will have 4 endpoints hosted on different ports and the benchmark script will route requests into each model equally)

Here are examples for deploying 2x70B chat models over 8 GPUs with vLLM.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --disable-log-requests --port 8000 
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --disable-log-requests --port 8001 
```
Once you have finished deployment, you can use the command below to run benchmark scripts in a separate terminal. 

```
python chat_vllm_benchmark.py
```
<!-- markdown-link-check-disable -->
If you are going to use [Azure AI content check](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety), then you should install dependencies as shown below in your terminal:
<!-- markdown-link-check-enable -->
```
pip install azure-ai-contentsafety azure-core
```
Besides chat models, we also provide benchmark scripts for running pretrained models for text completion tasks. To better simulate the real traffic, we generate configurable random token prompt as input. In this process, we select vocabulary that is longer than 2 tokens so the generated words are closer to the English, rather than symbols.
However, random token prompts can't be applied for chat model benchmarks, since the chat model expects a valid question. By feeding random prompts, chat models rarely provide answers that is meeting our ```MAX_NEW_TOKEN``` requirement, defeating the purpose of running throughput benchmarks. Hence for chat models, the questions are copied over to form long inputs such as for 2k and 4k inputs.   
To run pretrained model benchmark, follow the command below.
```
python pretrained_vllm_benchmark.py
```

