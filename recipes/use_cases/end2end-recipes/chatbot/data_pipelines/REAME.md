## Data Preprocessing Steps

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder.

In this case we have an example of [Llama 2 Getting started guide](https://llama.meta.com/get-started/) and other llama related documents such Llama2, Purple Llama, Code Llama papers along with Llama FAQ. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here.

### Step 2 : Prepare data (Q&A pairs)

The idea here is to use Llama 70B using OctoAI APIs, to create question and answer (Q&A) pair datasets from these documents, this APIs could be replaced by any other API from other providers or alternatively using your on prem solutions such as the [TGI](../../../examples/hf_text_generation_inference/) or [VLLM](../../../examples/vllm/). Here we will use the prompt in the [./config.yaml] to instruct the model on the expected format and rules for generating the Q&A pairs. This is only one way to handle this which is a popular method but beyond this any other preprocessing routine that help us making the Q&A pairs works. 


**NOTE** The generated data by these APIs/ the model needs to be vetted to make sure about the quality.

```bash
export OCTOAI_API_TOKEN="OCTOAI_API_TOKEN"
python generate_question_answers.py 
```

**NOTE** You need to be aware of your  RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. In our case we had to process each document at a time. Then merge all the Q&A `json` files to make our dataset. We aimed for a specific number of Q&A pairs per document anywhere between 50-100. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

### Step 2 : Prepare dataset for fine-tuning Llama 2 Chat model

Here, as we want to fine-tune a chatbot model so its preferred to start with Llama 2 Chat model which already is instruction fine-tuned to serve as an assistant and further fine-tuned it for our Llama related data.


### Step 3: Run the training

```bash
torchrun --nnodes 1 --nproc_per_node 1  examples/finetuning.py  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ./peft-7b-quantized  --num_epochs 1 --batch_size 1 --dataset "custom_dataset" --custom_dataset.file "examples/llama_dataset.py"  --run_validation False  --custom_dataset.data_path './dataset.json'
```