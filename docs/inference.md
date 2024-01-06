# Inference

For inference we have provided an [inference script](../examples/inference.py). Depending on the type of finetuning performed during training the [inference script](../examples/inference.py) takes different arguments.
To finetune all model parameters the output dir of the training has to be given as --model_name argument.
In the case of a parameter efficient method like lora the base model has to be given as --model_name and the output dir of the training has to be given as --peft_model argument.
Additionally, a prompt for the model in the form of a text file has to be provided. The prompt file can either be piped through standard input or given as --prompt_file parameter.

**Content Safety**
The inference script also supports safety checks for both user prompt and model outputs. In particular, we use two packages, [AuditNLG](https://github.com/salesforce/AuditNLG/tree/main) and [Azure content safety](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/).

**Note**
If using Azure content Safety, please make sure to get the endpoint and API key as described [here](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/) and add them as  the following environment variables,`CONTENT_SAFETY_ENDPOINT` and `CONTENT_SAFETY_KEY`.

Examples:

 ```bash
# Full finetuning of all parameters
cat <test_prompt_file> | python examples/inference.py --model_name <training_config.output_dir> --use_auditnlg
# PEFT method
cat <test_prompt_file> | python examples/inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --use_auditnlg
# prompt as parameter
python examples/inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg
 ```
The example folder contains test prompts for summarization use-case:
```
examples/samsum_prompt.txt
...
```

**Note**
Currently pad token by default in [Hugging Face Tokenizer is `None`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L110). We add the padding token as a special token to the tokenizer, which in this case requires to resize the token_embeddings as shown below:

```python
tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
model.resize_token_embeddings(model.config.vocab_size + 1)
```
Padding would be required for batch inference. In this this [example](../examples/inference.py), batch size = 1 so essentially padding is not required. However,We added the code pointer as an example in case of batch inference.

### Chat completion
The inference folder also includes a chat completion example, that adds built-in safety features in fine-tuned models to the prompt tokens. To run the example:

```bash
python examples/chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file examples/chat_completion/chats.json  --quantization --use_auditnlg

```
### Code Llama

Code llama was recently released with three flavors, base-model that support multiple programming languages, Python fine-tuned model and an instruction fine-tuned and aligned variation of Code Llama, please read more [here](https://ai.meta.com/blog/code-llama-large-language-model-coding/). Also note that the Python fine-tuned model and 34B models are not trained on infilling objective, hence can not be used for infilling use-case.

Find the scripts to run Code Llama [here](../examples/code_llama/), where there are two examples of running code completion and infilling.

**Note** Please find the right model on HF side [here](https://huggingface.co/codellama). 

Make sure to install Transformers from source for now

```bash

pip install git+https://github.com/huggingface/transformers

```

To run the code completion example:

```bash

python examples/code_llama/code_completion_example.py --model_name MODEL_NAME  --prompt_file examples/code_llama/code_completion_prompt.txt --temperature 0.2 --top_p 0.9

```

To run the code infilling example:

```bash

python examples/code_llama/code_infilling_example.py --model_name MODEL_NAME --prompt_file examples/code_llama/code_infilling_prompt.txt --temperature 0.2 --top_p 0.9

```

### Llama Guard

Llama Guard is a new experimental model that provides input and output guardrails for LLM deployments. For more details, please visit the main [repository](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard).

Find the inference script for Llama Guard [here](../examples/llama_guard/).

**Note** Please find the right model on HF side [here](https://huggingface.co/meta-llama/LlamaGuard-7b). 

Edit [inference.py](../examples/llama_guard/inference.py) to add test prompts for Llama Guard and execute it with this command:

`python examples/llama_guard/inference.py`

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up inference when used for batched inputs. This has been enabled in `optimum` library from Hugging Face as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
python examples/chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file examples/chat_completion/chats.json  --quantization --use_auditnlg --use_fast_kernels

python examples/inference.py --model_name <training_config.output_dir> --peft_model <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg --use_fast_kernels

```

## Loading back FSDP checkpoints

In case you have fine-tuned your model with pure FSDP and saved the checkpoints with "SHARDED_STATE_DICT" as shown [here](../src/llama_recipes/configs/fsdp.py), you can use this converter script to convert the FSDP Sharded checkpoints into Hugging Face checkpoints. This enables you to use the inference script normally as mentioned above.
**To convert the checkpoint use the following command**:

This is helpful if you have fine-tuned you model using FSDP only as follows:

```bash
torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16
```
Then convert your FSDP checkpoint to Hugging Face checkpoints using:
```bash
 python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path  PATH/to/FSDP/Checkpoints --consolidated_model_path PATH/to/save/checkpoints --HF_model_path_or_name PATH/or/HF/model_name

 # --HF_model_path_or_name specifies the HF Llama model name or path where it has config.json and tokenizer.json
 ```
By default, training parameter are saved in `train_params.yaml` in the path where FSDP checkpoints are saved, in the converter script we frist try to find the HugingFace model name used in the fine-tuning to load the model with configs from there, if not found user need to provide it.

Then run inference using:

```bash
python examples/inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> 

```

## Prompt Llama 2

As outlined by [this blog by Hugging Face](https://huggingface.co/blog/llama2#how-to-prompt-llama-2), you can use the template below to prompt Llama 2 chat models. Review the [blog article](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) for more information.

```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]

```

## Other Inference Options

Alternate inference options include:

[**vLLM**](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html):
To use vLLM you will need to install it using the instructions [here](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#installation).
Once installed, you can use the vllm/inference.py script provided [here](../examples/vllm/inference.py).

Below is an example of how to run the vLLM_inference.py script found within the inference folder.

``` bash
python examples/vllm/inference.py --model_name <PATH/TO/MODEL/7B>
```

[**TGI**](https://github.com/huggingface/text-generation-inference): Text Generation Inference (TGI) is another inference option available to you. For more information on how to set up and use TGI see [here](../examples/hf_text_generation_inference/README.md).

[Here](../demo_apps/llama-on-prem.md) is a complete tutorial on how to use vLLM and TGI to deploy Llama 2 on-prem and interact with the Llama API services.
