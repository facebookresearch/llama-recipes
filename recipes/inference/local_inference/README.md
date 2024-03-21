# Local Inference

For local inference we have provided an [inference script](inference.py). Depending on the type of finetuning performed during training the [inference script](inference.py) takes different arguments.
To finetune all model parameters the output dir of the training has to be given as --model_name argument.
In the case of a parameter efficient method like lora the base model has to be given as --model_name and the output dir of the training has to be given as --peft_model argument.
Additionally, a prompt for the model in the form of a text file has to be provided. The prompt file can either be piped through standard input or given as --prompt_file parameter.

**Content Safety**
The inference script also supports safety checks for both user prompt and model outputs. In particular, we use [AuditNLG](https://github.com/salesforce/AuditNLG/tree/main) and [Azure content safety](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/). Additionally, we offer Llama Guard with the default categories as an optional content safety checker as well.

**Note**
If using Azure content Safety, please make sure to get the endpoint and API key as described [here](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/) and add them as  the following environment variables,`CONTENT_SAFETY_ENDPOINT` and `CONTENT_SAFETY_KEY`.

**Note on Llama Guard**
Use this command for testing with a quantized Llama model, modifying the values accordingly:

`python inference.py --model_name <path_to_regular_llama_model> --prompt_file <path_to_prompt_file> --quantization --enable_llamaguard_content_safety`


Examples:

 ```bash
# Full finetuning of all parameters
cat <test_prompt_file> | python inference.py --model_name <training_config.output_dir> --use_auditnlg
# PEFT method
cat <test_prompt_file> | python inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --use_auditnlg
# prompt as parameter
python inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg
 ```
The  folder contains test prompts for summarization use-case:
```
samsum_prompt.txt
...
```

**Note**
Currently pad token by default in [HuggingFace Tokenizer is `None`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L110). We add the padding token as a special token to the tokenizer, which in this case requires to resize the token_embeddings as shown below:

```python
tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
model.resize_token_embeddings(model.config.vocab_size + 1)
```
Padding would be required for batch inference. In this this [example](inference.py), batch size = 1 so essentially padding is not required. However,We added the code pointer as an example in case of batch inference.


## Chat completion
The inference folder also includes a chat completion example, that adds built-in safety features in fine-tuned models to the prompt tokens. To run the example:

```bash
python chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file chat_completion/chats.json  --quantization --use_auditnlg

```

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up inference when used for batched inputs. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
python chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file chat_completion/chats.json  --quantization --use_auditnlg --use_fast_kernels

python inference.py --model_name <training_config.output_dir> --peft_model <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg --use_fast_kernels

```

## Loading back FSDP checkpoints

In case you have fine-tuned your model with pure FSDP and saved the checkpoints with "SHARDED_STATE_DICT" as shown [here](../../../src/llama_recipes/configs/fsdp.py), you can use this converter script to convert the FSDP Sharded checkpoints into HuggingFace checkpoints. This enables you to use the inference script normally as mentioned above.
**To convert the checkpoint use the following command**:

This is helpful if you have fine-tuned you model using FSDP only as follows:

```bash
torchrun --nnodes 1 --nproc_per_node 8  recipes/finetuning/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16
```
Then convert your FSDP checkpoint to HuggingFace checkpoints using:
```bash
 python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path  PATH/to/FSDP/Checkpoints --consolidated_model_path PATH/to/save/checkpoints --HF_model_path_or_name PATH/or/HF/model_name

 # --HF_model_path_or_name specifies the HF Llama model name or path where it has config.json and tokenizer.json
 ```
By default, training parameter are saved in `train_params.yaml` in the path where FSDP checkpoints are saved, in the converter script we frist try to find the HugingFace model name used in the fine-tuning to load the model with configs from there, if not found user need to provide it.

Then run inference using:

```bash
python inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> 

```