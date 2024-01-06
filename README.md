# Llama 2 Fine-tuning / Inference Recipes, Examples, Benchmarks and Demo Apps

**[Update Dec. 28, 2023] We added support for Llama Guard as a safety checker for our example inference script and also with standalone inference with an example script and prompt formatting. More details [here](./examples/llama_guard/README.md). For details on formatting data for fine tuning Llama Guard, we provide a script and sample usage [here](./src/llama_recipes/data/llama_guard/README.md).**

**[Update Dec 14, 2023] We recently released a series of Llama 2 demo apps [here](./demo_apps). These apps show how to run Llama (locally, in the cloud, or on-prem),  how to use Azure Llama 2 API (Model-as-a-Service), how to ask Llama questions in general or about custom data (PDF, DB, or live), how to integrate Llama with WhatsApp and Messenger, and how to implement an end-to-end chatbot with RAG (Retrieval Augmented Generation).**

The 'llama-recipes' repository is a companion to the [Llama 2 model](https://github.com/facebookresearch/llama). The goal of this repository is to provide examples to quickly get started with fine-tuning for domain adaptation and how to run inference for the fine-tuned models. For ease of use, the examples use Hugging Face converted versions of the models. See steps for conversion of the model [here](#model-conversion-to-hugging-face).

In addition, we also provide a number of demo apps, to showcase the Llama 2 usage along with other ecosystem solutions to run Llama 2 locally, in the cloud, and on-prem.

Llama 2 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios. In order to help developers address these risks, we have created the [Responsible Use Guide](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf). More details can be found in our research paper as well. For downloading the models, follow the instructions on [Llama 2 repo](https://github.com/facebookresearch/llama).


# Table of Contents
1. [Quick start](#quick-start)
2. [Model Conversion](#model-conversion-to-hugging-face)
3. [Fine-tuning](#fine-tuning)
    - [Single GPU](#single-gpu)
    - [Multi GPU One Node](#multiple-gpus-one-node)
    - [Multi GPU Multi Node](#multi-gpu-multi-node)
4. [Inference](./docs/inference.md)
5. [Demo Apps](#demo-apps)
6. [Repository Organization](#repository-organization)
7. [License and Acceptable Use Policy](#license)

# Quick Start

[Llama 2 Jupyter Notebook](./examples/quickstart.ipynb): This jupyter notebook steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum). The notebook uses parameter efficient finetuning (PEFT) and int8 quantization to finetune a 7B on a single GPU like an A10 with 24GB gpu memory.

# Installation
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

## Install with pip
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```

## Install with optional dependencies
Llama-recipes offers the installation of optional packages. There are three optional dependency groups.
To run the unit tests we can install the required dependencies with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[auditnlg]
```
Optional dependencies can also be combines with [option1,option2].

## Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```
For development and contributing to llama-recipes please install all optional dependencies:
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .[tests,auditnlg,vllm]
```

⚠️ **Note** ⚠️  Some features (especially fine-tuning with FSDP + PEFT) currently require PyTorch nightlies to be installed. Please make sure to install the nightlies if you're using these features following [this guide](https://pytorch.org/get-started/locally/).

**Note** All the setting defined in [config files](src/llama_recipes/configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

**For more in depth information checkout the following:**

* [Single GPU Fine-tuning](./docs/single_gpu.md)
* [Multi-GPU Fine-tuning](./docs/multi_gpu.md)
* [LLM Fine-tuning](./docs/LLM_finetuning.md)
* [Adding custom datasets](./docs/Dataset.md)
* [Inference](./docs/inference.md)
* [FAQs](./docs/FAQ.md)

# Where to find the models?

You can find Llama 2 models on Hugging Face hub [here](https://huggingface.co/meta-llama), where models with `hf` in the name are already converted to Hugging Face checkpoints so no further conversion is needed. The conversion step below is only for original model weights from Meta that are hosted on Hugging Face model hub as well.

# Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Llama 2 model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under models/7B you can install all requirements and convert the checkpoint with:

```bash
## Install Hugging Face Transformers from source
pip freeze | grep transformers ## verify it is version 4.31.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

# Fine-tuning

For fine-tuning Llama 2 models for your domain-specific use cases recipes for PEFT, FSDP, PEFT+FSDP have been included along with a few test datasets. For details see [LLM Fine-tuning](./docs/LLM_finetuning.md).

## Single and Multi GPU Finetune

If you want to dive right into single or multi GPU fine-tuning, run the examples below on a single GPU like A10, T4, V100, A100 etc.
All the parameters in the examples and recipes below need to be further tuned to have desired results based on the model, method, data and task at hand.

**Note:**
* To change the dataset in the commands below pass the `dataset` arg. Current options for integrated dataset are `grammar_dataset`, `alpaca_dataset`and  `samsum_dataset`. Additionally, we integrate the OpenAssistant/oasst1 dataset as an [example for a custom dataset](./examples/custom_dataset.py).  A description of how to use your own dataset and how to add custom datasets can be found in [Dataset.md](./docs/Dataset.md#using-custom-datasets). For  `grammar_dataset`, `alpaca_dataset` please make sure you use the suggested instructions from [here](./docs/single_gpu.md#how-to-run-with-different-datasets) to set them up.

* Default dataset and other LORA config has been set to `samsum_dataset`.

* Make sure to set the right path to the model in the [training config](src/llama_recipes/configs/training.py).

* To save the loss and perplexity metrics for evaluation, enable this by passing `--save_metrics` to the finetuning script. The file can be plotted using the [plot_metrics.py](./examples/plot_metrics.py) script, `python examples/plot_metrics.py --file_path path/to/metrics.json`

### Single GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name /path_of_model_folder/7B --output_dir path/to/save/PEFT/model

```

Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. To run the command above make sure to pass the `peft_method` arg which can be set to `lora`, `llama_adapter` or `prefix`.

**Note** if you are running on a machine with multiple GPUs please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`

**Make sure you set `save_model` parameter to save the model. Be sure to check the other training parameter in [train config](src/llama_recipes/configs/training.py) as well as others in the config folder as needed. All parameter can be passed as args to the training script. No need to alter the config files.**


### Multiple GPUs One Node:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /path_of_model_folder/7B --fsdp_config.pure_bf16 --output_dir path/to/save/PEFT/model

```

Here we use FSDP as discussed in the next section which can be used along with PEFT methods. To make use of PEFT methods with FSDP make sure to pass `use_peft` and `peft_method` args along with `enable_fsdp`. Here we are using `BF16` for training.

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up the fine-tuning job. This has been enabled in `optimum` library from Hugging Face as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /path_of_model_folder/7B --fsdp_config.pure_bf16 --output_dir path/to/save/PEFT/model --use_fast_kernels
```

### Fine-tuning using FSDP Only

If you are interested in running full parameter fine-tuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --use_fast_kernels

```

### Fine-tuning using FSDP on 70B Model

If you are interested in running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --model_name /path_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

### Multi GPU Multi Node:

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```
You can read more about our fine-tuning strategies [here](./docs/LLM_finetuning.md).

# Demo Apps
This folder contains a series of Llama2-powered apps:
* Quickstart Llama deployments and basic interactions with Llama
1. Llama on your Mac and ask Llama general questions
2. Llama on Google Colab
3. Llama on Cloud and ask Llama questions about unstructured data in a PDF
4. Llama on-prem with vLLM and TGI
5. Llama chatbot with RAG (Retrieval Augmented Generation)
6. Azure Llama 2 API (Model-as-a-Service)

* Specialized Llama use cases:
1. Ask Llama to summarize a video content
2. Ask Llama questions about structured data in a DB
3. Ask Llama questions about live data on the web
4. Build a Llama-enabled WhatsApp chatbot

# Benchmarks
This folder contains a series of benchmark scripts for Llama 2 models inference on various backends:
1. On-prem - Popular serving frameworks and containers (i.e. vLLM)
2. (WIP) Cloud API - Popular API services (i.e. Azure Model-as-a-Service)
3. (WIP) On-device - Popular on-device inference solutions on Android and iOS (i.e. mlc-llm, QNN)
4. (WIP) Optimization - Popular optimization solutions for faster inference and quantization (i.e. AutoAWQ)

# Repository Organization
This repository is organized in the following way:
[benchmarks](./benchmarks): Contains a series of benchmark scripts for Llama 2 models inference on various backends.

[configs](src/llama_recipes/configs/): Contains the configuration files for PEFT methods, FSDP, Datasets.

[docs](docs/): Example recipes for single and multi-gpu fine-tuning recipes.

[datasets](src/llama_recipes/datasets/): Contains individual scripts for each dataset to download and process. Note: Use of any of the datasets should be in compliance with the dataset's underlying licenses (including but not limited to non-commercial uses)

[demo_apps](./demo_apps): Contains a series of Llama2-powered apps, from quickstart deployments to how to ask Llama questions about unstructured data, structured data, live data, and video summary.

[examples](./examples/): Contains examples script for finetuning and inference of the Llama 2 model as well as how to use them safely.

[inference](src/llama_recipes/inference/): Includes modules for inference for the fine-tuned models.

[model_checkpointing](src/llama_recipes/model_checkpointing/): Contains FSDP checkpoint handlers.

[policies](src/llama_recipes/policies/): Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode).

[utils](src/llama_recipes/utils/): Utility files for:

- `train_utils.py` provides training/eval loop and more train utils.

- `dataset_utils.py` to get preprocessed datasets.

- `config_utils.py` to override the configs received from CLI.

- `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.

- `memory_utils.py` context manager to track different memory stats in train loop.

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
