# Facebook Meta에서 개발한 Llama 2 Fine-tuning과 inference를 쉽게하기 위한 codebook

본 repository는 meta에서 개발한 [Llama 2](https://github.com/facebookresearch/llama)를 잘 사용하기 위해 만들어진 'llama-recipes'를 한국어로 번역한 것입니다. Llama 2를 domain에 맞게 fine-tuing하고, tuning한 model을 inference하는 방법도 확인할 수 있습니다. Fine-tuning과 inference를 하기 위해서는 Hugging Face model로 바꾸는 것이 필요로 하며, [여기](#model-conversion-to-hugging-face)에서 방법을 확인할 수 있습니다.

대학원에서 LLM과 Multimodal 관련해서 연구하는데, 제가 본 레포지토리를 사용하면서 얻었던 경험을 공유하고 싶어서 한글화 하고 있습니다. (~~도움이 되었다면 star 좀 흑흑..~~) 
이렇게 정리하는 것이 처음이라 궁금하거나, 이해가 안되는 부분 알려주시면 제가 경험해본 선에서 답변해드리겠습니다.

Llama 2모델은 상업적으로 사용하능하고, 더 자세한 부분은 [Llama 2 repo](https://github.com/facebookresearch/llama)를 확인해주세요.

# Table of Contents
1. [시작하기](#quick-start)
2. [Fine-tuning 하기](#fine-tuning)
    - [Single GPU](#single-gpu)
    - [Multi GPU One Node](#multiple-gpus-one-node)
    - [Multi GPU Multi Node](#multi-gpu-multi-node)
3. [Inference 하기](./docs/inference.md)
4. [Hugging face model로 변환하기](#model-conversion-to-hugging-face)
5. [Repository Organization](#repository-organization)
6. [License and Acceptable Use Policy](#license)



# Quick Start

[Llama 2 기초 jupyter notebook](quickstart.ipynb): Llama 2 model을 어떻게 fine-tuning하고 inference할 수 있는지 확인할 수 있는 notebook입니다. Fine-tuning은 [samsum](https://huggingface.co/datasets/samsum) 이라는 대화를 요약하는 데이터셋으로 진행을 합니다. PEFT와 LoRA를 사용해서 GPU 메모리가 24GB일 경우 7B와 13B 모델을 파인튜닝 시킬 수 있습니다. 
제가 실험한 환경은 RTX 3090 24GB 4대를 사용하였고, 13B model을 학습할 때 15GB 정도를 잡고 있고, 시간은 1시간 정도 걸린 것 같습니다.

**Note** All the setting defined in [config files](./configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

**Note** In case need to run PEFT model with FSDP, please make sure to use the PyTorch Nightlies.

**For more in depth information checkout the following:**

* [Single GPU Fine-tuning](./docs/single_gpu.md)
* [Multi-GPU Fine-tuning](./docs/mutli_gpu.md)
* [LLM Fine-tuning](./docs/LLM_finetuning.md)
* [Adding custom datasets](./docs/Dataset.md)
* [Inference](./docs/inference.md)
* [FAQs](./docs/FAQ.md)

## Requirements
To run the examples, make sure to install the requirements using

```bash

pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

# Fine-tuning

For fine-tuning Llama 2 models for your domain-specific use cases recipes for PEFT, FSDP, PEFT+FSDP have been included along with a few test datasets. For details see [LLM Fine-tuning](./docs/LLM_finetuning.md).

## Single and Multi GPU Finetune

If you want to dive right into single or multi GPU fine-tuning, run the examples below on a single GPU like A10, T4, V100, A100 etc.
All the parameters in the examples and recipes below need to be further tuned to have desired results based on the model, method, data and task at hand.

**Note:**
* To change the dataset in the commands below pass the `dataset` arg. Current options for dataset are `grammar_dataset`, `alpaca_dataset`and  `samsum_dataset`. A description of the datasets and how to add custom datasets can be found in [Dataset.md](./docs/Dataset.md). For  `grammar_dataset`, `alpaca_dataset` please make sure you use the suggested instructions from [here](./docs/single_gpu.md#how-to-run-with-different-datasets) to set them up.

* Default dataset and other LORA config has been set to `samsum_dataset`.

* Make sure to set the right path to the model in the [training config](./configs/training.py).

### Single GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. To run the command above make sure to pass the `peft_method` arg which can be set to `lora`, `llama_adapter` or `prefix`.

**Note** if you are running on a machine with multiple GPUs please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`

**Make sure you set [save_model](configs/training.py) in [training.py](configs/training.py) to save the model. Be sure to check the other training settings in [train config](configs/training.py) as well as others in the config folder as needed or they can be passed as args to the training script as well.**


### Multiple GPUs One Node:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model

```

Here we use FSDP as discussed in the next section which can be used along with PEFT methods. To make use of PEFT methods with FSDP make sure to pass `use_peft` and `peft_method` args along with `enable_fsdp`. Here we are using `BF16` for training.

### Fine-tuning using FSDP Only

If you are interested in running full parameter fine-tuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

### Multi GPU Multi Node:

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```
You can read more about our fine-tuning strategies [here](./docs/LLM_finetuning.md).


# Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Llama 2 model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under models/7B you can install all requirements and convert the checkpoint with:

```bash
## Install HuggingFace Transformers from source
pip install git+https://github.com/huggingface/transformers
cd transformers

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir models_hf/7B
```

# Repository Organization
This repository is organized in the following way:

[configs](configs/): Contains the configuration files for PEFT methods, FSDP, Datasets.

[docs](docs/): Example recipes for single and multi-gpu fine-tuning recipes.

[ft_datasets](ft_datasets/): Contains individual scripts for each dataset to download and process. Note: Use of any of the datasets should be in compliance with the dataset's underlying licenses (including but not limited to non-commercial uses)


[inference](inference/): Includes examples for inference for the fine-tuned models and how to use them safely.

[model_checkpointing](model_checkpointing/): Contains FSDP checkpoint handlers.

[policies](policies/): Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode).

[utils](utils/): Utility files for:

- `train_utils.py` provides training/eval loop and more train utils.

- `dataset_utils.py` to get preprocessed datasets.

- `config_utils.py` to override the configs received from CLI.

- `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.

- `memory_utils.py` context manager to track different memory stats in train loop.

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
