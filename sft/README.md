# SFT Training for SpaceVista

This repo follows [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with changes for model and dataloader.

## Requirements

Development for the repo is done in Python 3.10.18

This code base is adapted from [LLaMA-factory](https://github.com/hiyouga/LLaMA-Factory), [R1-V](https://github.com/StarsfieldAI/R1-V), [VG-LLM](https://github.com/LaVi-Lab/VG-LLM) and [Easy-R1](https://github.com/hiyouga/EasyR1). Sincere thanks to the engineers for their great work.

We use the light venv package for the Python environment. (Do not use other tools like conda at the same time)
```bash
git clone 
cd SpaceVista

# pip install uv

uv venv -p python3.10.18
source .venv/bin/activate
UV_HTTP_TIMEOUT=600 uv pip install -r requirements_sft.txt --no-deps -i http://mirrors.aliyun.com/pypi/simple/

# For flash_attn
MAX_JOBS=64 uv pip install flash_attn==2.7.1.post4 --no-build-isolation -i http://mirrors.aliyun.com/pypi/simple/

ln -s "$(pwd)/dependency/transformers" ".venv/lib/python3.10/site-packages/transformers"
```

## SFT Training

Change the form of the dataset first. Note, this might be simplified in the future

```bash
cd dataset
python flatten.py -i your_path/meta.json -o your_path/meta_flatten.json
```

1. Download the pretrained [Qwen2.5VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) model and [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
2. (Optional) Download the pretrained [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) model.
3. Change the `dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` and `vggt/ckpt` path in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` to your path.

```bash
# source the same environment
cd sft

# (Optional checking) `training_load = True` in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

sed -i 's/self\.training_load = False/self\.training_load = True/g' \
"../.venv/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

llamafactory-cli train examples/train_full/qwen2_5_vl_spatial_full_sft_video_dinov3.yaml
```

## SFT Training w/. Expert

Preliminary: If you train the model with an additional adapter for DINOv3, you need to use a roughly trained SFT model as the pre-trained base. Otherwise, PEFT will only save the LoRA weights.

1. Training each expert on the SFT model
   - (Optional checking) `training_load = False` in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

```bash
# source the same environment
cd sft

sed -i 's/self\.training_load = True/self\.training_load = False/g' \
"../.venv/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_spacevista_cross_outdoor.yaml
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_spacevista_cross_table.yaml
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_spacevista_cross_tabletop.yaml
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_spacevista_cross_indoor.yaml
```

2. Change the path of each expert in `sft/src/llamafactory/model/adapter.py` to the checkpoint saved on the above step
   - (Optional checking) `training_load = False` in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

```bash
# source the same environment
cd sft

llamafactory-cli train examples/train_lora/qwen2_5_vl_spatial_full_sft_video_expert.yaml
```

## RL w/. GRPO

To be updated.
