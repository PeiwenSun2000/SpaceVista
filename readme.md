# <img src=".asset/logo_v2.png" width="10%"> **SpaceVista**: All-Scale Visual Spatial Reasoning from $mm$ to $km$

Keywords: ![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) ![Text-to-Audio](https://img.shields.io/badge/Task-Text--to--Audio-red) ![Spatial Audio](https://img.shields.io/badge/Task-Spatial--Audio-red) 

The official repo for SpaceVista: All-Scale Visual Spatial Reasoning from $mm$ to $km$.

<div style='display:flex; gap: 0.25rem; '>
Community Contribution: 
<!-- <a href='http://143.89.224.6:2436/'><img src='https://img.shields.io/badge/Gradio-Demo_nl-blue'></a><a href='http://143.89.224.6:2437/'><img src='https://img.shields.io/badge/Gradio-Demo_attr-blue'></a> -->
<a href='https://huggingface.co/datasets/SpaceVista/Data-Preview'><img src='https://img.shields.io/badge/Data Preview-Huggingface-yellow'></a>
<a href='https://github.com/PeiwenSun2000/SpaceVista'><img src='https://img.shields.io/badge/Code-Github-blue'></a>
<a href='https://arxiv.org/abs/2410.10676'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## Outlines
- [💥 News 💥]()
- [👀 About SpaceVista]()
- [📊 SpaceVista-1M Dataset]()
- [🏆 Usage]()
- [📝 Evaluation]()
- [📜 License]()
- [🤝 Contributors]()

## 💥 News 💥
[2025.10.10] Our preview SFT code base is released for preview. <a href='https://github.com/PeiwenSun2000/SpaceVista'><img src='https://img.shields.io/badge/Code-Github-blue'></a>.

[2025.10.10] Our preview 100K subset of SpaceVista-1M is now available at <a href='https://huggingface.co/datasets/SpaceVista/Data-Preview'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'></a>.

[2025.10.10] Our initial paper is now accessible at <a href='https://arxiv.org/abs/2410.10676'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>.

## Overall Structure

*  Dataset: [Preview 100K subset of SpaceVista-1M](https://huggingface.co/datasets/SpaceVista/Data-Preview) <a href='https://huggingface.co/datasets/SpaceVista/Data-Preview'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'></a>
*  SFT training: [SFT code for SpaceVista](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open/tree/main/models) <a href='https://github.com/PeiwenSun2000/SpaceVista'><img src='https://img.shields.io/badge/Code-Github-blue'></a>.

- [ ]  Release the full SpaceVista-1M dataset
- [ ]  Release the GRPO codebase and checkpoints
- [ ]  Release the SpaceVista-Bench benchmark

## SpaceVista

<p align="center">
    <img src=".asset/teaser.jpg" width="60%"> <br>
</p>

Spatial reasoning is the ability to perceive, interpret, and act across spatial scales, from millimeter-sized components to distant aerial scenes. All-scale spatial reasoning is fundamental to next-generation intelligent systems and supports diverse applications: mm sensing for advanced manufacturing, cm and m perception for embodied agents, 10m operation for autonomous driving, and 100m for drone based sensing.
Despite progress, existing work shows clear limitations in both model design and dataset coverage. Current scene perception research mostly targets indoor scenes, narrow object classes, and limited spatial ranges, and lacks training paradigms engineered for end to end, cross scale reasoning. SpaceVista addresses this gap by presenting the first systematic optimization across both data and model dimensions to enable robust, full scene spatial reasoning.

# Requirements

Development for the repo is done in Python 3.10.18

This code base is adapted from [LLaMA-factory](https://github.com/hiyouga/LLaMA-Factory), [R1-V](https://github.com/StarsfieldAI/R1-V), [VG-LLM](https://github.com/LaVi-Lab/VG-LLM) and [Easy-R1](https://github.com/hiyouga/EasyR1). Sincere thanks to the engineers for their great work.

We use the light venv package for python enviroment. (Do not use other tools like conda at the same time)
```
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

# Dataset Usages

We provide the dataset in ShareGPT format, along with up to 32 extracted frames.

You may download the orginal mp4 video from the source.

# Model Gallery

The model will be released soon after the sensitivity check.

| Model           | 🤗 HF      | Detail                                            |
|-----------------|----------|---------------------------------------------------|
| To Be Updated    | To Be  Updated               | To Be  Updated

# Usage

Before everything, sincere apology for some part of our code is still hard code. We are actively seeking for easy usage of this repo.


## SFT training:

To generate audio from a text prompt using our pretrained model:

1. Download the pretrained [Qwen2.5VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) model and [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
2. (Optional) Download the pretrained [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) model.
3. Change the `dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` and `vggt/ckpt` path in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` to your path.

```
# source the same environment
cd sft
# (Optional checking) `training_load = True` in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

sed -i 's/self\.training_load = False/self\.training_load = True/g' \
"../.venv/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

llamafactory-cli train examples/train_full/qwen2_5_vl_spatial_full_sft_video_dinov3.yaml
```


## SFT training w/. Expert:


Preliminary: If you train the model with an additional adapter for DINOv3, you need to use a roughly trained SFT model as the pre-trained base. Otherwise, PEFT will only save the LoRA weights.
1. Training each expert on the SFT model
      - (Optional checking) `training_load = False` in `../dependency/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
```
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

```
# source the same environment
cd sft

llamafactory-cli train examples/train_lora/qwen2_5_vl_spatial_full_sft_video_expert.yaml
```

## RL w/. GRPO

To be updated


```
# to be updated
```

# Evaluation

1. Please be sure to use the venv provided.
2. Change the benchmark path to your path
```
DATASET_CONFIGS = {
        "vsibench": {
            "dataset_path": "./vsi-bench/test-00000-of-00001.parquet",
            "video_dir": "./vsi-bench",
            "evaluation_fn": ...,
            "metric_fn": ...,
        },
        "mmsibench": {
            "dataset_path": "./MMSI_Bench.parquet",
            "video_dir": "", # Not needed as images are in the parquet file
            "evaluation_fn": ...,
            "metric_fn": ...,
        },
        "spacevista": {
            "dataset_path": "./unified_qa.jsonl", # will be released soon.
            "video_dir": "./frames/all", # will be released soon.
            "evaluation_fn": ...,
            "metric_fn": ...,
        },
        "sparbench": {
            "dataset_path": ["./SPAR-Bench/data/test-00000-of-00004.parquet","./SPAR-Bench/data/test-00001-of-00004.parquet",\
                "./SPAR-Bench/data/test-00002-of-00004.parquet","./SPAR-Bench/data/test-00003-of-00004.parquet"],
            "video_dir": "",
            "evaluation_fn": ...,
            "metric_fn": ...,
        },
        "stibench": {
            "dataset_path": "./sti-bench/qa.parquet",
            "video_dir": "", # Not needed as images are in the parquet file
            "evaluation_fn": ...,
            "metric_fn": ...,
        }
    }

```

3. Use this script to evaluate a model on a chosen dataset. Example: 

```
cd eval

sed -i 's/self\.training_load = True/self\.training_load = False/g' \
"../.venv/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

# source the same environment

# vsibench
python eval_multi_model_mp.py --model_path /path/to/model --dataset vsibench --output_dir ./eval_results --gpu_ids 0,1 --num_processes 4 --num_frames 32 --batch_size 1

# spacevista
python eval_multi_model_mp.py --model_path /path/to/model --dataset spacevista --output_dir ./eval_results --gpu_ids 0,1 --num_processes 4 --num_frames 32 --batch_size 1

```

Required: `--model_path` (checkpoint or folder) and `--dataset` (one of: vsibench, mmsibench, spacevista, sparbench, stibench). Optional: `--output_dir` (results dir, default ./eval_results), `--gpu_ids` (comma-separated GPU IDs), `--num_processes` (parallel workers), `--num_frames` (frames per video), `--batch_size` (inference batch size), `--debug` (enable quick run), and `--debug_size` (samples used when debug is on).


# Reference

If you find this repo useful, please cite our papers:

```
@article{sun2024both,
  title={...},
  author={Sun, Peiwen ...},
  journal={...},
  year={2024}
}
```
