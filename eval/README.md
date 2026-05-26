# SpaceVista-Bench Evaluation

Evaluate VLM models on the [SpaceVista-Bench](https://arxiv.org/abs/2510.09606) dataset via any OpenAI-compatible API (OpenRouter, OpenAI, POE, etc.). The benchmark tests spatial reasoning across 4 scales: mm (TinyTabletop), cm (Tabletop), m (Indoor), <km (Outdoor).

## Environment Setup

```bash
# 1. Install dependencies (no GPU/torch required for API evaluation)
pip install openai pillow numpy tqdm pandas

# 2. Set API key as environment variable
export API_KEY="your-api-key-here"
```

## Dataset

```bash
# Download SpaceVista-Bench
huggingface-cli download SpaceVista/SpaceVista-Bench --repo-type dataset --local-dir ./SpaceVista-Bench
```

## Quick Start

Two evaluation scripts are provided depending on API input support:

- `evaluate_api.py` — sends sampled frames as individual images. For APIs that support **image (frame)** input.
- `evaluate_api_video.py` — synthesizes sampled frames into an MP4 video. For APIs that support **video** input. Requires `ffmpeg`.

### For APIs that support frame input

```bash
python evaluate_api.py --model qwen/qwen2.5-vl-72b-instruct
```

### For APIs that support video input

```bash
python evaluate_api_video.py --model qwen/qwen2.5-vl-72b-instruct
```

### Use a different API provider

Override `--base_url` and `--api_key` to point to any OpenAI-compatible endpoint:

```bash
# OpenRouter (default)
python evaluate_api.py --model qwen/qwen2.5-vl-72b-instruct

# OpenAI
python evaluate_api.py --model gpt-4o \
    --base_url https://api.openai.com/v1 \
    --api_key $OPENAI_API_KEY

# POE
python evaluate_api.py --model gpt-4o \
    --base_url https://api.poe.com/v1 \
    --api_key $POE_API_KEY
```

## Arguments

### Common arguments (both scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | `bench_all_clean_corrected.json` | Path to the benchmark JSON file |
| `--model` | `qwen/qwen2.5-vl-72b-instruct` | Model name |
| `--api_key` | `$API_KEY` env var | API key |
| `--base_url` | `https://openrouter.ai/api/v1` | API base URL |
| `--output_dir` | `./eval_results_api` | Output directory |
| `--num_frames` | `16` | Number of frames to sample per entry |
| `--max_image_size` | `448` | Resize longest edge to this size |
| `--max_workers` | `8` | Parallel API threads (1 = sequential) |
| `--max_retries` | `3` | Retry count on API failure |
| `--max_tokens` | `8196` | Max output tokens per API call |
| `--resume` | None | Path to existing JSONL to resume from |
| `--category` | None | Filter to one category: Indoor/Outdoor/TinyTabletop/Tabletop |
| `--debug` | False | Enable debug mode |
| `--debug_size` | `20` | Number of entries in debug mode |
| `--metrics_only` | None | Recompute metrics from existing JSONL (no API calls) |

### Video-only argument (`evaluate_api_video.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--fps` | `2` | Frames per second for the synthesized video |

## Resume Interrupted Runs

```bash
python evaluate_api.py --resume ./eval_results_api/<model_name>/<timestamp>/results_<model_name>.jsonl
```

## Recompute Metrics Only

```bash
python evaluate_api.py --metrics_only ./path/to/results.jsonl
```

## Output Format

Results are saved as JSONL (one JSON object per line):

```json
{
  "id": 42,
  "input_type": "image_bbox_text",
  "category": "Indoor",
  "scale": "m",
  "original_type": "OBJECT_COUNTING",
  "question": "How many objects like the one enclosed by the red bounding box...",
  "ground_truth": "3",
  "predicted_answer": "<think>...</think>\n<answer>3</answer>",
  "answer_type": "numerical",
  "num_frames_sent": 16
}
```

Metrics are printed as a summary table and saved as `*_metrics.json`.

## Annotation Handling

Entries with `input_type` other than `image_text` have visual annotations drawn on frames before sending to the API:

- **image_point_text**: Red circle drawn at the specified point
- **image_bbox_text**: Red bounding box drawn around the target region
- **image_mask_text**: Semi-transparent red mask overlay on the target object

## Reference

```bibtex
@article{sun2025spacevista,
  title={SpaceVista: All-Scale Visual Spatial Reasoning from mm to km},
  author={Sun, Peiwen and Lang, Shiqiang and Wu, Dongming and Ding, Yi and Feng, Kaituo and Liu, Huadai and Ye, Zhen and Liu, Rui and Liu, Yun-Hui and Wang, Jianan and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2510.09606},
  year={2025}
}
```
