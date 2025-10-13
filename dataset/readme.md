# **SpaceVista**: All-Scale Visual Spatial Reasoning from $mm$ to $km$

<p align="center">
          &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/SpaceVista/Data-Preview">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2510.09606">Paper</a> &nbsp&nbsp | &nbsp&nbsp ⚙️ <a href="https://github.com/PeiwenSun2000/SpaceVista">Github</a> &nbsp&nbsp | 🖥️ <a href="https://peiwensun2000.github.io/mm2km/">Home Page</a>&nbsp&nbsp
</p>


Peiwen Sun*, Shiqiang Lang*, Dongming Wu, Yi Ding, Kaituo Feng, Huadai Liu, Zhen Ye, Rui Liu, Yun-Hui Liu, Jianan Wang, Xiangyu Yue

The data preview for SpaceVista: All-Scale Visual Spatial Reasoning from $mm$ to $km$.

# Important!

This is a preview of SpaceVista, containing only a subset of tasks and scenes. As you can see, the data format is **not yet unified**, and the meta JSON keys remain inconsistent. We’ve provided a reasonable preview here—please stay tuned for the full, up-to-date release.

By the way, we would **not** recommend you train your model with your own dataloader. The format of this dataset needs to be **refined completely**. As the final version may necessitate a complete redesign of your dataloader, we **have provided and will provide** the dataloaders for this version and future versions.

# What is **All-Scale Spatial Reasoning**

<img src=".asset/teaser.jpg" width="50%">

Spatial reasoning is the ability to **perceive, interpret, and act** across spatial scales, from millimeter-sized components to distant aerial scenes. All-scale spatial reasoning is fundamental to next-generation intelligent systems and supports diverse applications: mm sensing for advanced manufacturing, cm and m perception for embodied agents, 10m operation for autonomous driving, and 100m for drone based sensing.
Despite progress, existing work shows clear limitations in both model design and dataset coverage. Current scene perception research mostly targets indoor scenes, narrow object classes, and limited spatial ranges, and lacks training paradigms engineered for end-to-end, cross-scale reasoning. SpaceVista addresses this gap by presenting the **first systematic solution** across both data and model dimensions to enable robust, full scene spatial reasoning.

# Data Construction


<img src=".asset/dataset.jpg" width="50%">

The limited data and performance constraints in existing models necessitate the creation of a dataset with all-scale spatial context. We propose **SpaceVista-1M**, a **diverse, real-world, all-scale** reasoning dataset, as the **first** to the best of our knowledge. SpaceVista-1M primarily comprises diverse spatial reasoning question–answer pairs, with rich semantic (category, rationale), 2D (mask, box, point), and 3D (depth, camera parameters, point cloud) annotations, obtained either natively or through processing. The construction pipeline in the above figure follows the step-by-step procedure of preparing, transforming, and generating to obtain an all-scale dataset by integrating specialized models.

# Data Format

The overall data format roughly follows the **shareGPT format** with the minor change in extra_info for efficiency.

- **messages:** user instruction and expected structured output format.
- **videos:** lists of image frame paths per sample. (Only extracted frames are shown here. For full-resolution viewing, please refer to the original source video.)
- **extra_info:** task type, prompt annotations (points/bboxes/masks), indices mapping prompts to frames, and sometimes reference answers.

# Data Structure

```
Preview
├─ DL3DV_frames/      # DL3DV dataset frames (RGB)
├─ DL3DV_masks/       # DL3DV corresponding masks/labels
├─ uco3d_frames/      # UCO3D dataset frames (RGB)
├─ uco3d_masks/       # UCO3D corresponding masks/labels
├─ wildrgbd_frames/   # WildRGBD dataset frames (RGB)
├─ wildrgbd_mask/     # WildRGBD corresponding masks/labels
└─ meta.json
```

# Full SpaceVista-1M

Full SpaceVista-1M will be released soon. Please stay tuned.

# Sample Usage

This section provides guidance on how to download, prepare, and preview the SpaceVista dataset.

### 1. Download the Dataset
First, download the dataset from Hugging Face and extract its contents. The preview data requires `215GB` after uncompressing. Chunks for `.tar.gz` are `202GB` in total.

<!-- split -b 10G datasets_bundle.tar.gz SpaceVista_preview_chunk_ -->

```bash
huggingface-cli download SpaceVista/Data-Preview --repo-type dataset --local-dir . --resume-download

# Combine chunks and decompress
cat SpaceVista_preview_chunk_* > datasets_bundle.tar.gz
tar -xvzf datasets_bundle.tar.gz
```

After uncompressing, the preview data requires `215GB`. These are frames with the original resolutions from the source datasets, which are later normalized to the same resolution during training.

```
110GB — ./DL3DV_frames
0.8GB — ./DL3DV_masks
38GB — ./wildrgbd_frames
3.1GB — ./wildrgbd_mask
60GB — ./uco3d_frames
6.4GB — ./uco3d_masks
0.6GB - ./meta.json
```

### 2. Prepare Metadata for Training (Optional)
If you plan to use this dataset for Supervised Fine-Tuning (SFT), you might need to flatten the metadata. This step prepares the `meta.json` file into a more usable format (e.g., `meta_flatten.json`).

```bash
cd dataset
python flatten.py -i your_path/meta.json -o your_path/meta_flatten.json
```

### 3. Dataloader for Preview Version
A reference dataloader based on LLaMA-factory can be found in the GitHub repository:
[mm_plugin](https://github.com/PeiwenSun2000/SpaceVista/blob/main/sft/src/llamafactory/data/mm_plugin.py)

### 4. Data Preview and Visualization
You can visualize the video frames with annotations and QA pairs using the provided Jupyter Notebook or a Python script.

**Using Jupyter Notebook:**

```bash
jupyter notebook
```
Then, inside Jupyter, open and run the `vis.ipynb` notebook:
```python
# Run in jupyter 
# vis.ipynb
```

**Using a Python script:**
Alternatively, you can save visualizations directly using the `save_visualization.py` script. Remember to adjust the dataset paths.

```bash
# mind the dataset paths

python save_visualization.py --json meta.json 
```

The jupyter outputs are the preview of video frames with annotation and QA.

<img src=".asset/case1.png" width="30%"><img src=".asset/case4.png" width="25%">

<img src=".asset/case2.png" width="33%"><img src=".asset/case3.png" width="34%">


# Reference

```
@article{sun2025spacevista,
  title={SpaceVista: All-Scale Visual Spatial Reasoning from mm to km}, 
  author={Sun, Peiwen and Lang, Shiqiang and Wu, Dongming and Ding, Yi and Feng, Kaituo and Liu, Huadai and Ye, Zhen and Liu, Rui and Liu, Yun-Hui and Wang, Jianan and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2510.09606},
  year={2025}
}
```
