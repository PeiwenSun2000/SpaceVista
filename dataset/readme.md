# Important!

This is a preview of SpaceVista, containing only a subset of tasks and scenes. As you can see, the data format is **not yet unified**, and the meta JSON keys remain inconsistent. We’ve provided a reasonable preview here—please stay tuned for the full, up-to-date release.

By the way, we would **not** recommend you train your model with your own dataloader. The format of this dataset needs to be **refined completely**. As the final version may necessitate a complete redesign of your dataloader, we **have provided and will provide** the dataloaders for this version and future versions.

# What is **All-Scale Spatial Reasoning**

<img src="../.asset/teaser.jpg" width="50%">

Spatial reasoning is the ability to **perceive, interpret, and act** across spatial scales, from millimeter-sized components to distant aerial scenes. All-scale spatial reasoning is fundamental to next-generation intelligent systems and supports diverse applications: mm sensing for advanced manufacturing, cm and m perception for embodied agents, 10m operation for autonomous driving, and 100m for drone based sensing.
Despite progress, existing work shows clear limitations in both model design and dataset coverage. Current scene perception research mostly targets indoor scenes, narrow object classes, and limited spatial ranges, and lacks training paradigms engineered for end-to-end, cross-scale reasoning. SpaceVista addresses this gap by presenting the **first systematic solution** across both data and model dimensions to enable robust, full scene spatial reasoning.

# Data Construction


<img src="../.asset/dataset.jpg" width="50%">

The limited data and performance constraints in existing models necessitate the creation of a dataset with all-scale spatial context. We propose **SpaceVista-1M**, a **diverse, real-world, all-scale** reasoning dataset, as the **first** to the best of our knowledge. SpaceVista-1M primarily comprises diverse spatial reasoning question–answer pairs, with rich semantic (category, rationale), 2D (mask, box, point), and 3D (depth, camera parameters, point cloud) annotations, obtained either natively or through processing. The construction pipeline in the above figure follows the step-by-step procedure of preparing, transforming, and generating to obtain an all-scale dataset by integrating specialized models.

# Data Usage

<!-- split -b 10G datasets_bundle.tar.gz SpaceVista_preview_chunk_ -->

```
huggingface-cli download SpaceVista/Data-Preview --repo-type dataset --local-dir . --resume-download

cat SpaceVista_preview_chunk_* > datasets_bundle.tar.gz

tar -xvzf datasets_bundle.tar.gz
```

# Disk Requirement

Chunks for `.tar.gz` are `202GB` in total.

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

# Dataloader for Preview Version

See the loader of LLaMA-factory in [mm_plugin](https://github.com/PeiwenSun2000/SpaceVista/blob/main/sft/src/llamafactory/data/mm_plugin.py)

# Full SpaceVista-1M

Full SpaceVista-1M will be released soon. Please stay tuned.

# Data Preview

[Jupyter Notebook](https://github.com/PeiwenSun2000/SpaceVista/blob/main/dataset/vis.ipynb) for preview.

```
jupyter notebook

# Run in jupyter 
# vis.ipynb

```

or

[Save annotation for preview.](https://github.com/PeiwenSun2000/SpaceVista/blob/main/dataset/save_visualization.py) 

```

# mind the dataset paths

python save_visualization.py --json meta.json 

```


The Jupyter outputs are the preview of video frames with annotation and QA.

<img src="../.asset/case1.png" width="30%"><img src="../.asset/case4.png" width="25%">

<img src="../.asset/case2.png" width="33%"><img src="../.asset/case3.png" width="34%">


# Reference

```
@article{sun2025spacevista,
  title={SpaceVista: All-Scale Visual Spatial Reasoning from mm to km}, 
  author={Sun, Peiwen and Lang, Shiqiang and Wu, Dongming and Ding, Yi and Feng, Kaituo and Liu, Huadai and Ye, Zhen and Liu, Rui and Liu, Yun-Hui and Wang, Jianan and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2510.09606},
  year={2025}
}
```

