<p align="center">
  <h3 align="center"><strong>ShapeAtlas: Repurposing 2D Diffusion Models for 3D Shape Completion</strong></h3>


<p align="center">
    <a href="https://shockwavehe.github.io/">Yao He</a><sup>1</sup>,
    <a href="https://youngjoongunc.github.io/">Youngjoong Kwon</a><sup>1</sup>,
    <a href="https://ai.stanford.edu/~xtiange/">Tiange Xiang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=9K3ox0QAAAAJ&hl=en">Wenxiao Cai</a><sup>1</sup>,
    <a href="https://stanford.edu/~eadeli/">Ehsan Adeli</a><sup>1</sup>,
    <br>
    <sup>1</sup>Stanford University
    <br>
</p>



<div align="center">

<a href='https://arxiv.org/abs/2512.13991'><img src='https://img.shields.io/badge/arXiv-2408.02555-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://shockwavehe.github.io/ShapeAtlas_website/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;

</div>


<p align="center">
    <img src="demo/demo_video.gif" alt="Demo GIF" width="512px" />
</p>

### Official implementation of paper: Repurposing 2D Diffusion Models for 3D Shape Completion.

## Contents
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Important Notes](#important-notes)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Installation
1. Clone our repo and create conda environment
```
git clone https://github.com/shockwaveHe/ShapeAtlas.git && cd ShapeAtlas
git submodule --init
conda create -n ShapeAtlas python==3.10.13 -y
conda activate ShapeAtlas
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_data_process.txt
pip install -U gradio
# Installing pytorch3d, we recommend using the prebuilt wheel found here
# https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.2.2cu118
```
2. Install NVdiffrast
```
conda activate ShapeAtlas
cd shapeatlas_utils/nvdiffrast
pip install -e .
```
3. Install fast-lapjv
```
conda activate ShapeAtlas
cd shapeatlas_utils/fast-lapjv
pip install -e .
```
4. Installing environment to train unet
```
conda create -n ShapeAtlas_train python=3.10 -y
conda activate ShapeAtlas_train
pip install -r requirements_train.txt --extra-index-url https://download.pytorch.org/whl/cu118
cd nvdiffrast
pip install -e .
```
## Usage

### Dataset Processing

Prepare point cloud data from raw meshes using `shapeatlas_utils/data_processor.py`. The script supports ShapeNet, Objaverse, and PCN datasets.
```
cd shapeatlas_utils
python data_processor.py --base_dir <path_to_raw_meshes> --out_dir <output_path>
```
- `--base_dir`: path to the raw mesh dataset (e.g., ShapeNetCore, Objaverse downloads)
- `--out_dir`: output directory for processed point clouds

Run `python data_processor.py --help` for additional options.

### Atlas Generation

Atlas generation maps a 3D point cloud to a 2D grid representation via optimal transport in two steps:

**Step 1: Sphere mapping** -- Map point clouds onto a unit sphere using OT, with optional visibility computation via nvdiffrast rasterization against a fixed camera set.
```
cd shapeatlas_utils
python uneven_ot_structuralization.py sphere --source_root data/PCN_processed --do_visibility --num_workers 4 --split 1
```
- `--source_root`: directory containing processed point cloud data (output from dataset processing)
- `--do_visibility`: enable visibility-aware atlas generation
- `--num_workers`: number of parallel workers for processing
- `--split`: split index for distributed processing across multiple jobs (omit to process all data in one run)

**Step 2: Atlas mapping** -- Map the spherical representation to a 2D grid using precomputed sphere-to-grid correspondences.
```
cd shapeatlas_utils
python uneven_ot_structuralization.py atlas --source_root data/PCN_processed --split 1
```
- `--source_root`: same directory as Step 1 (sphere mapping results are read from subdirectories)
- `--correspondance_path`: path to precomputed sphere-to-grid correspondences (defaults to `shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz`)
- `--split`: split index, matching Step 1

### Single Point Cloud to Atlas

To convert a single point cloud (`.ply`) to an atlas without the full dataset pipeline:
```
python single_atlas_generation.py
```
This script runs both the spherical OT and plane OT steps on individual `.ply` files and saves the resulting atlas (xyz, normal, mask) as `.pt` and visualization images. Edit the `source_root` and `save_root` paths in the script to point to your data.

## Training

Training consists of two stages. All training scripts use config files under `config/` and support multi-GPU training via `accelerate`.

### Stage 1: Train UNet

Train the base UNet model:
```
# Multi-GPU (adjust CUDA_VISIBLE_DEVICES and --num_processes to match your GPU count)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29500 train_unet.py --config config/train_unet.yaml

# Single GPU (for debugging)
CUDA_VISIBLE_DEVICES=0 python train_unet.py --config config/train_unet.yaml
```
- `--config`: path to the YAML config file controlling datasets, hyperparameters, and output paths
- `--main_process_port`: port for inter-process communication (change if the default 29500 is occupied)

### Stage 2: Train Conditional UNet with Chamfer Loss

Fine-tune with chamfer distance and point-mesh distance losses. Set `resume_from_checkpoint` in the config to the Stage 1 checkpoint path.
```
# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 29500 train_conditional_unet_w_chamfer_loss.py --config config/train_condition_unet_w_chamfer_loss.yaml

# Single GPU (for debugging)
CUDA_VISIBLE_DEVICES=0 python train_conditional_unet_w_chamfer_loss.py --config config/train_condition_unet_w_chamfer_loss.yaml
```
- `--config`: path to the YAML config file; Stage 2 configs add loss weights (`pointcloud_reconstruction_loss_weight`, `point_mesh_distance_loss_weight`) on top of the base diffusion loss
- `--tag`: optional string appended to the experiment name for distinguishing runs

### Evaluation
```
python eval_contition_unet_new.py --config config/eval_condition_unet_new.yaml
```
- `--config`: path to the eval config; set `ckpt_dir` in the config to point to your trained model checkpoint

## Important Notes
- Config files under `config/` contain placeholder relative paths for datasets and pretrained models. Update `dataset_dir`, `train_split`, `test_split`, `correspondence_file`, `base_model_path`, `vae_model_path`, `image_encoder_path`, and `ckpt_dir` before running.
- The input mesh will be normalized to a unit bounding box. The up vector of the input mesh should be +Y for better results.

## Acknowledgement

Our code is built on top of:

* [MeshAnythingV2](https://github.com/buaacyw/MeshAnything) - Dataset processing pipeline
* [GaussianCube](https://github.com/GaussianCube/GaussianCube) - Optimal transport structuralization
* Feed-forward Human Performance Capture via Progressive Canonical Space Updates (not yet released) - Atlas-based diffusion architecture

## BibTeX
```
@article{he2025repurposing,
  title={Repurposing 2D Diffusion Models for 3D Shape Completion},
  author={He, Yao and Kwon, Youngjoong and Xiang, Tiange and Cai, Wenxiao and Adeli, Ehsan},
  journal={arXiv preprint arXiv:2512.13991},
  year={2025}
}
```
