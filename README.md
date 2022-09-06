# SADA_UOAIS

## Environment Setup

```
# Clone project
git clone https://github.com/YueBro/sada_uoais
cd sada_uoais

# Create virtual environment
python3 -m venv sada_env
source sada_env/bin/activate

# Install pytorch 1.8.1 for cuda 10.2 and some other modules
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install shapely torchfile opencv-python pyfastnoisesimd rapidfuzz termcolor

# Install specific detectron2 commit
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'

# Build custom AdelaiDet and other modules
rm -rf build/ **/*.so
python setup.py build develop
```


## Download model checkpoint

Download at [GDrive](https://drive.google.com/file/d/10JpK7RmkGrTqw3X9WLUVLVQSOX0dSu87/view?usp=sharing) [1]. Create folders as `output/eval_model` and place the checkpoint into the folder.


## Download CG-Net

Download at [GDrive](https://drive.google.com/file/d/1-YmmckaLXSZGh9BJkzWXdabNkIeXYh2j/view?usp=sharing) [1]. Place the checkpoint into `foreground_segmentation`.


## Download Datasets

### Download OSD [2]
```
# OSD dataset
wget https://data.acin.tuwien.ac.at/index.php/s/EpX1yej4NShzVtH/download
unzip download
rm download
mv OSD-0.2-depth datasets/OSD-0.2-depth
```

Download amodal annotation at [GDrive](https://drive.google.com/file/d/1ddS721aO1q2wr88gc5ndSOYqYApqmLPe/view?usp=sharing) [1] and unzip into `datasets/OSD-0.2-depth`.

### Download UOAIS-Sim [1]

Download at [GDrive](https://drive.google.com/file/d/1yN6ixntOxQRx-UFPV9-wi9R7iODJh-Nw/view?usp=sharing). Create folder `datasets/UOAIS-Sim` and unzip into the folder.


`datasets` folder structure should look like
```
uoais
├── output
└── datasets
       ├── OSD-0.20-depth
       │     └──amodal_annotation # OSD-amodal
       │     └──annotation
       │     └──disparity
       │     └──image_color
       │     └──occlusion_annotation # OSD-amodal
       │     └──OSD_load_all.json
       │     └──OSD_load_train.json
       │     └──OSD_load_val.json
       └── UOAIS-Sim # for training
              └──annotations
              └──train
              └──val
```


## Visualization

Automatically using checkpoint `output/eval_model/*.pth`

```
# Visualize OSD
python mycode/visualization/visualize_images.py --dataset-name osd --use-cgnet -s 0

# Visualize UOAIS-Sim
python mycode/visualization/visualize_images.py --dataset-name uoais -s 0
```

## Evaluation

Automatically using checkpoint `output/eval_model/*.pth`

```
# Evaluate OSD
python mycode/evaluation/evaluation.py --dataset-name osd --use-cgnet
```

## Train model

```
python mycode/train/train_net_SADA.py
```

## Inspect Architecture

### Config File

`configs/DA_SADA.yaml`

### UOAIS-Net Teacher Model [1]

Meta architecture: `GeneralizedRCNN_FeatureOutput` in `adet/modeling/domain_shift_modules/meta_arch.py`

Backbone: `build_resnet_rgbd_latefusion_fpn_backbone` in `adet/modeling/backbone/rgbdfpn.py`

HOM heads: `ORCNNROIHeads` in `adet/modeling/rcnn/rcnn_heads.py`

### Student Model

`StudentAccusingDiscriminator` in `adet/modeling/domain_shift_modules/disc_for_rcnn.py`


## Reference

[1] S. Back, J. Lee, T. Kim, S. Noh, R. Kang, S. Bak, and K. Lee, “Unseen object amodal instance segmentation via hierarchical occlusion modeling,” 2021.

[2] A. Richtsfeld, T. M¨orwald, J. Prankl, M. Zillich, and M. Vincze, “Segmentation of unknown objects in indoor environments,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 4791–4796.
