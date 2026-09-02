# LaST: A Transformer-Based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness

This repository contains the official PyTorch and PyTorch Lightning implementation of **LaST**, together with training configurations and pretrained checkpoints.

- **Paper:** [LaST: A transformer-based network for spatio-temporal predictive learning with dynamic local awareness](https://www.sciencedirect.com/science/article/pii/S0950705126004624)
- **Languages:** English | [简体中文](docs/cn/README_CN.md)

## Overview

LaST is a Transformer-based network for spatio-temporal predictive learning (STPL). Its Spatio-Temporal Local-Aware Attention (STLAA) mechanism combines query-centered local attention with global self-attention in a single attention layer. LaST also uses Depthwise Convolutional Gated Linear Units (DCGLU) and three-dimensional spatio-temporal positional encoding to improve local feature modeling and preserve spatial and temporal structure.

Experiments cover six datasets across traffic forecasting, meteorology, ocean dynamics, and human motion capture. LaST achieves consistent improvements while using fewer parameters than competing methods; detailed results and ablation studies are available in the paper.

![Overall structure of LaST.](docs/figs/Figure_2.jpg)

*Overall structure of LaST.*

![Detailed architecture of STLAA blocks.](docs/figs/Figure_3.jpg)

*Each STLAA block integrates a Temporal Local-Aware Attention Block (TLAAB) and a Spatial Local-Aware Attention Block (SLAAB). Their local branches operate on $1 \times 3$ temporal windows and $3 \times 3$ spatial neighborhoods, respectively.*

## Installation

The project requires Python 3.12 or later. Dependencies are managed through `pyproject.toml` and `uv.lock`.

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
.\.venv\Scripts\activate
```

### Linux and macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Data Preparation

![Datasets used in the experiments.](docs/figs/Table1.jpg)

The dataset modules and configuration files are stored under `data/`. For details about the data interface and adding a custom dataset, see the [data module documentation](docs/en/data.md).

| Dataset | OneDrive | Baidu Netdisk | Expected location |
| --- | --- | --- | --- |
| [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ) | [Download](https://1drv.ms/u/c/b756f405097b8e82/ETbnKFeKkNVDjOB5UwtXn_0BXR_VoNS3_2uPPcJbcopvyg) | [Download](https://pan.baidu.com/s/1VDHPuy61GGwqt05t4NVH8A?pwd=iSHU) | `data/TaxiBJ/dataset.npz` |
| [WeatherBench](https://github.com/pangeo-data/WeatherBench) (T2m, Tcc, Rl) | [Download](https://1drv.ms/u/c/b756f405097b8e82/ETbnKFeKkNVDjOB5UwtXn_0BXR_VoNS3_2uPPcJbcopvyg) | [Download](https://pan.baidu.com/s/1Wa1S2qjV0fAb0bWlMswnYg?pwd=iSHU) | `data/WeatherBench/5_625/2_temperature/{xxx}.nc` |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php) | [Download](https://1drv.ms/f/c/b756f405097b8e82/Ep1YpOl6MhFBi0vEZ7zGKJQB9u7rssMvxgob4kTizr36CQ) | [Download](https://pan.baidu.com/s/1Rt69aYiugVPQci9YJK25Tg?pwd=iSHU) | `data/Human/images` and `data/Human/images_txt` |
| [CORAv2.0](https://mds.nmdis.org.cn/) | - | - | Apply for access from the dataset provider. |

## Usage

Train LaST with a dataset configuration:

```bash
python main.py --conf TaxiBJ
```

A saved experiment configuration can also be used:

```bash
python main.py --args path/to/args.yaml
```

Evaluate a pretrained checkpoint:

```bash
python main.py --eval \
  --ckpt LaST_best_checkpoints/taxi_beijing/best.ckpt \
  --args LaST_best_checkpoints/taxi_beijing/args.yaml
```

Run `python main.py --help` to see the available command-line options.

## Project Structure

```text
LaST/
├── main.py                    # Training and evaluation entry point
├── batch_runner.py            # Sequential experiment runner
├── pyproject.toml             # Project configuration and dependencies
├── data/                      # Dataset modules and data loaders
├── method/                    # LaST and baseline implementations
├── utils/                     # Training utilities and callbacks
├── docs/                      # Documentation and figures
└── LaST_best_checkpoints/     # Pretrained checkpoints and configurations
```

## Acknowledgements

The training framework was inspired by [OpenSTL](https://github.com/chengtan9907/OpenSTL) and adapted to follow the PyTorch Lightning design. The model was also influenced by [PredFormer](https://arxiv.org/abs/2410.04733).

## Citation

If you find this repository useful, please cite:

```bibtex
@article{Liu2026LaST,
  title   = {LaST: A Transformer-based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness},
  author  = {Zijian Liu and Yehao Wang and Zhuolin Li and Jie Yu and Chengci Wang and Zhiyu Liu and Shuai Zhang and Lingyu Xu},
  journal = {Knowledge-Based Systems},
  volume  = {340},
  pages   = {115722},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.115722}
}
```

## License

This project is released under the [MIT License](LICENSE).
