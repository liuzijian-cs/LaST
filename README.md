# LaST: A Transformer-Based Network for Spatio-temporal Predictive Learning with Dynamic Local Awareness

This repository contains the training framework, model implementation, configuration files, and checkpoints for our paper, "LaST: A Transformer-based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness." The implementation is based on PyTorch and PyTorch Lightning.

English | [简体中文](docs/cn/README_CN.md)

## Status 🔬

**核心训练代码已全部上传完成。目前，我们正致力于完善项目文档，并对已上传代码进行新一轮的可用性复核与测试。**

**All training-related code has been uploaded. We are currently actively updating the documentation and verifying the usability of the uploaded code to ensure a smooth experience for everyone.**

- [X] [2025-11-22] **Code Release**（**The relevant model code has been uploaded**, and the remaining code will be gradually supplemented.）
- [X] [2025-11-30] We are actively organizing and uploading all code related to training and validation.
- [ ] We are actively writing relevant documentation to help readers quickly understand and reproduce our research.

- Eval

```python
python main.py --eval --ckpt LaST_best_checkpoints/taxi_beijing/best.ckpt --args LaST_best_checkpoints/taxi_beijing/args.yaml
```

# 1. Quick Start 🎇:

## 1.1 Environment

### UV

`uv` is a fast, modern Python package and project manager developed by Astral, and we recommend using it to quickly set up environments.

#### For Windows System:

- Install uv:

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Sync and Activate Environment:

```bash
uv sync
.venv\Scripts\activate
```

#### For Linux and macOS Systems:

- Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Sync and Activate Environment:

```bash
uv sync
source .venv/bin/activate
```

### Conda & Forge

TODO: 后续更新

```shell
conda create -n LaST python=3.12
conda activate LaST

# Install the required packages
pip install lightning -i https://mirrors.aliyun.com/pypi/simple
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy -i https://mirrors.aliyun.com/pypi/simple

# (Optional) For Jupter Notebook users, you can install the kernel with the following command:
python -m ipykernel install --user --name=last
```

## 1.2 Data Preparation

## 1.3 Inference with Model Checkpoints

## 1.4 Training the Model from Scratch

# 2. Implementation Framework of LaST

Ours Framework (the toolkit for training LaST) is a spatio-temporal modeling and video prediction framework built on the Lightning platform. It is designed for efficient data processing, model construction, result analysis, and visualization. The framework supports a wide range of spatio-temporal prediction tasks, including weather forecasting, video analysis, and traffic flow prediction, among others.

All modules are developed in strict adherence to Lightning's design principles, utilizing Callback functions to enable a modular architecture. This approach ensures simplicity, extensibility, and efficiency, providing researchers and developers with a powerful, flexible solution for spatio-temporal data modeling.

TODO: 这里说明整个框架结构

```text
LaST/


```

# 2. Train 🏋️‍♂️ :

![](/docs/figs/Table1.jpg)
Overview of the datasets employed in our experiments.

## 2.1. Download the dataset 🗂️:

To make it easier for everyone, we have organized and uploaded some commonly used datasets to Google Drive and Baidu Drive. You can directly download these datasets or download them yourself as needed.

The code structure for the data part is as follows:

```text
├── data
│   ├── __init__.py  # If you need to add your own dataset, you need to include it in the data_dict dictionary and the setup_data() function in this file
│   ├── TaxiBJ
│   │   ├── __init__.py
│   │   ├── conf.yaml               # Configuration file containing dataset-related parameters
│   │   ├── dataset.npz             # This is the TaxiBJ dataset file
│   │   └── TaxiBJDataModule.py     # Data processing file
...
```

For a complete explanation of the data module and instructions on **how to train on your own datasets**, [please click here](docs/en/data.md).

<summary>📥 Click to expand full dataset download table</summary>

| Dataset Name                                                            | OneDrive                                                                                     | BaiduNetDisk                                                      | Description                                               |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)      | [Download](https://1drv.ms/u/c/b756f405097b8e82/ETbnKFeKkNVDjOB5UwtXn_0BXR_VoNS3_2uPPcJbcopvyg) | [Download](https://pan.baidu.com/s/1VDHPuy61GGwqt05t4NVH8A?pwd=iSHU) | `data/TaxiBJ/dataset.npz`                               |
| [Weather Bench](https://github.com/pangeo-data/WeatherBench)(T2m, Tcc, Rl) | [Download](https://1drv.ms/u/c/b756f405097b8e82/ETbnKFeKkNVDjOB5UwtXn_0BXR_VoNS3_2uPPcJbcopvyg) | [Download](https://pan.baidu.com/s/1Wa1S2qjV0fAb0bWlMswnYg?pwd=iSHU) | `data/WeatherBench/5_625/2_temperature/{xxx}.nc`        |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)               | [Download](https://1drv.ms/f/c/b756f405097b8e82/Ep1YpOl6MhFBi0vEZ7zGKJQB9u7rssMvxgob4kTizr36CQ) | [Download](https://pan.baidu.com/s/1Rt69aYiugVPQci9YJK25Tg?pwd=iSHU) | `data/Human/images`&`data/Human/images_txt`           |
| [CORAv2.0](https://mds.nmdis.org.cn/)                                      | -                                                                                            | -                                                                 | Please apply for the dataset at https://mds.nmdis.org.cn. |

## 2.3. Training 🏋️‍♂️:

We provide two main methods for training your model, along with a script example for sequential training:

### ✅ Method 1: Prepare Configuration Files

### ✅ Method 2: Use Command-Line Arguments

### 🔁 Sequential Training Script

# Acknowledgements & References 🔗:

1. 🫡 Our overall training framework is largely inspired by [OpenSTL](https://github.com/chengtan9907/OpenSTL), which we adapted and refactored to better align with the standard PyTorch Lightning usage paradigm.
2. 🫡 Our core ideas are also significantly influenced by [PredFormer](https://arxiv.org/abs/2410.04733).

# Citation 📚:

If you find this repository useful, please consider citing our paper (citation to be updated upon publication):

```bibtex
@ARTICLE{Liu2025LaST,
    title = {LaST: A Transformer-based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness},
    author = {Zijian Liu, Yehao Wang, Zhuolin Li, Jie Yu, Chengci Wang, Zhiyu Liu, Shuai Zhang and Lingyu Xu},
    booktitile = {},
    note = {Under review},
    year={2025}
}
```

---

If you spot any issues or have improvement ideas, we sincerely appreciate you opening an issue or submitting a pull request😊.

受学识所限，如您发现任何问题或有改进建议，恳请在Issues中提出或直接提交Pull Request，我们将不胜感激并第一时间处理😊。
