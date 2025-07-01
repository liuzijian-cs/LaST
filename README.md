**We are actively organizing the training framework and code. Once the paper is reviewed, the code and model checkpoints will be released here immediately 🫡**

**我们正在积极的整理训练框架和代码，论文一旦完成审阅，代码和模型检查点将立即公布在此处🫡**

# LaST

English | [简体中文](docs/cn/README_CN.md)

This repository contains the code and models for our paper "LaST: A Transformer-based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness". The implementation is based on PyTorch and PyTorch Lightning frameworks.

# Status 🔬:

Our paper has now entered the peer-review process. We have diligently completed the writing and undergone multiple rounds of revision. We will also be continually uploading and updating non-core modules of the code. The full code will, of course, be released upon publication. Stay tuned!🫡

Expected Timeline:
- [x] [2024-11-13] Model Implementation
- [x] [2024-12-26] Experimental Results
- [x] [2025-02-17] Further Analysis and Paper Writing
- [x] [2025-05-27] Paper Writing and Revision
- [ ] [now] The paper is under review & We are organising and uploading partial code.
- [ ] Code Release


# 1. Quick Start 🎇:
```shell
conda create -n LaST python=3.12
conda activate LaST

# Install the required packages
pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy -i https://mirrors.aliyun.com/pypi/simple

# (Optional) For Jupter Notebook users, you can install the kernel with the following command:
python -m ipykernel install --user --name=last
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
For a complete explanation of the data module and instructions on how to train on your own datasets, [please click here](docs/en/data.md).


<summary>📂 Click to expand full dataset download table</summary>

| Dataset Name                                                               | Google Drive Link                       | Baidu Drive Link                                                    | Description                                               |
|----------------------------------------------------------------------------|-----------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)      | [Download](https://drive.google.com/file/d/1HDN_hF2pOP2JT97kB8VCREIfe5Z22Co-/view?usp=sharing)                            | [Download](https://pan.baidu.com/s/1VDHPuy61GGwqt05t4NVH8A?pwd=iSHU) | `data/TaxiBJ/dataset.npz`                                 |
| [Weather Bench](https://github.com/pangeo-data/WeatherBench)(T2m, Tcc, Rl) | |                                                                     | `data/WeatherBench/5_625/2_temperature/{xxx}.nc`          |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)               |                                         |                                                                     | `data/Human/images`&`data/Human/images_txt`               |
| [CORAv2.0](https://mds.nmdis.org.cn/)(Ssh)                                 | -                                       | -                                                                   | Please apply for the dataset at https://mds.nmdis.org.cn. |



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



