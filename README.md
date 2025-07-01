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
## 2.1. Download the dataset 🗂️:

为了方便大家使用，我们已经将一些常用的数据集整理并上传到Google Drive和Baidu Drive上。您可以直接下载这些数据集，或者根据需要自行下载。

涉及数据部分的代码结构如下：

```text
├── data
│   ├── __init__.py  # 如果你需要添加自己的数据，需要在这个文件中的data_dict字典和setup_data()函数中加入引入你的数据集
│   ├── TaxiBJ
│   │   ├── __init__.py
│   │   ├── conf.yaml       # 配置文件，包含数据集的相关参数
│   │   ├── dataset.npz     # 这是TaxiBJ数据集文件
│   │   └── TaxiBJDataModule.py     # 数据处理文件
...
```
完整的数据模块说明请[点击这里](docs/en/data.md)查看。



<summary>📂 Click to expand full dataset download table</summary>

| Dataset Name                                                               | Google Drive Link                                      | Baidu Drive Link                           |Description|
|----------------------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------|---------------------------------------------|
| [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)      | [Download]()              | [Download]() pwd: `abcd`                   ||
| [Weather Bench](https://github.com/pangeo-data/WeatherBench)(T2m, Tcc, Rl) | [Download](https://drive.google.com/yyy)              | ||
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)               |||
| [CORAv2.0](https://mds.nmdis.org.cn/)(Ssh)                                 |||



## 2.2. How to Train on Your Own Dataset ☝️:


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



