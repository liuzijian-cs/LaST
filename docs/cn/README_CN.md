# LaST
简体中文 | [English](../../README.md)

本仓库包含了我们论文《LaST：一种面向时空预测的动态局部感知Transformer网络》的代码与模型。实现基于PyTorch与PyTorch Lightning框架。

---

## 项目状态 🔬

我们的论文已进入同行评审阶段。我们已完成多轮论文撰写与修订，并会持续上传和更新部分非核心代码模块。全部代码将在论文发表后第一时间公开，敬请关注！🫡

**进度时间线：**
- [x] [2024-11-13] 模型实现
- [x] [2024-12-26] 实验结果
- [x] [2025-02-17] 深入分析与论文撰写
- [x] [2025-05-27] 论文修订
- [ ] [当前] 论文评审中 & 部分代码整理上传
- [ ] 代码全部开源

---

## 1. 快速开始 🎇

```shell
conda create -n LaST python=3.12
conda activate LaST

# 安装依赖包
pip install lightning -i https://mirrors.aliyun.com/pypi/simple
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy -i https://mirrors.aliyun.com/pypi/simple

# （可选）Jupyter Notebook 用户可用如下命令安装内核：
python -m ipykernel install --user --name=last
```

---

## 2. 训练 🏋️‍♂️

![](/docs/figs/Table1.jpg)
实验所用数据集统计信息。

### 2.1. 下载数据集 🗂️

我们已将常用数据集整理并上传至 Google Drive 和百度网盘，您可直接下载，或按需自行准备。

数据部分的代码结构如下：

```text
├── data
│   ├── __init__.py  # 如需添加自定义数据集，请在此文件的 data_dict 字典和 setup_data() 函数中注册
│   ├── TaxiBJ
│   │   ├── __init__.py
│   │   ├── conf.yaml       # 配置文件，包含数据集参数
│   │   ├── dataset.npz     # TaxiBJ 数据集文件
│   │   └── TaxiBJDataModule.py     # 数据处理文件
...
```

完整数据模块说明及自定义数据集训练方法，请参见[数据模块介绍](data.md)。

**常用数据集下载表：**

| 数据集名称 | Google Drive | 百度网盘 | 说明 |
|---|---|---|---|
| [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ) | [下载](https://drive.google.com/file/d/1HDN_hF2pOP2JT97kB8VCREIfe5Z22Co-/view?usp=sharing) | [下载](https://pan.baidu.com/s/1VDHPuy61GGwqt05t4NVH8A?pwd=iSHU) | `data/TaxiBJ/dataset.npz` |
| [Weather Bench](https://github.com/pangeo-data/WeatherBench) (T2m, Tcc, Rl) | [下载](https://drive.google.com/file/d/1wxIXK-1vZ9tST_5xB3Ph3QpVB6Q9YhB1/view?usp=sharing) | [下载](https://pan.baidu.com/s/1Wa1S2qjV0fAb0bWlMswnYg?pwd=iSHU) | `data/WeatherBench/5_625/2_temperature/{xxx}.nc` |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php) | [下载](https://drive.google.com/file/d/1jwrXUO6eBh8689NJO8WYoeNMwXtoUD8t/view?usp=sharing) | [下载](https://pan.baidu.com/s/1x78V54ueiW3Iz2CgMOb6zA?pwd=iSHU) | `data/Human/images` & `data/Human/images_txt` |
| [CORAv2.0](https://mds.nmdis.org.cn/) | - | - | 请前往 https://mds.nmdis.org.cn 申请下载 |

---

### 2.2. 训练方法

我们提供两种主要训练方式，并支持顺序训练脚本：

#### ✅ 方法一：配置文件训练

（请补充具体用法示例）

#### ✅ 方法二：命令行参数训练

（请补充具体用法示例）

#### 🔁 顺序训练脚本

（请补充具体用法示例）

---

## 3. 项目结构简介

- `data/`：数据集及其处理模块
- `utils/`：常用工具函数（如日志输出、彩色打印等）
- `docs/`：文档与说明
- `README.md`：英文主文档
- `LICENSE`：MIT开源协议

---

## 4. 致谢与参考 🔗

1. 🫡 本项目训练框架主要参考 [OpenSTL](https://github.com/chengtan9907/OpenSTL)，并根据PyTorch Lightning范式重构。
2. 🫡 核心思想亦受到 [PredFormer](https://arxiv.org/abs/2410.04733) 启发。

---

## 5. 论文引用 📚

如果本仓库对您的研究有帮助，欢迎引用我们的论文（正式发表后会补充完整信息）：

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

## 6. 反馈与贡献

如您发现任何问题或有改进建议，欢迎在 Issues 区留言或直接提交 Pull Request，我们会第一时间处理并致谢😊。

---

## 7. 授权协议

本项目采用 MIT License 开源协议，详见 LICENSE 文件。
