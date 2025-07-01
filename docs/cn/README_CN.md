# LaST
简体中文 | [English](../../README.md)

本代码仓库包含了我们的论文《LaST：一种面向时空学习的动态局部感知注意力网络模型》，相关代码实现基于PyTorch以及PyTorch Lightning训练框架。


# 二、训练 🏋️‍♂️ ：
![](/docs/figs/Table1.jpg)
我们实验所用数据集的统计信息。

## 2.1. 下载数据集 🗂️：

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
完整的数据模块说明以及如何使用自定义数据集进行训练的方法，请[点击这里](data.md)查看。
