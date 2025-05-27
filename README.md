# LaST
This repository contains the code and models for our paper "LaST: A Transformer-based Network for Spatio-Temporal Predictive Learning with Dynamic Local Awareness". The implementation is based on PyTorch and PyTorch Lightning frameworks.

# Status 🔬:

Our paper has now entered the peer-review process. We have diligently completed the writing and undergone multiple rounds of revision. We will also be continually uploading and updating non-core modules of the code. The full code will, of course, be released upon publication. Stay tuned!🫡

Expected Timeline:
- [x] [2024-11-13] Model Implementation
- [x] [2024-12-26] Experimental Results
- [x] [2025-02-17] Futher Analysis and Paper Writing
- [x] [2025-05-27] Paper Writing and Revision
- [ ] The paper is under review & We are organising and uploading partial code.
- [ ] Code Release


# Quick Start 🎇:
```shell
conda create -n LaST python=3.12
conda activate LaST
pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy
# pip install lightning wandb opencv-python torchmetrics torchvision matplotlib rich ipykernel xarray netcdf4 cartopy -i https://mirrors.aliyun.com/pypi/simple

# python -m ipykernel install --user --name=last
```

# Download the dataset 🗂️:





# Acknowledgements & References 🔗:

1. Our overall training framework is largely inspired by [OpenSTL](https://github.com/chengtan9907/OpenSTL), which we adapted and refactored to better align with the standard PyTorch Lightning usage paradigm.
2. Our core ideas are also significantly influenced by [PredFormer](https://arxiv.org/abs/2410.04733).





---

If you spot any issues or have improvement ideas, we sincerely appreciate you opening an issue or submitting a pull request😊. 

受学识所限，如您发现任何问题或有改进建议，恳请在Issues中提出或直接提交Pull Request，我们将不胜感激并第一时间处理😊。



