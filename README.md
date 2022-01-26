# Pytorch-MSCRED

This repository is the implementation of MSCRED using PyTorch.

It has been modified to mini-batch training is available.

Original paper:
[http://arxiv.org/abs/1811.08055](http://arxiv.org/abs/1811.08055)

Original Repository:
[https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED](https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED)

TensorFlow implementation address:
[https://github.com/7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED)

The specific process is as follows:
- First convert the time series data to image matrices
   > python ./utils/matrix_generator.py

- Then train the model and generate the corresponding reconstructed matrices for the test set
   > python main.py

- Finally evaluate the model, the results are stored in the `outputs` folder
   > python ./utils/evaluate.py