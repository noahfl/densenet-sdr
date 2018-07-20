Stochastic Delta Rule implemtation using DenseNet with TensorFlow
-----------------------------------------------------------------

*NOTE* This is repository is based off of [Illarion Khlestov's DenseNet implementation](https://github.com/ikhlestov/vision_networks/ "ikhlestov/vision_networks/"). Check out his blog post about implementing DenseNet in TensorFlow [here](https://medium.com/@illarionkhlestov/notes-on-the-implementation-densenet-in-tensorflow-beeda9dd1504#.55qu3tfqm).



This repository holds the code for the paper 

Dropout is a special case of the stochastic delta rule: faster and more accurate deep learning (submitted to NIPS)

[Noah Frazier-Logue](https://www.linkedin.com/in/noah-frazier-logue-1524b796/), [Stephen Jose Hanson](http://nwkpsych.rutgers.edu/~jose/)

Stochastic Delta Rule is a weight update mechanism that assigns to each weight a standard deviation that changes as a function of the gradients every training iteration. At the beginning of each training iteration, the weights are re-initialized using a normal distribution bound by their standard deviations. Over the course of the training iterations and epochs, the standard deviations converge towards zero as the network becomes more sure of what the values of each of the weights should be. For a more detailed description of the method and its properties, have a look at the paper [link here].



Two types of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- CIFAR-10
- CIFAR-10+ (with data augmentation)
- CIFAR-100
- CIFAR-100+ (with data augmentation)
- SVHN

A number of layers, blocks, growth rate, image normalization and other training params may be changed trough shell or inside the source code.

Example run:

```
    python run_dense_net.py --train --test --dataset=C10
```

List all available options:

```    
    python run_dense_net.py --help
```

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet) - they may be useful also.

Citation:

```     
     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }
```


Test run
--------

Test results on various datasets. Image normalization per channels was used. Results reported in paper provided in parenthesis. For Cifar+ datasets image normalization was performed before augmentation. This may cause a little bit lower results than reported in paper.

|Model type            |Depth  |C10         |C10+       |C100          |C100+       |
|:---------------------|:------|:-----------|:----------|:-------------|:-----------|
|DenseNet(*k* = 12)    |40     |6.67(7.00)  |5.44(5.24) |27.44(27.55)  |25.62(24.42)|
|DenseNet-BC(*k* = 12) |100    |5.54(5.92)  |4.87(4.51) |24.88(24.15)  |22.85(22.27)|


Difference compared to the [original](https://github.com/liuzhuang13/DenseNet) implementation
---------------------------------------------------------
The existing model should use identical hyperparameters to the original code. If you note some errors - please open an issue.

Dependencies
------------

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10 FOR DROPOUT ONLY. SDR was added using a development environment with TensorFlow 1.7 so it may require 1.0+.

Repo supported with requirements files - so the easiest way to install all just run:

- in case of CPU usage `pip install -r requirements/cpu.txt`.
- in case of GPU usage `pip install -r requirements/gpu.txt`.

