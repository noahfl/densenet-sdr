# Stochastic Delta Rule implementation using DenseNet in TensorFlow


## THIS REPOSITORY IS NO LONGER IN USE. THE NEW SDR REPOSITORY CAN BE FOUND [HERE](https://github.com/noahfl/sdr-densenet-pytorch).

**NOTE** This is repository is based off of [Illarion Khlestov's DenseNet implementation](https://github.com/ikhlestov/vision_networks/ "ikhlestov/vision_networks/"). Check out his blog post about implementing DenseNet in TensorFlow [here](https://medium.com/@illarionkhlestov/notes-on-the-implementation-densenet-in-tensorflow-beeda9dd1504#.55qu3tfqm).


---------------------------------------------------------------------------------------


Check out @lifeiteng's [results from implementing SDR with WaveNet](https://twitter.com/FeitengLi/status/1029166830844227584).


**UPDATE**: Due to a bug found by @basveeling which has now been corrected, the testing errors are being recalculated. Here are the preliminary results, which I will continue to update as the results come out. "-----" indicates results that have not yet been redone.

|Model type            |Depth  |C10              |C100              |
|:---------------------|:------|:----------------|:-----------------|
|DenseNet(*k* = 12)    |40     |-----(-----)     |-----(-----)      |
|DenseNet(*k* = 12)    |100    |**-----**(-----) |**-----**(-----)  |
|DenseNet-BC(*k* = 12) |100    |-----(-----)     |-----(-----)      |



This repository holds the code for the paper 

'Dropout is a special case of the stochastic delta rule: faster and more accurate deep learning' (submitted to NIPS; on [arXiv](https://arxiv.org/abs/1808.03578))

[Noah Frazier-Logue](https://www.linkedin.com/in/noah-frazier-logue-1524b796/), [Stephen Jose Hanson](http://nwkpsych.rutgers.edu/~jose/)

Stochastic Delta Rule (SDR) is a weight update mechanism that assigns to each weight a standard deviation that changes as a function of the gradients every training iteration. At the beginning of each training iteration, the weights are re-initialized using a normal distribution bound by their standard deviations. Over the course of the training iterations and epochs, the standard deviations converge towards zero as the network becomes more sure of what the values of each of the weights should be. For a more detailed description of the method and its properties, have a look at the paper [link here].



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

## Usage

Example run:

```
    python run_dense_net.py --depth=40 --train --test --dataset=C10 --sdr
```

This run uses SDR instead of dropout. To use dropout, run something like

```
    python run_dense_net.py --depth=40 --train --test --dataset=C10 --keep_prob=0.8
```

where `keep_prob` is the probability (in this case 80%) that a neuron is *kept* during dropout.

**NOTE:** the `--sdr` argument will override the `--keep_prob` argument. For example:

```
    python run_dense_net.py --depth=40 --train --test --dataset=C10 --keep_prob=0.8 --sdr
```

will use SDR and not dropout.


List all available options:

```    
    python run_dense_net.py --help
```

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet) - they may be useful.

Citation:

```     
     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }
```

**KNOWN ISSUES**

 - model will not save due to graph definiton being larger than 2GB

If you see anything wrong, feel free to open an issue!


## Results from SDR paper

This table shows the results on CIFAR shown in the paper. Parameters are all the same as what are used in the paper, except for a batch size of 100 and an epoch size of 100. SDR's beta value was 0.1 and zeta was 0.01. The augmented datasets were not tested on because dropout was not used on these datasets in the original paper, however they may be added in the future (as will the SVHN results and results with higher layer counts).

|Model type            |Depth  |C10              |C100              |
|:---------------------|:------|:----------------|:-----------------|
|DenseNet(*k* = 12)    |40     |2.256(5.160)     |09.36(22.60)      |
|DenseNet(*k* = 12)    |100    |**1.360**(3.820) |**05.16**(11.06)  |
|DenseNet-BC(*k* = 12) |100    |2.520(6.340)     |11.12(25.08)      |



### Epochs to error rate

The below tables show the number of training epochs required to reach a training error of 15, 10, and 5, respectively. For example, the dropout version of DenseNet-40 on CIFAR-10 took 8 epochs to reach a training error of 15, 16 epochs to reach a training error of 10, and 94 epochs to reach a training error of 5. In contrast, the SDR version of DenseNet-40 on CIFAR-10 took 5 epochs to reach a training error of 15, 5 epochs to reach a training error of 10, and 15 epochs to reach a training error of 5. Best results for each value, across both dropout and SDR, are bolded.


#### Dropout 

|Model type            |Depth  |C10             |C100             |
|:---------------------|:------|:---------------|:----------------|
|DenseNet(*k* = 12)    |40     |8 \ 16 \ 94     |95 \ -- \ --     |
|                      |       |                |                 |
|DenseNet(*k* = 12)    |100    |8 \ 13 \ 25     |28 \ 60 \ --     |
|                      |       |                |                 |
|DenseNet-BC(*k* = 12) |100    |10 \ 25 \ --    |-- \ -- \ --     |
|                      |       |                |                 |

#### SDR

|Model type            |Depth  |C10                     |C100                       |
|:---------------------|:------|:-----------------------|:--------------------------|
|DenseNet(*k* = 12)    |40     |**5** \  **8** \ **15** |27 \ 48 \ --               |
|                      |       |                        |                           |
|DenseNet(*k* = 12)    |100    |6 \ 9  \ **15**         |**17** \ **21** \ **52**   |
|                      |       |                        |                           |
|DenseNet-BC(*k* = 12) |100    |**5**  \ **8**  \ 17    |31 \ 87 \ --               |
|                      |       |                        |                           |

Comparison to original DenseNet implementation with dropout
--------

Test results on various datasets. Image normalization per channels was used. Results reported in paper provided in parenthesis. For Cifar+ datasets image normalization was performed before augmentation. This may cause a little bit lower results than reported in paper.

|Model type            |Depth  |C10         |C10+       |C100          |C100+       |
|:---------------------|:------|:-----------|:----------|:-------------|:-----------|
|DenseNet(*k* = 12)    |40     |6.67(7.00)  |5.44(5.24) |27.44(27.55)  |25.62(24.42)|
|DenseNet-BC(*k* = 12) |100    |5.54(5.92)  |4.87(4.51) |24.88(24.15)  |22.85(22.27)|


Difference compared to the [original](https://github.com/liuzhuang13/DenseNet) implementation
---------------------------------------------------------
The existing model should use identical hyperparameters to the original code.

Dependencies
------------

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10 FOR DROPOUT ONLY. SDR was added using a development environment with TensorFlow 1.7 so it may require 1.0+.

Repo supported with requirements files - so the easiest way to install all just run:

- in case of CPU usage `pip install -r requirements/cpu.txt`.
- in case of GPU usage `pip install -r requirements/gpu.txt`.

