# Training DCFNet for Image Classification

Code for training image classification with DCFNet.


## Supports

Currently we provide code 

Note that the scripts here do not intend to reproduce the exactly same results as shown in the original paper.

### Datasets

CIFAR-10

CUB-200 bird recognition dataset

### Networks

VGG

ResNet

LeNet

More network architectures are coming soon.

## Usage
Modify train.py to select network achitecture and configurations. 

E.g., you can try training VGG16 with Fourier Bessel bases on CIFAR-10 by setting

```python
net = VGG_DCF('VGG16', bases_grad=False, num_class=NUM_CLASS)
```
in main.py, and run command
```bash
python main.py --gpu 0 --log_dir log_dcf_vgg16
```
The script will creat a new folder named log_dcf_vgg16 for saving checkpoints and log file. Note that the script will remove the folder first and creat an empty one if it already exists. You can remove this by deleting Line 39 in main.py.

## Accuracy
Update soon


