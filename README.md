# CycleGAN using PyTorch
CycleGAN is one of the most interesting work I have read. Although, the idea behind cycleGAN looks quite intuitive after you read the paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), I found the official PyTorch implementation of cycleGAN a bit difficult to understand. I had to implemented cycleGAN to use it for some other work. So, I thought of making my clean and lucid implementation of cycleGAN using PyTorch public for others. 

## Requirements
- The code has been written in Python (3.5.2) and PyTorch (0.4.1)

## How to run
* To download datasets (eg. horse2zebra)
```
$ sh ./download_dataset.sh horse2zebra
```
* To run training
```
$ python main.py --training True
```
* To run testing
```
$ python main.py --testing True
```

## Results
Coming Soon
