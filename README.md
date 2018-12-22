# CycleGAN using PyTorch
CycleGAN is one of the most interesting works I have read. Although the idea behind cycleGAN looks quite intuitive after you read the paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), the official PyTorch implementation by [junyanz](https://github.com/junyanz) is difficult to understand for beginners (The code is really well written but it's just that it has multiple things implemented together).  As I am writing a simpler version of the code for some other work, I thought of making my version of cycleGAN  public for those who are looking for an easier implementation of the paper. 

All the credit goes to the authors of the paper.
This code is inspired by the actual implementation by [junyanz](https://github.com/junyanz) which can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

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
* Try tweaking the arguments to get best performance according to the dataset.

## Results

* For horse to zebra dataset. ( Real - Generated - Reconstructed)

<p float="left">
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/horse_real.png" width="250" />
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/zebra_generated.png" width="250" />
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/horse_reconstructed.png" width="250" />
</p>

<p float="left">
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/zebra_real.png" width="250" />
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/horse_generated.png" width="250" />
  <img src="https://github.com/arnab39/cycleGAN-PyTorch/blob/master/images/zebra_reconstructed.png" width="250" />
</p>
