import os
from argparse import ArgumentParser
import model as md
from utils import create_link
import test as tst


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--img_height', type=int, default=128)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/horse2zebra')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/horse2zebra')
    args = parser.parse_args()
    return args


def main():
  args = get_args()

  create_link(args.dataset_dir)
  if args.training:
      print("Training")
      model = md.cycleGAN(args)
      model.train(args)
  if args.testing:
      print("Testing")
      tst.test(args)


if __name__ == '__main__':
    main()