import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from ops import conv_bn_lrelu, conv_bn_relu, dconv_bn_relu, ResidualBlock


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))

    def forward(self, x):
        return self.dis(x)



class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.ReflectionPad2d(3),
                                conv_bn_relu(3, dim * 1, 7, 1),
                                conv_bn_relu(dim * 1, dim * 2, 3, 2, 1),
                                conv_bn_relu(dim * 2, dim * 4, 3, 2, 1),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                ResidualBlock(dim * 4, dim * 4),
                                dconv_bn_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                                dconv_bn_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                                nn.ReflectionPad2d(3),
                                nn.Conv2d(dim, 3, 7, 1),
                                nn.Tanh())

    def forward(self, x):
        return self.gen(x)



'''
Class for CycleGAN with train() as a member function

'''
class cycleGAN(object):
    def __init__(self,args):

        utils.cuda_devices([args.gpu_id])

        # Define the network 
        self.Da = Discriminator()
        self.Db = Discriminator()
        self.Gab = Generator()
        self.Gba = Generator()

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        utils.cuda([self.Da, self.Db, self.Gab, self.Gba])

        self.da_optimizer = torch.optim.Adam(self.Da.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.db_optimizer = torch.optim.Adam(self.Db.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gab_optimizer = torch.optim.Adam(self.Gab.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.gba_optimizer = torch.optim.Adam(self.Gba.parameters(), lr=args.lr, betas=(0.5, 0.999))


        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.da_optimizer.load_state_dict(ckpt['da_optimizer'])
            self.db_optimizer.load_state_dict(ckpt['db_optimizer'])
            self.gab_optimizer.load_state_dict(ckpt['gab_optimizer'])
            self.gba_optimizer.load_state_dict(ckpt['gba_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0



    def train(self,args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.img_height,args.img_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                                    batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                                    batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):
            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1

                # set train
                self.Gab.train()
                self.Gba.train()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])

                # Forward pass through generators
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                # Adversarial losses
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                a_cycle_loss = self.L1(a_recon, a_real)
                b_cycle_loss = self.L1(b_recon, b_real)

                # Total generators losses
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss * args.lamda + b_cycle_loss * args.lamda

                # Update generators
                self.Gab.zero_grad()
                self.Gba.zero_grad()
                gen_loss.backward()
                self.gab_optimizer.step()
                self.gba_optimizer.step()

                # Sample from history of generated images
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = a_dis_real_loss + a_dis_fake_loss
                b_dis_loss = b_dis_real_loss + b_dis_fake_loss

                # Update discriminators
                self.Da.zero_grad()
                self.Db.zero_grad()
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.da_optimizer.step()
                self.db_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss,a_dis_loss+b_dis_loss))

            # Override the latest checkpoint 
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'da_optimizer': self.da_optimizer.state_dict(),
                                   'db_optimizer': self.db_optimizer.state_dict(),
                                   'gab_optimizer': self.gab_optimizer.state_dict(),
                                   'gba_optimizer': self.gba_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
