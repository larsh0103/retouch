from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils
from torchvision.models import vgg19

class skip_block(torch.nn.Module):

    def __init__(self,ch_in, ch_out,k_size,stride=1,padding=1):
        super(skip_block,self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.ch_in,out_channels=self.ch_out,kernel_size=self.k_size,stride=self.stride,padding=self.padding),
            nn.BatchNorm2d(self.ch_out),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.ch_in,out_channels=self.ch_out,kernel_size=self.k_size,stride=self.stride,padding=self.padding),
            nn.BatchNorm2d(self.ch_out),
            nn.LeakyReLU(0.2)

        )
        self.skip = nn.Sequential()
    def forward(self,x):
        out = self.block(x)
        out += self.skip(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.init_size = opt.img_size // 4
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=7,stride=2)
        )

        self.skip_layers = nn.Sequential(*[skip_block(ch_in=128,ch_out=128,k_size=3,stride=1,padding=1) for i in range(15)])
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(512,512), mode = "nearest")
        )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1))


    def forward(self, x):
        out_1 = self.conv1(x)
        out = self.conv2(out_1)
        out = self.skip_layers(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size = 4, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        ## This configuration works for input slices of 6 x 150 x 150 
        self.model = nn.Sequential(
            *discriminator_block(3, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(in_channels=512,kernel_size=3,out_channels=1,stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(in_features = 258064, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out

class VggShort(nn.Module):
    def __init__(self):
        super(VggShort, self).__init__()
        features = list(vgg19(pretrained = True).features)[:6]
        self.features = nn.ModuleList(features).eval()
    
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {5}:
                results.append(x)
        return results[0]



class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
