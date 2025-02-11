import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class LargeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(LargeGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
        

class DCGAN_Generator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_Generator, self).__init__()
        self.nz = nz
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            nn.Sigmoid(),
            #nn.Sigmoid()
        )

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class DCGAN_CondGenerator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, num_classes,  nz=100, n_emb=50, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_CondGenerator, self).__init__()
        self.nz = nz
        self.emb = nn.Embedding(num_classes, n_emb)
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz+n_emb, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            #nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        y = self.emb(y)
        z = torch.cat([z, y], dim=1)
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc=3, img_size=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class WGAN_GP_Generator(torch.nn.Module):
    def __init__(self, nz, nc):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=nz, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=nc, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class WGAN_GP_Discriminator(torch.nn.Module):
    def __init__(self, nc):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=nc, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class CMIGeneratorV3(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(CMIGeneratorV3, self).__init__()


        self.layer_names=['l1','layer1','up_layer1','conv_layer1','up_layer2','conv_layer2','conv_layer3']


        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))
        
        self.layer1 =  nn.BatchNorm2d(ngf * 2)

        self.up_layer1 = nn.Upsample(scale_factor=2)
        

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True))


        self.up_layer2 = nn.Upsample(scale_factor=2)
        

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        layer1_out = self.layer1(out)
        out = self.up_layer1(layer1_out)
        conv1_out = self.conv_layer1(out)
        out = self.up_layer2(conv1_out)
        conv2_out = self.conv_layer2(out)
        img = self.conv_layer3(conv2_out)
        return img, [layer1_out,conv1_out,conv2_out]

    def generate_using_convout(self,feature_input,conv_idx):


        if conv_idx==0:

            out = self.up_layer1(feature_input)
            conv1_out = self.conv_layer1(out)
            out = self.up_layer2(conv1_out)
            conv2_out = self.conv_layer2(out)
            img = self.conv_layer3(conv2_out)


        elif conv_idx==1:
            out = self.up_layer2(feature_input)
            conv2_out = self.conv_layer2(out)
            img = self.conv_layer3(conv2_out)

        elif conv_idx==2:

            img = self.conv_layer3(feature_input)

        return img

    def reset_model(self,conv_idx):


        # assert (conv_idx in [0,1,2])


        if conv_idx==0:

            models = [self.l1, self.layer1]   

        elif conv_idx==1:

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1]

        elif conv_idx==2:

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1, self.up_layer2, self.conv_layer2]

        elif conv_idx=='all':

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1, self.up_layer2, self.conv_layer2, self.conv_layer3]

        elif conv_idx=='none':

            models = []


        for model in models:        

            for m in model.modules():
                if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, (nn.BatchNorm2d)) or isinstance(m, nn.SyncBatchNorm):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)