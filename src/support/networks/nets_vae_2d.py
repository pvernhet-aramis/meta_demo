from src.support.base_miccai import *
from src.in_out.data_miccai import *
from torchvision.utils import save_image
from functools import reduce
from operator import mul

# -------------------------------------------


class Conv2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ConvTranspose2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Linear_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.view(-1, self.in_ch)).view(-1, self.out_ch)


# ===========================================
# 2D NETWORKS
# ===========================================

# -------------------------------------------
# 5 convolutions networks


class Encoder2d__5_down_O1(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, init_var__s=1.0, dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -5 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.init_var__s = init_var__s

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.down4 = Conv2d_Tanh(8, 16, bias=False)
        self.down5 = Conv2d_Tanh(16, 32, bias=False)
        self.do1 = nn.Dropout(dropout)

        self.linear_mean_1__s = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)

        print('>> Encoder2d__5_down_O1 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.do1(x)

        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)

        return mean__s, logv__s


class Encoder2d__5_down(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, latent_dimension__a, init_var__s=1.0, init_var__a=1.0,
                 dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -5 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a
        self.init_var__s = init_var__s
        self.init_var__a = init_var__a

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.down4 = Conv2d_Tanh(8, 16, bias=False)
        self.down5 = Conv2d_Tanh(16, 32, bias=False)
        self.do1 = nn.Dropout(dropout)

        self.linear_mean_1__s = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)

        self.linear_mean_1__a = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_mean_2__a = nn.Linear(8 * n, latent_dimension__a, bias=False)
        self.linear_logv_1__a = Linear_Tanh(32 * n, 8 * n, bias=False)
        self.linear_logv_2__a = nn.Linear(8 * n, latent_dimension__a, bias=False)

        print('>> Encoder2d__5_down has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.do1(x)

        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)
        mean__a = self.linear_mean_2__a(self.linear_mean_1__a(x.view(bts, -1)))
        logv__a = self.linear_logv_2__a(self.linear_logv_1__a(x.view(bts, -1))) + np.log(self.init_var__a)

        return mean__s, logv__s, mean__a, logv__a


class DeepDecoder2d__5_up(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size_x * out_grid_size_y * 2
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size, last_nonlinearity='tanh', dropout=.2):
        nn.Module.__init__(self)
        assert last_nonlinearity.lower() in ['sigmoid', 'tanh']
        self.inner_grid_size = [int(elt * 2 ** -5) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 32 * self.n, bias=False)
        self.linear2 = Linear_Tanh(32 * self.n, 32 * self.n, bias=False)
        self.up1 = ConvTranspose2d_Tanh(32, 32, bias=False)
        self.up2 = ConvTranspose2d_Tanh(32, 16, bias=False)
        self.up3 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up4 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up5 = nn.ConvTranspose2d(4, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.do1 = nn.Dropout(dropout)

        if last_nonlinearity.lower() == 'sigmoid':
            self.nonlin_layer = nn.Sigmoid()
        elif last_nonlinearity.lower() == 'tanh':
            self.nonlin_layer = nn.Tanh()
        else:
            raise AssertionError
        print('>> DeepDecoder2d__5_up has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        expanded_size = tuple([batch_size, 32] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.do1(x)
        x = self.linear2(x).view(expanded_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.nonlin_layer(x)
        return x


# -------------------------------------------
# 4 convolutions networks


class Encoder2d__4_down_O1(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, init_var__s=1.0, dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -4 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.init_var__s = init_var__s

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.down4 = Conv2d_Tanh(8, 16, bias=False)
        self.do1 = nn.Dropout(dropout)
        self.linear_mean_1__s = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)

        print('>> Encoder2d__4_down_O1 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.do1(x)
        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)

        return mean__s, logv__s


class Encoder2d__4_down(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, latent_dimension__a, init_var__s=1.0, init_var__a=1.0,
                 dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -4 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a
        self.init_var__s = init_var__s
        self.init_var__a = init_var__a

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.down4 = Conv2d_Tanh(8, 16, bias=False)
        self.do1 = nn.Dropout(dropout)

        self.linear_mean_1__s = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(8 * n, latent_dimension__s, bias=False)

        self.linear_mean_1__a = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_mean_2__a = nn.Linear(8 * n, latent_dimension__a, bias=False)
        self.linear_logv_1__a = Linear_Tanh(16 * n, 8 * n, bias=False)
        self.linear_logv_2__a = nn.Linear(8 * n, latent_dimension__a, bias=False)

        print('>> Encoder2d__4_down has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.do1(x)
        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)
        mean__a = self.linear_mean_2__a(self.linear_mean_1__a(x.view(bts, -1)))
        logv__a = self.linear_logv_2__a(self.linear_logv_1__a(x.view(bts, -1))) + np.log(self.init_var__a)

        return mean__s, logv__s, mean__a, logv__a


class DeepDecoder2d__4_up(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size_x * out_grid_size_y * 2
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size, last_nonlinearity='tanh',
                 dropout=.2):
        nn.Module.__init__(self)
        assert last_nonlinearity.lower() in ['sigmoid', 'tanh']
        self.inner_grid_size = [int(elt * 2 ** -4) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 16 * self.n, bias=False)
        self.linear2 = Linear_Tanh(16 * self.n, 32 * self.n, bias=False)
        self.up1 = ConvTranspose2d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up4 = nn.ConvTranspose2d(4, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.do1 = nn.Dropout(dropout)

        if last_nonlinearity.lower() == 'sigmoid':
            self.nonlin_layer = nn.Sigmoid()
        elif last_nonlinearity.lower() == 'tanh':
            self.nonlin_layer = nn.Tanh()
        else:
            raise AssertionError
        print('>> DeepDecoder2d__4_up has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        expanded_size = tuple([batch_size, 32] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.do1(x)
        x = self.linear2(x).view(expanded_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.nonlin_layer(x)
        return x

# -------------------------------------------
# 3 convolutions networks


class Encoder2d__3_down_O1(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, init_var__s=1.0, dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -3 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.init_var__s = init_var__s

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.do1 = nn.Dropout(dropout)

        self.linear_mean_1__s = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(4 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(4 * n, latent_dimension__s, bias=False)

        print('>> Encoder2d__3_down_O1 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.do1(x)

        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)

        return mean__s, logv__s


class Encoder2d__3_down(nn.Module):
    """
    in: (in_grid_size_x, in_grid_size_y)
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension__s, latent_dimension__a, init_var__s=1.0, init_var__a=1.0,
                 dropout=.2):
        nn.Module.__init__(self)
        n = int(reduce(mul, [elt * 2 ** -3 for elt in in_grid_size]))
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a
        self.init_var__s = init_var__s
        self.init_var__a = init_var__a

        self.down1 = Conv2d_Tanh(1, 2, bias=False)
        self.down2 = Conv2d_Tanh(2, 4, bias=False)
        self.down3 = Conv2d_Tanh(4, 8, bias=False)
        self.do1 = nn.Dropout(dropout)

        self.linear_mean_1__s = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_mean_2__s = nn.Linear(4 * n, latent_dimension__s, bias=False)
        self.linear_logv_1__s = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_logv_2__s = nn.Linear(4 * n, latent_dimension__s, bias=False)

        self.linear_mean_1__a = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_mean_2__a = nn.Linear(4 * n, latent_dimension__a, bias=False)
        self.linear_logv_1__a = Linear_Tanh(8 * n, 4 * n, bias=False)
        self.linear_logv_2__a = nn.Linear(4 * n, latent_dimension__a, bias=False)

        print('>> Encoder2d__3_down has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.do1(x)

        mean__s = self.linear_mean_2__s(self.linear_mean_1__s(x.view(bts, -1)))
        logv__s = self.linear_logv_2__s(self.linear_logv_1__s(x.view(bts, -1))) + np.log(self.init_var__s)
        mean__a = self.linear_mean_2__a(self.linear_mean_1__a(x.view(bts, -1)))
        logv__a = self.linear_logv_2__a(self.linear_logv_1__a(x.view(bts, -1))) + np.log(self.init_var__a)

        return mean__s, logv__s, mean__a, logv__a


class DeepDecoder2d__3_up(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size_x * out_grid_size_y * 2
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size, last_nonlinearity='tanh',
                 dropout=.2):
        nn.Module.__init__(self)
        assert last_nonlinearity.lower() in ['sigmoid', 'tanh']
        self.inner_grid_size = [int(elt * 2 ** -3) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 8 * self.n, bias=False)
        self.linear2 = Linear_Tanh(8 * self.n, 16 * self.n, bias=False)
        self.up1 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up2 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up3 = nn.ConvTranspose2d(4, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.do1 = nn.Dropout(dropout)

        if last_nonlinearity.lower() == 'sigmoid':
            self.nonlin_layer = nn.Sigmoid()
        elif last_nonlinearity.lower() == 'tanh':
            self.nonlin_layer = nn.Tanh()
        else:
            raise AssertionError
        print('>> DeepDecoder2d__3_up has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        expanded_size = tuple([batch_size, 16] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.do1(x)
        x = self.linear2(x).view(expanded_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        self.nonlin_layer(x)
        return x


# -------------------------------------------
# 2 and 1 convolutions networks (for upscaling very low dimensions)


class DeepDecoder2d__2_up(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size_x * out_grid_size_y * 2
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size, dropout=.2):
        nn.Module.__init__(self)
        self.inner_grid_size = [int(elt * 2 ** -2) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 4 * self.n, bias=False)
        self.linear2 = Linear_Tanh(4 * self.n, 8 * self.n, bias=False)
        self.linear3 = Linear_Tanh(8 * self.n, 16 * self.n, bias=False)
        self.do1 = nn.Dropout(dropout)
        self.up1 = ConvTranspose2d_Tanh(16, 4, bias=False)
        self.up2 = ConvTranspose2d_Tanh(4, out_channels, bias=False)
        print('>> DeepDecoder2d__2_up has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        expanded_size = tuple([batch_size, 16] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x).view(expanded_size)
        x = self.up1(x)
        x = self.up2(x)
        return x


class DeepDecoder2d__1_up(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size_x * out_grid_size_y * 2
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size, last_nonlinearity='tanh', dropout=.2):
        nn.Module.__init__(self)
        assert last_nonlinearity.lower() in ['sigmoid', 'tanh']
        self.inner_grid_size = [int(elt * 2 ** -1) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 4 * self.n, bias=False)
        self.linear2 = Linear_Tanh(4 * self.n, 8 * self.n, bias=False)
        self.linear3 = Linear_Tanh(8 * self.n, 16 * self.n, bias=False)
        self.up1 = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.do1 = nn.Dropout(dropout)
        if last_nonlinearity.lower() == 'sigmoid':
            self.nonlin_layer = nn.Sigmoid()
        elif last_nonlinearity.lower() == 'tanh':
            self.nonlin_layer = nn.Tanh()
        else:
            raise AssertionError
        print('>> DeepDecoder2d__1_up has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        expanded_size = tuple([batch_size, 16] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.do1(x)
        x = self.linear3(x).view(expanded_size)
        x = self.up1(x)
        x = self.nonlin_layer(x)
        return x

# -------------------------------------------
# Metamorphic 2D Atlas


class MetamorphicAtlas2d(nn.Module):
    """
    Metamorphic Atlas compatible with dimension = 2
    """

    def __init__(self, template_intensities, number_of_time_points, downsampling_data, downsampling_grid,
                 latent_dimension__s, latent_dimension__a,
                 kernel_width__s, kernel_width__a,
                 initial_lambda_square__s=1., initial_lambda_square__a=1., noise_variance=0.1 ** 2,
                 dropout=.2):
        nn.Module.__init__(self)

        # ----------- SET PARAMETERS
        self.decode_count = 0

        assert 0 <= dropout <= 1, "Dropout value not compatible"
        self.dimension = len(template_intensities.size()) - 2      # (batch, channel, width, height)
        assert self.dimension == 2, "specific to dimension 2"
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a

        self.downsampling_data = downsampling_data               # measures to how much depth network can go
        assert self.downsampling_data in [1, 2, 4], "Only supports initial downsampling by 1, 2 and 4"
        self.downsampling_grid = downsampling_grid
        assert self.downsampling_grid in [1, 2, 4], "Only supports grid downsampling by 1, 2 and 4"
        self.grid_size = tuple(template_intensities.size()[2:])
        self.downsampled_grid_size = tuple([gs // self.downsampling_grid for gs in self.grid_size])

        self.v_star_average = torch.zeros(tuple([self.dimension] + list(self.downsampled_grid_size)))
        self.n_star_average = torch.zeros(tuple([1] + list(self.grid_size)))

        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

        self.kernel_width__s = kernel_width__s
        self.kernel_width__a = kernel_width__a

        self.lambda_square__s = initial_lambda_square__s
        self.lambda_square__a = initial_lambda_square__a
        self.noise_dimension = reduce(mul, self.grid_size)
        self.noise_variance = noise_variance
        self.template_intensities = nn.Parameter(template_intensities)
        print('>> Template intensities are {} = {} parameters'.format((template_intensities.size()[1:]),
                                                                      template_intensities.view(-1).size(0)))

        # ----------- SET MODEL
        if self.downsampling_data == 1:
            # ---------- 5 convolutions available
            self.encoder = Encoder2d__5_down(self.grid_size, latent_dimension__s, latent_dimension__a,
                                             init_var__s=(initial_lambda_square__s / np.sqrt(latent_dimension__s)),
                                             init_var__a=(initial_lambda_square__a / np.sqrt(latent_dimension__a)),
                                             dropout=dropout)
            self.decoder__a = DeepDecoder2d__5_up(latent_dimension__a, 1, self.grid_size, dropout=dropout)
            if self.downsampling_grid == 1:
                self.decoder__s = DeepDecoder2d__5_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 2:
                self.decoder__s = DeepDecoder2d__4_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 4:
                self.decoder__s = DeepDecoder2d__3_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            else:
                raise RuntimeError

        elif self.downsampling_data == 2:
            # ---------- 4 convolutions available
            self.encoder = Encoder2d__4_down(self.grid_size, latent_dimension__s, latent_dimension__a,
                                             init_var__s=(initial_lambda_square__s / np.sqrt(latent_dimension__s)),
                                             init_var__a=(initial_lambda_square__a / np.sqrt(latent_dimension__a)),
                                             dropout=dropout)
            self.decoder__a = DeepDecoder2d__4_up(latent_dimension__a, 1, self.grid_size, dropout=dropout)
            if self.downsampling_grid == 1:
                self.decoder__s = DeepDecoder2d__4_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 2:
                self.decoder__s = DeepDecoder2d__3_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 4:
                self.decoder__s = DeepDecoder2d__2_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            else:
                raise RuntimeError

        elif self.downsampling_data == 4:
            # ---------- 3 convolutions available
            self.encoder = Encoder2d__3_down(self.grid_size, latent_dimension__s, latent_dimension__a,
                                             init_var__s=(initial_lambda_square__s / np.sqrt(latent_dimension__s)),
                                             init_var__a=(initial_lambda_square__a / np.sqrt(latent_dimension__a)),
                                             dropout=dropout)
            self.decoder__a = DeepDecoder2d__3_up(latent_dimension__a, 1, self.grid_size, dropout=dropout)
            if self.downsampling_grid == 1:
                self.decoder__s = DeepDecoder2d__3_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 2:
                self.decoder__s = DeepDecoder2d__2_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            elif self.downsampling_grid == 4:
                self.decoder__s = DeepDecoder2d__1_up(latent_dimension__s, self.dimension, self.downsampled_grid_size,
                                                      dropout=dropout)
            else:
                raise RuntimeError

        else:
            raise RuntimeError

        print('>> Metamorphic 2D BayesianAtlas has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def encode(self, x):
        """
        x -> z
        """
        return self.encoder(x - self.template_intensities.detach())

    def decode(self, s, a):
        """
        z -> y
        """

        # INIT
        bts = s.size(0)
        assert bts == a.size(0)
        ntp = self.number_of_time_points
        kws = self.kernel_width__s
        kwa = self.kernel_width__a
        dim = self.dimension
        gs = self.grid_size                 # (tuple) now a tuple of length = dimension (=2)
        dgs = self.downsampled_grid_size    # (tuple) now a tuple of length = dimension (=2)
        dsf = self.downsampling_grid        # (int) assumed identical along all dimensions

        v_star = self.decoder__s(s) - self.v_star_average.type(str(s.type()))
        n_star = self.decoder__a(a) - self.n_star_average.type(str(a.type()))

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v_star, kws, scaled=False)
        n = batched_scalar_smoothing(n_star, kwa, scaled=False)

        # NORMALIZE
        s_norm_squared = torch.sum(s.view(bts, -1) ** 2, dim=1)
        a_norm_squared = torch.sum(a.view(bts, -1) ** 2, dim=1)

        v_norm_squared = torch.sum(v * v_star, dim=tuple(range(1, dim + 2)))
        n_norm_squared = torch.sum(n * n_star, dim=tuple(range(1, dim + 2)))
        normalizer__s = torch.where(s_norm_squared > 1e-10,
                                    torch.sqrt(s_norm_squared / v_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(s.type())))
        normalizer__a = torch.where(a_norm_squared > 1e-10,
                                    torch.sqrt(a_norm_squared / n_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(a.type())))

        normalizer__s = normalizer__s.view(*([bts] + (dim+1)*[1])).expand(v.size())
        normalizer__a = normalizer__a.view(*([bts] + (dim+1)*[1])).expand(n.size())

        v = v * normalizer__s
        n = n * normalizer__a
        assert not torch.isnan(v).any(), "NaN detected v"
        assert not torch.isnan(n).any(), "NaN detected n"

        if self.decode_count < 10:
            print('>> normalizer shape  = %.3E ; max(abs(v)) = %.3E' %
                  (normalizer__s.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(v.detach().cpu().numpy()))))
            print('>> normalizer appea  = %.3E ; max(abs(n)) = %.3E' %
                  (normalizer__a.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(n.detach().cpu().numpy()))))
            print('torch.max(n) = %.3f \n' % torch.max(n))
            self.decode_count += 1

        # FLOW | GRID (batch, dim, dgs_x, dgs_y, dgs_z)
        grid = torch.stack(torch.meshgrid([torch.linspace(0.0, elt - 1.0, delt) for elt, delt in zip(gs, dgs)])
                           ).type(str(s.type())).view(*([1, dim] + list(dgs))).repeat(*([bts] + (dim+1)*[1]))

        x = grid.clone() + v / float(2 ** ntp)
        for t in range(ntp):
            x += batched_vector_interpolation_adaptive(x - grid, x, dsf)
        intensities = batched_scalar_interpolation_adaptive(self.template_intensities + n, x)
        assert not torch.isnan(intensities).any(), "NaN detected intensities"
        return intensities

    def forward(self, s, a):
        return self.decode(s, a)

    def tamper_template_gradient(self, kw, lr, print_info=False):
        pass

    def write(self, observations, prefix, is_half=False):
        s, _, a, _ = self.encode(observations)

        # INIT
        bts = s.size(0)
        assert bts == a.size(0)
        ntp = self.number_of_time_points
        kws = self.kernel_width__s
        kwa = self.kernel_width__a
        dim = self.dimension
        gs = self.grid_size
        dgs = self.downsampled_grid_size
        dsf = self.downsampling_grid

        v_star = self.decoder__s(s) - self.v_star_average.type(str(s.type()))
        n_star = self.decoder__a(a) - self.n_star_average.type(str(a.type()))

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v_star, kws, scaled=False)
        n = batched_scalar_smoothing(n_star, kwa, scaled=False)

        # NORMALIZE
        s_norm_squared = torch.sum(s.view(bts, -1) ** 2, dim=1)
        a_norm_squared = torch.sum(a.view(bts, -1) ** 2, dim=1)
        v_norm_squared = torch.sum(v * v_star, dim=tuple(range(1, dim + 2)))
        n_norm_squared = torch.sum(n * n_star, dim=tuple(range(1, dim + 2)))
        normalizer__s = torch.where(s_norm_squared > 1e-10,
                                    torch.sqrt(s_norm_squared / v_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(s.type())))
        normalizer__a = torch.where(a_norm_squared > 1e-10,
                                    torch.sqrt(a_norm_squared / n_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(a.type())))

        normalizer__s = normalizer__s.view(*([bts] + (dim + 1) * [1])).expand(v.size())
        normalizer__a = normalizer__a.view(*([bts] + (dim + 1) * [1])).expand(n.size())
        v = v * normalizer__s
        n = n * normalizer__a

        # FLOW
        grid = torch.stack(torch.meshgrid([torch.linspace(0.0, elt - 1.0, delt) for elt, delt in zip(gs, dgs)])
                           ).type(str(s.type())).view(*([1, dim] + list(dgs))).repeat(*([bts] + (dim + 1) * [1]))

        x = grid.clone() + v / float(2 ** ntp)
        for t in range(ntp):
            x += batched_vector_interpolation_adaptive(x - grid, x, dsf)

        # INTERPOLATE
        intensities = batched_scalar_interpolation_adaptive(self.template_intensities + n, x).float()

        # WRITE
        template = self.template_intensities.float().mul(255).cpu()

        sliced_images = []
        for i in range(bts):
            # Get data
            appearance = (self.template_intensities + n[i]).float().cpu().mul(255)
            shape = batched_scalar_interpolation_adaptive(self.template_intensities.float().cpu(),
                                                          x[i].float().unsqueeze(0).detach().cpu())[0].mul(255)
            metamorphosis = intensities[i].float().mul(255).cpu()
            target = observations[i].float().mul(255).cpu()

            # Get sliced image
            images_i = [template.squeeze(1), appearance.squeeze(1), shape.squeeze(1),
                        metamorphosis.squeeze(1), target.squeeze(1)]
            sliced_images += images_i

        sliced_images = torch.cat(sliced_images)
        save_image(sliced_images.unsqueeze(1), prefix + '__reconstructions.pdf',
                   nrow=5, normalize=True, range=(0., float(gpu_numpy_detach(torch.max(sliced_images)))))

