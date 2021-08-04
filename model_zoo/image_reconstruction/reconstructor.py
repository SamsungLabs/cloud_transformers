from torch import nn
from layers.multihead_ct_adain import MultiHeadUnionAdaIn, forward_style
from layers.utils import AdaIn1dUpd

import torchvision.models as models


def sum_hack(lists):
    start = lists[0]
    for i in range(1, len(lists)):
        start += lists[i]

    return start


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


class Model(nn.Module):
    def __init__(self, num_latent=512):
        super().__init__()

        self.model_dim = 512

        self.res50_model = nn.Sequential(ResNet50Bottom(models.resnet50(pretrained=True)),
                                         nn.AdaptiveAvgPool2d((1, 1)))

        self.mapping = nn.Sequential(nn.Linear(2048, num_latent),
                                     nn.ReLU(inplace=True))

        self.start = nn.Sequential(nn.Conv1d(in_channels=3,
                                   out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(True))

        self.attentions_decoder = nn.ModuleList(sum_hack([[MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                features_dims=[4, 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[128, 32],
                                                                model_dim_out=self.model_dim,
                                                                n_latent=num_latent,
                                                                tensor_dims=[2, 3]),
                                                 MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[64, 16],
                                                                model_dim_out=self.model_dim,
                                                                n_latent=num_latent,
                                                                tensor_dims=[2, 3]),
                                                 MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 8],
                                                                heads=[16, 16],
                                                                tensor_sizes=[16, 8],
                                                                model_dim_out=self.model_dim,
                                                                n_latent=num_latent,
                                                                tensor_dims=[2, 3])] for _ in range(4)]))

        self.final = nn.Sequential(nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=3, kernel_size=1),
                                   nn.Sigmoid())

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, noise, input, return_lattice=False):
        z = self.res50_model(input).reshape(-1, 2048)
        z = self.mapping(z)

        x = forward_style(self.start, noise, z)

        lattices_sizes = []

        for i in range(len(self.attentions_decoder)):
            x, lattice_size = self.attentions_decoder[i](x, z, noise)
            lattices_sizes += lattice_size

        x = forward_style(self.final, x, z)

        return x.unsqueeze(2), lattices_sizes
