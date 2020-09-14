import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision
import torchvision.models as models

import copy


layer_names = ['Conv1_1', 'ReLU1_1', 'Conv1_2', 'ReLU1_2', 'Pool1',
               'Conv2_1', 'ReLU2_1', 'Conv2_2', 'ReLU2_2', 'Pool2',
               'Conv3_1', 'ReLU3_1', 'Conv3_2', 'ReLU3_2', 'Conv3_3', 'ReLU3_3', 'Conv3_4', 'ReLU3_4', 'Pool3',
               'Conv4_1', 'ReLU4_1', 'Conv4_2', 'ReLU4_2', 'Conv4_3', 'ReLU4_3', 'Conv4_4', 'ReLU4_4', 'Pool4',
               'Conv5_1', 'ReLU5_1', 'Conv5_2', 'ReLU5_2', 'Conv5_3', 'ReLU5_3', 'Conv5_4', 'ReLU5_4', 'Pool5']

style_layers = ['Conv1_1', 'Conv2_1', 'Conv3_1', 'Conv4_1', 'Conv5_1']
content_layers = ['Conv4_2']

def gram(activation):
    F = activation.view((activation.size(1)), -1)
    g = torch.mm(F, F.t())
    return g / F.numel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)


class Model(nn.Module):

    def __init__(self, content_img=None, style_img=None):
        super(Model, self).__init__()
        copied = copy.deepcopy(vgg19)
        self.layers = []

        for name, layer in zip(layer_names, vgg19.children()):
            if 'Conv' in name:
                self.layers.append(layer)
            elif 'ReLU' in name:
                self.layers.append(nn.ReLU())
            elif 'Pool' in name:
                self.layers.append(nn.AvgPool2d(2))

        if content_img is not None: self.set_content_img(content_img)
        if style_img is not None: self.set_style_img(style_img)

    def forward(self, x, calc_loss=True):
        new_content_activations = []
        new_gram_matrices = []

        x = (x - normalization_mean) / normalization_std

        for name, layer in zip(layer_names, self.layers):
            x = layer(x)
            if name in content_layers:
                new_content_activations.append(x if calc_loss else x.detach())
            if name in style_layers:
                new_gram_matrices.append(gram(x if calc_loss else x.detach()))

        if calc_loss:
            content_loss = torch.zeros(1).to(device)
            for new, old in zip(new_content_activations, self.content_activations):
                content_loss += functional.mse_loss(new, old)
            content_loss /= len(self.content_activations)

            style_loss = torch.zeros(1).to(device)
            for new, old in zip(new_gram_matrices, self.gram_matrices):
                style_loss += functional.mse_loss(new, old)
            style_loss /= len(self.gram_matrices)

            return content_loss, style_loss

        else:
            return new_content_activations, new_gram_matrices

    def set_content(self, content_img):
        self.content_activations, _ = self.forward(content_img, calc_loss=False)

    def set_style(self, style_img):
        _, self.gram_matrices = self.forward(style_img, calc_loss=False)
