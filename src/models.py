import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

# custom packages
from .config import config


#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 199,210 parameters
class TwoNN(nn.Module):
    def __init__(self, name, in_features, num_hiddens, num_classes):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    # def flatten_model(self):
    #     model_dict = self.state_dict()
    #     tensor = np.array([])
    #
    #     for key in model_dict.keys():
    #         tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))
    #
    #     return torch.tensor(tensor).squeeze()

    def flatten_model(self, model_dict=None):
        if model_dict is None:
            model_dict = self.state_dict()
        tensor = np.array([])

        for key in model_dict.keys():
            tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))

        return torch.tensor(tensor).squeeze()

    def unflatten_model(self, flatted_model):
        model_dict = self.state_dict()

        new_model_dict = {}
        for key in model_dict.keys():
            t = model_dict[key].cpu().numpy()
            shape = t.shape
            length = len(t.flatten())

            new_tensor = flatted_model[:length]
            flatted_model = flatted_model[length:]
            new_tensor = np.reshape(new_tensor, shape)

            new_model_dict[key] = new_tensor

        return new_model_dict


class LR(nn.Module):
    def __init__(self, in_channels, name):
        super(LR, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(in_features=in_channels, out_features=1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def flatten_model(self):
        model_dict = self.state_dict()
        tensor = np.array([])

        for key in model_dict.keys():
            tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))

        return torch.tensor(tensor).squeeze()


# for CIFAR10
class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

    # def flatten_model(self):
    #     model_dict = self.state_dict()
    #     tensor = np.array([])
    #
    #     for key in model_dict.keys():
    #         tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))
    #
    #     return torch.tensor(tensor).squeeze()

    def flatten_model(self, model_dict=None):
        if model_dict is None:
            model_dict = self.state_dict()
        tensor = np.array([])

        for key in model_dict.keys():
            tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))

        return torch.tensor(tensor).squeeze()

    def unflatten_model(self, flatted_model):
        model_dict = self.state_dict()

        new_model_dict = {}
        for key in model_dict.keys():
            t = model_dict[key].cpu().numpy()
            shape = t.shape
            length = len(t.flatten())

            new_tensor = flatted_model[:length]
            flatted_model = flatted_model[length:]
            new_tensor = np.reshape(new_tensor, shape)

            new_model_dict[key] = torch.tensor(new_tensor)

        return new_model_dict


def init_weights(model, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN, gpu_ids=config.GPU_IDS):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    # if len(gpu_ids) > 0:
    #     assert (torch.cuda.is_available())
    #     model.to(gpu_ids[0])
    #     model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model
