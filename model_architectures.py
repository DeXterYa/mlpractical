from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()
        # SE_block parameter

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_layers):  # for number of layers times

            # self.layer_dict['avg_pooling_{}'.format(i)] = nn.AdaptiveAvgPool2d(1) # use adaptiveavgpool2d to control the output of average pooling
            # # b, c, _, _ = out.size()
            # # out = self.layer_dict['avg_pooling_{}'.format(i)](out).view(b, c)
            #
            # self.layer_dict['fc_{}'.format(i)] = nn.Sequential(
            #     nn.Linear(out.shape[1], out.shape[1] // self.se_reduction, bias=False),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(out.shape[1] // self.se_reduction, out.shape[1], bias=False),
            #     nn.Sigmoid()
            # )

            # out = self.layer_dict['fc_{}'.format(i)](out)
            # out = out.view(b, c, 1, 1)

            # comment out the steps above because the SE_block won't change the shape of out

            self.layer_dict['conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters, padding=1,
                                                             bias=self.use_bias)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = F.relu(out)  # apply relu
            print(out.shape)
            if self.dim_reduction_type == 'strided_convolution':  # if dim reduction is strided conv, then add a strided conv
                self.layer_dict['dim_reduction_strided_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=2,
                                                                                       dilation=1)

                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # use strided conv to get an output
                out = F.relu(out)  # apply relu to the output
            elif self.dim_reduction_type == 'dilated_convolution':  # if dim reduction is dilated conv, then add a dilated conv, using an arbitrary dilation rate of i + 2 (so it gets smaller as we go, you can choose other dilation rates should you wish to do it.)
                self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=1,
                                                                                       dilation=i + 2)
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](
                    out)  # run dilated conv on input to get output
                out = F.relu(out)  # apply relu on output

            elif self.dim_reduction_type == 'max_pooling':
                self.layer_dict['dim_reduction_max_pool_{}'.format(i)] = nn.MaxPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                self.layer_dict['dim_reduction_avg_pool_{}'.format(i)] = nn.AvgPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

            print(out.shape)
        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out,
                                        2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_layers):  # for number of layers

            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            out = F.relu(out)  # pass conv outputs through ReLU
            if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # pass previous outputs through a strided convolution indexed i
                out = F.relu(out)  # pass strided conv outputs through ReLU

            elif self.dim_reduction_type == 'dilated_convolution':
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                out = F.relu(out)

            elif self.dim_reduction_type == 'max_pooling':
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()


class BatchRelationalModule(nn.Module):
    def __init__(self, input_shape, num_filters, num_layers, num_outputs):
        super(BatchRelationalModule, self).__init__()

        self.input_shape = input_shape
        self.block_dict = nn.ModuleDict()
        self.first_time = True
        self.num_filters = num_filters
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.build_block()

    def build_block(self):
        out_img = torch.zeros(self.input_shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            print(out_img.shape)
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape
        print(out_img.shape)
        # x_flat = (64 x 25 x 24)
        self.coord_tensor = []
        for i in range(length):
            self.coord_tensor.append(torch.Tensor(np.array([i])))

        self.coord_tensor = torch.stack(self.coord_tensor, dim=0).unsqueeze(0)

        if self.coord_tensor.shape[0] != out_img.shape[0]:
            self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

        out_img = torch.cat([out_img, self.coord_tensor], dim=2)

        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)

        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        print(out.shape)
        for idx_layer in range(self.num_layers):
            self.block_dict['g_fcc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=self.num_filters)
            out = F.relu(self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out))

        # reshape again and sum
        print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)
        print('here', out.shape)
        """f"""
        self.post_processing_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_filters)
        out = self.post_processing_layer.forward(out)
        out = F.relu(out)
        self.output_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_outputs)
        out = self.output_layer.forward(out)
        print('Block built with output volume shape', out.shape)

    def forward(self, x_img):

        out_img = x_img
        # print("input", out_img.shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        if self.coord_tensor.shape[0] != out_img.shape[0]:
            self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

        out_img = torch.cat([out_img, self.coord_tensor.to(x_img.device)], dim=2)
        # x_flat = (64 x 25 x 24)
        # print('out_img', out_img.shape)
        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)
        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        for idx_layer in range(3):
            out = F.relu(self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out))

        # reshape again and sum
        # print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)

        """f"""
        out = self.post_processing_layer.forward(out)
        out = F.relu(out)
        out = self.output_layer.forward(out)
        # print('Block built with output volume shape', out.shape)
        return out


class BatchRelationalWithoutLocationsModule(nn.Module):
    def __init__(self, input_shape):
        super(BatchRelationalWithoutLocationsModule, self).__init__()

        self.input_shape = input_shape
        self.block_dict = nn.ModuleDict()
        self.first_time = True
        self.build_block()

    def build_block(self):
        out_img = torch.zeros(self.input_shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            print(out_img.shape)
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)

        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        print(out.shape)
        for idx_layer in range(3):
            self.block_dict['g_fcc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=32)
            out = F.relu(self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out))

        # reshape again and sum
        print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)
        print('here', out.shape)
        """f"""
        self.post_processing_layer = nn.Linear(in_features=out.shape[1], out_features=32)
        out = self.post_processing_layer.forward(out)
        out = F.relu(out)
        self.output_layer = nn.Linear(in_features=out.shape[1], out_features=32)
        out = self.output_layer.forward(out)
        print('Block built with output volume shape', out.shape)

    def forward(self, x_img):

        out_img = x_img
        # print("input", out_img.shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        # x_flat = (64 x 25 x 24)
        # print('out_img', out_img.shape)
        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)
        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        for idx_layer in range(3):
            out = F.relu(self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out))

        # reshape again and sum
        # print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)

        """f"""
        out = self.post_processing_layer.forward(out)
        out = F.relu(out)
        out = self.output_layer.forward(out)
        # print('Block built with output volume shape', out.shape)
        return out


class _SELayer(nn.Module):
    def __init__(self, input_shape, pooling_type, network_type, pooling_size, num_layers, num_filters):
        super(_SELayer, self).__init__()
        self.input_shape = input_shape
        self.pooling_type = pooling_type
        self.network_type = network_type
        self.pooling_size = pooling_size
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.build_block()

    def build_block(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)  # b, c, h, w
        out = x.clone()
        if self.pooling_type == 'avg_pool':
            out = F.adaptive_avg_pool2d(out, self.pooling_size)  # b, c, pooling_size, pooling_size
        elif self.pooling_type == 'max_pool':
            out = F.adaptive_max_pool2d(out, self.pooling_size)  # b, c, pooling_size, pooling_size
        else:
            raise ModuleNotFoundError('Pooling type cant be found', self.pooling_type)

        if self.network_type == 'fcc_network':
            out = out.squeeze()
            for i in range(self.num_layers - 1):
                self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[-1],
                                                                out_features=self.num_filters)
                out = F.leaky_relu(self.layer_dict['fcc_{}'.format(i)].forward(out))

            self.layer_dict['fcc_output'] = nn.Linear(in_features=out.shape[-1],
                                                      out_features=x.shape[1])
            attention_regions = self.layer_dict['fcc_output'].forward(out).sigmoid()

        elif self.network_type == 'relational_network':
            self.relational_network = BatchRelationalModule(input_shape=out.shape, num_filters=self.num_filters,
                                                            num_layers=self.num_layers, num_outputs=x.shape[1])
            attention_regions = self.relational_network.forward(out).sigmoid()
        else:
            raise ModuleNotFoundError('network type can\'t be found')

        out = x * attention_regions.unsqueeze(2).unsqueeze(2)

        print('Built module', print(type(self)), 'with output shape', out.shape, 'and modules',
              self)

    def forward(self, x):
        out = x.clone()
        if self.pooling_type == 'avg_pool':
            out = F.adaptive_avg_pool2d(out, self.pooling_size)  # b, c, pooling_size, pooling_size
        elif self.pooling_type == 'max_pool':
            out = F.adaptive_max_pool2d(out, self.pooling_size)  # b, c, pooling_size, pooling_size
        else:
            return ModuleNotFoundError('Pooling type cant be found')

        if self.network_type == 'fcc_network':
            out = out.squeeze()
            for i in range(self.num_layers - 1):
                out = F.leaky_relu(self.layer_dict['fcc_{}'.format(i)].forward(out))
            attention_regions = self.layer_dict['fcc_output'].forward(out).sigmoid()

        elif self.network_type == 'relational_network':
            attention_regions = self.relational_network.forward(out).sigmoid()

        out = x * attention_regions.unsqueeze(2).unsqueeze(2)

        return out


class BottleNeckDenseLayer(nn.Sequential):
    def __init__(self, input_shape, growth_rate, drop_rate, attention_pooling_type, attention_network_type,
                 attention_pooling_size,
                 num_attention_layers, num_attention_filters):
        super(BottleNeckDenseLayer, self).__init__()
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.num_attention_layers = num_attention_layers
        self.num_attention_filters = num_attention_filters
        self.attention_pooling_type = attention_pooling_type
        self.attention_network_type = attention_network_type
        self.attention_pooling_size = attention_pooling_size
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['squeeze_excite_attention_layer'] = _SELayer(input_shape=out.shape,
                                                                     pooling_type=self.attention_pooling_type,
                                                                     pooling_size=self.attention_pooling_size,
                                                                     network_type=self.attention_network_type,
                                                                     num_layers=self.num_attention_layers,
                                                                     num_filters=self.num_attention_filters)
        out = self.layer_dict['squeeze_excite_attention_layer'].forward(out)

        self.layer_dict['bottleneck_conv_layer'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.growth_rate,
                                                             padding=0, stride=1, kernel_size=1)
        out = self.layer_dict['bottleneck_conv_layer'].forward(out)
        self.layer_dict['bottleneck_bn_layer'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bottleneck_bn_layer'].forward(out)
        out = F.relu(out)

        self.layer_dict['processing_conv_layer'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.growth_rate,
                                                             padding=1, stride=1, kernel_size=3)
        out = self.layer_dict['processing_conv_layer'].forward(out)
        self.layer_dict['processing_bn_layer'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['processing_bn_layer'].forward(out)
        out = F.relu(out)

        print('Built module', print(type(self)), 'with output shape', out.shape, 'and modules',
              self)

    def forward(self, x):
        out = x
        out = self.layer_dict['squeeze_excite_attention_layer'].forward(out)
        out = self.layer_dict['bottleneck_conv_layer'].forward(out)
        out = self.layer_dict['bottleneck_bn_layer'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['processing_conv_layer'].forward(out)
        out = self.layer_dict['processing_bn_layer'].forward(out)
        out = F.relu(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, input_shape, num_layers, growth_rate, drop_rate,
                 attention_pooling_type, attention_network_type,
                 attention_pooling_size,
                 num_attention_layers, num_attention_filters):
        super(_DenseBlock, self).__init__()
        x = torch.zeros(input_shape)
        out = x
        for i in range(num_layers):
            layer = BottleNeckDenseLayer(input_shape=out.shape, growth_rate=growth_rate, drop_rate=drop_rate,
                                         attention_pooling_type=attention_pooling_type,
                                         attention_network_type=attention_network_type,
                                         attention_pooling_size=attention_pooling_size,
                                         num_attention_layers=num_attention_layers,
                                         num_attention_filters=num_attention_filters
                                         )
            out = layer.forward(out)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self, input_shape, attention_pooling_type, attention_network_type, attention_pooling_size,
                 num_attention_layers, num_attention_filters, growth_rate=12, block_config=(4, 4, 4),
                 compression=0.5,
                 num_init_feature=24, drop_rate=0, num_classes=10, avgpool_size=8):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'


        # First Convolution
        x = torch.zeros(input_shape)
        out = x
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(x.shape[1], num_init_feature, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        out = self.features[0].forward(out)
        # Each denseblock

        num_features = num_init_feature
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(input_shape=out.shape, num_layers=num_layers,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                attention_pooling_type=attention_pooling_type,
                                attention_network_type=attention_network_type,
                                attention_pooling_size=attention_pooling_size,
                                num_attention_layers=num_attention_layers, num_attention_filters=num_attention_filters)
            out = block.forward(out)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                out = trans.forward(out)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.avgpool_size = avgpool_size

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out

