import os
from collections import namedtuple
import sys
sys.path.append('../')
from model_architectures import *

config = namedtuple('config', ['experiment_name',
                               'dataset_name',
                               'seed',
                               'attention_pooling_type',
                               'attention_network_type',
                               'attention_pooling_size',
                               'num_attention_layers',
                               'num_attention_filters',
                               'conv_bn_relu_type_bottleneck',
                               'conv_bn_relu_type_processing'])

experiment_templates_json_dir = '../experiment_config_temp/'
experiment_config_target_json_dir = '../cluster_experiment_scripts'

config_list = [config(experiment_name='experiment_1_1', dataset_name='cifar10', seed=0,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_1_2', dataset_name='cifar10', seed=1,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_1_3', dataset_name='cifar10', seed=2,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_2_1', dataset_name='cifar10', seed=0,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=2,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_2_2', dataset_name='cifar10', seed=1,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=2,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_2_3', dataset_name='cifar10', seed=2,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=2,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_3_1', dataset_name='cifar10', seed=0,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=3,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_3_2', dataset_name='cifar10', seed=1,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=3,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_3_3', dataset_name='cifar10', seed=2,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=3,
                      num_attention_layers=3,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_4_1', dataset_name='cifar10', seed=0,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=5,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_4_2', dataset_name='cifar10', seed=1,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=5,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               config(experiment_name='experiment_4_3', dataset_name='cifar10', seed=2,
                      attention_pooling_type='avg_pool',
                      attention_network_type='fcc_network',
                      attention_pooling_size=1,
                      num_attention_layers=5,
                      num_attention_filters=50,
                      conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
                      conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
               ]

if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)


def fill_template(script_text, config):
    for key, value in config.items():
        script_text = script_text.replace('${}$'.format(key), str(value))
    return script_text


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.read()

    return template


def write_text_to_file(text, filepath):
    with open(filepath, mode='w') as filewrite:
        filewrite.write(text)


for subdir, dir, files in os.walk(experiment_templates_json_dir):
    for template_file in files:
        filepath = os.path.join(subdir, template_file)

        for config in config_list:
            loaded_template_file = load_template(filepath=filepath)
            config_dict = config._asdict()

            cluster_script_text = fill_template(script_text=loaded_template_file,
                                                config=config_dict)

            cluster_script_name = '{}/{}_{}.sh'.format(experiment_config_target_json_dir,
                                                       template_file.replace(".sh", ""),
                                                       config.experiment_name)
            cluster_script_name = os.path.abspath(cluster_script_name)
            write_text_to_file(cluster_script_text, filepath=cluster_script_name)


print('generation completed')