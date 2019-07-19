import os
import sys
from collections import namedtuple

sys.path.append('../')

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
experiment_config_cluster_scripts_dir = '../cluster_experiment_scripts'
experiment_config_local_scripts_dir = '../local_experiment_scripts'
config_list = []
# experiment_name -> autogenerate
exp_idx = 0
for seed in range(0, 3):
    for dataset_name in ['cifar100', 'cifar10']:
        for attention_pooling_type in ['avg_pool', 'max_pool']:
            for attention_network_type in ['relational_network', 'fcc_network']:
                for num_attention_layers in range(1, 4):
                    for attention_pooling_size in [1, 3, 5]:
                        for num_attention_filters in [50]:
                            for conv_bn_relu_type_bottleneck in ['Conv2dNormLeakyReLU', 'HyperConv2dNormLeakyReLU',
                                                                 'SqueezeExciteConv2dNormLeakyReLU',
                                                                 'WeightAttentionalConv2dNormLeakyReLU']:
                                for conv_bn_relu_type_processing in ['Conv2dNormLeakyReLU', 'HyperConv2dNormLeakyReLU',
                                                                     'SqueezeExciteConv2dNormLeakyReLU',
                                                                     'WeightAttentionalConv2dNormLeakyReLU']:
                                    temp_conf = config(
                                        experiment_name='experiment_{dataset_name}_{conv_bn_relu_type_bottleneck}_'
                                                        '{conv_bn_relu_type_processing}_{attention_pooling_type}_'
                                                        '{num_attention_layers}_{attention_pooling_size}_'
                                                        '{num_attention_filters}_{seed}_{exp_idx}'.format(
                                            dataset_name=dataset_name,
                                            conv_bn_relu_type_bottleneck=conv_bn_relu_type_bottleneck,
                                            conv_bn_relu_type_processing=conv_bn_relu_type_processing,
                                            attention_pooling_type=attention_pooling_type, attention_pooling_size=attention_pooling_size,
                                        num_attention_layers=num_attention_layers,
                                        num_attention_filters=num_attention_filters, seed=seed, exp_idx=exp_idx),

                                        dataset_name=dataset_name, seed=seed,
                                        attention_pooling_type=attention_pooling_type,
                                        attention_network_type=attention_network_type,
                                        attention_pooling_size=attention_pooling_size,
                                        num_attention_layers=num_attention_layers,
                                        num_attention_filters=num_attention_filters,
                                        conv_bn_relu_type_bottleneck=conv_bn_relu_type_bottleneck,
                                        conv_bn_relu_type_processing=conv_bn_relu_type_processing)
                                    config_list.append(temp_conf)
                                    exp_idx += 1

# config_list = [
#
#     config(experiment_name='experiment_5_1', dataset_name='cifar100', seed=0,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_5_2', dataset_name='cifar100', seed=1,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_5_3', dataset_name='cifar100', seed=2,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#
#     config(experiment_name='experiment_6_1', dataset_name='cifar100', seed=0,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_6_2', dataset_name='cifar100', seed=1,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_6_3', dataset_name='cifar100', seed=2,
#            attention_pooling_type='max_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#
#     config(experiment_name='experiment_7_1', dataset_name='cifar100', seed=0,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_7_2', dataset_name='cifar100', seed=1,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_7_3', dataset_name='cifar100', seed=2,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=3,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#
#     config(experiment_name='experiment_8_1', dataset_name='cifar100', seed=0,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_8_2', dataset_name='cifar100', seed=1,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
#     config(experiment_name='experiment_8_3', dataset_name='cifar100', seed=2,
#            attention_pooling_type='avg_pool',
#            attention_network_type='relational_network',
#            attention_pooling_size=7,
#            num_attention_layers=4,
#            num_attention_filters=50,
#            conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#            conv_bn_relu_type_processing='Conv2dNormLeakyReLU'),
# ]

for storage_dir in [experiment_config_cluster_scripts_dir, experiment_config_local_scripts_dir]:
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)


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
        if 'local' in template_file:
            target_dir = experiment_config_local_scripts_dir
        else:
            target_dir = experiment_config_cluster_scripts_dir

        filepath = os.path.join(subdir, template_file)

        for config in config_list:
            loaded_template_file = load_template(filepath=filepath)
            config_dict = config._asdict()

            cluster_script_text = fill_template(script_text=loaded_template_file,
                                                config=config_dict)

            cluster_script_name = '{}/{}_{}.sh'.format(target_dir,
                                                       template_file.replace(".sh", ""),
                                                       config.experiment_name)
            cluster_script_name = os.path.abspath(cluster_script_name)
            write_text_to_file(cluster_script_text, filepath=cluster_script_name)

print('generation completed')

# config(experiment_name='experiment_1_1', dataset_name='cifar100', seed=0,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_1_2', dataset_name='cifar100', seed=1,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_1_3', dataset_name='cifar100', seed=2,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#
#
#                config(experiment_name='experiment_2_1', dataset_name='cifar10', seed=0,
#                       attention_pooling_type='avg_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_2_2', dataset_name='cifar10', seed=1,
#                       attention_pooling_type='avg_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_2_3', dataset_name='cifar10', seed=2,
#                       attention_pooling_type='avg_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='WeightAttentionalConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='WeightAttentionalConv2dNormLeakyReLU'),
#
#
#                config(experiment_name='experiment_3_1', dataset_name='cifar100', seed=0,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_3_2', dataset_name='cifar100', seed=1,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_3_3', dataset_name='cifar100', seed=2,
#                       attention_pooling_type='max_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
#
#
#
               # config(experiment_name='experiment_4_1', dataset_name='cifar10', seed=0,
               #        attention_pooling_type='avg_pool',
               #        attention_network_type='relational_network',
               #        attention_pooling_size=7,
               #        num_attention_layers=3,
               #        num_attention_filters=50,
               #        conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
               #        conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_4_2', dataset_name='cifar10', seed=1,
#                       attention_pooling_type='avg_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
#                config(experiment_name='experiment_4_3', dataset_name='cifar10', seed=2,
#                       attention_pooling_type='avg_pool',
#                       attention_network_type='relational_network',
#                       attention_pooling_size=7,
#                       num_attention_layers=3,
#                       num_attention_filters=50,
#                       conv_bn_relu_type_bottleneck='SqueezeExciteConv2dNormLeakyReLU',
#                       conv_bn_relu_type_processing='SqueezeExciteConv2dNormLeakyReLU'),
