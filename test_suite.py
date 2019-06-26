from model_architectures import *

x = torch.zeros(32, 64, 28, 28)
# test_module = BottleNeckDenseLayer(input_shape=x.shape, growth_rate=32, drop_rate=0.,
#                                    attention_network_type='relational_network', attention_pooling_size=5,
#                                    attention_pooling_type='avg_pool', num_attention_filters=32, num_attention_layers=3)
test_module = DenseNet(input_shape=x.shape, drop_rate=0.,
                       attention_network_type='relational_network', attention_pooling_size=5,
                       attention_pooling_type='avg_pool', num_attention_filters=32, num_attention_layers=3)
out = test_module.forward(x)
print(out.shape)

# experiments : 3 independent runs on cifar10/100,
# fcc_network : avg_pool with pool_size 1, max_pool with pool size 1, avg_pool with size 5
# relational_network: avg_pool with pool_size 3, 5, 7 and then repeat with max_pool