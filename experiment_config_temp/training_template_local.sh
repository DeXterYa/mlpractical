cd ..

python train_evaluate_emnist_classification_system.py --batch_size 50 --continue_from_epoch -1 --seed $seed$ \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --num_epochs 100 --experiment_name "$experiment_name$" \
                                                      --growth_rate 24 --block_config "(5, 5, 5, 5)" --compression 0.5 \
                                                      --num_init_feature 24 --bn_size 8 --drop_rate 0.2 --avgpool_size 8 \
                                                      --attention_pooling_type "$attention_pooling_type$" --dataset_name "$dataset_name$"\
                                                      --attention_network_type "$attention_network_type$" --attention_pooling_size $attention_pooling_size$\
                                                      --num_attention_layers $num_attention_layers$  --num_attention_filters $num_attention_filters$\
                                                      --conv_bn_relu_type_bottleneck "$conv_bn_relu_type_bottleneck$" \
                                                      --conv_bn_relu_type_processing "$conv_bn_relu_type_processing$" \
                                                      --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0.00001