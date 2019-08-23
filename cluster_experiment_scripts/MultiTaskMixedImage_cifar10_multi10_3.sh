#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..

python train_evaluate_emnist_classification_system.py --batch_size 250 --continue_from_epoch -1 --seed 2 \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --num_epochs 100 --experiment_name "MultiTaskMixedImage_cifar10_multi10_3" \
                                                      --growth_rate 24 --block_config "(5, 5, 5, 5)" --compression 0.5 \
                                                      --num_init_feature 24 --bn_size 8 --drop_rate 0.2 --avgpool_size 8 \
                                                      --attention_pooling_type "avg_pool" --dataset_name "cifar10_mixed"\
                                                      --attention_network_type "fcc_network" --attention_pooling_size 1\
                                                      --num_attention_layers 3  --num_attention_filters 50\
                                                      --conv_bn_relu_type_bottleneck "Conv2dNormLeakyReLU" \
                                                      --conv_bn_relu_type_processing "Conv2dNormLeakyReLU" \
                                                      --num_images_per_input 4 --multi 1.0\
                                                      --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0.001