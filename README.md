# Multi-label Image Classification

Contains code for training and testing CNNs for multi-label image classification using various multi-label loss functions: Softmax, Sigmoid, Pairwise Ranking, WARP, LSEP using Tensorflow. Codebase follows Tensorflow's image classification tutorial using _slim_, and incorporates custom loss functions for multi-labels. 

## Requirements
- Tensorflow 1.3
- Tensorflow Slim
- Python 2
- Numpy

## Extracting CNN features

Extract CNN features of images from various models like vgg, inception, resnet and save them in a matfile. Run the following, having changed any needed arguments.

```
dataset_dir=/home/ayushi/Git/research/dataset/nuswide/images/Flickr
checkpoint_path=../data/pretrained/vgg_16.ckpt
eval_file_image_list=../data/nuswide/nus1_train_list.txt
eval_file_image_features=../data/nuswide/net-vgg16/nus1_test_vgg16.mat
python extract.py \
    --dataset_dir=${dataset_dir} \
    --model_name=vgg_16 \
    --checkpoint_path=${checkpoint_path} \
    --bottleneck_scope=PreLogitsFlatten \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --eval_file_image_list=${eval_file_image_list} \
    --eval_file_image_features=${eval_file_image_features} \
    --num_classes=81 \
    --bottleneck_shape=4096 \
    --batch_size=10 
```
where, _dataset_dir_ refers to the directory which contains all the dataset images, _checkpoint_path_ refers to the checkpoint file that can be downloaded from Tensorflow's checkpoint releases, _eval_file_image_list_ contains the list of image names and _eval_file_image_feaures_ refers to the matfile where the extracted features will be saved. 

## Training CNN

### Train with end-to-end network from images

Data Setup: 

### Train with extracted CNN features(faster)

## Testing and Evaluation
