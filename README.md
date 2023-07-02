# VisionTransformer-vs-ResNet-CIFAR100
 设计了同ResNet18，ResNet50以及ResNet101，相似参数量的VisionTransformer网络模型，对CIFAR100数据集进行分类。

## 模型介绍
model里面包含了resnet18,resnet34,resnet50等等模型，用于CIFAR100数据集的分类。同时增添了VisionTransformer网络模型,其中包含VisionTransformer1,VisionTransformer2,VisionTransformer3三个结构不同，不同参数量的模型。

## 训练模型
python train.py -net VisionTransformer1 -gpu 

使用默认设置训练VisionTransformer1模型

python train.py -net VisionTransformer2 -gpu -warm 40 -b 64 -lr 0.05

使用预热操作，设置为40步，batch_size设置为64，初始学习率设置为0.05训练VisionTransformer2模型

## 可视化训练过程
tensorboard --logdir='runs' --port=6006 --host='localhost'

会显示训练集loss曲线，测试集loss曲线和测试集accuracy曲线，以及网络各层参数。

## 测试模型
python test.py -net VisionTransformer2 -weights path.pth

后面为模型权重文件。

## 训练结果

训练得到的模型保存在百度网盘上。

链接：https://pan.baidu.com/s/1gS5R45Wso29ilUXwt9vn3w?pwd=ymqh

提取码: ymqh
                
                

