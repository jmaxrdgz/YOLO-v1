# YOLO's training 
#### Roadmap to fit the paper's training 

*Training will be performed on Google Colab using the free GPU T4 (because students' budgets can barely afford the "G" in GPU) and pretrained resnet-50 backbone instead of the original backbone.*

First step was to test if the code was running properly. I trained it on 100 epochs with the "100examples" csv file. It overfitted succesfully.

## Training in the paper
### 1. Pretraining 
Pretrain on ImageNet (1000 class) for a week (until 88% on validation set), input resolution is 224x224.
First 20 Convolutional layers + 1 Average-pooling layer + 1 Fully-connected layer.

### 2. Training the full model with increased resolution
Train for around 135 epochs on Pascal VOC 2007 + 2012 with 448x448 resolution.
Add 4 Convolutional layers + 2 Fully-connected layers (All initialized with randow weights).

| Batch size | Momentum | Weight Decay |
|:----------:|:----------:|:----------:|
| 64 | 0.9 | 0.0005 | 

| Epochs | Lr |
|----------|----------|
| not specified | 1e-3 -> 1e-2 |
| 0 - 75 | 1e-2 |
| 75 - 105 | 1e-3 |
| 105 - 135 | 1e-4 |

Data augementation is used :
- 20% scaling and translation
- 1.5 factor exposure and saturation
