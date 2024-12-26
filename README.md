# YOLO-from-paper
Full YOLO computer vision model annotated code from the [research paper](https://arxiv.org/pdf/1506.02640). 

## 1. Implementing the model
![image](https://github.com/user-attachments/assets/88100a78-ff3b-45a2-956b-86bb71f0548f)

Two parts can be considered separatly : 
- the darknet (not sure why it is called this way), consisting of the first 24 convolutionnal layers.
- the 2 fully-connected layers.

The **darknet** implementation is quite straightforward as it is fully explicited in the figure above.
One nice hack I discovered while coding this project is using a list to instanciate the model's layers. Props to Aladdin Persson for showcasing it in his [YOLO implementation video](https://www.youtube.com/watch?v=n9_XyCGr-MI&t=24s). We create a list using tuples for layer arguments, lists for redundant blocks of layers and characters for Max-Pooling layers. We then define a *_create_layers* function acting as a parser based on instance types.
This will make it way easier when creating YOLO architecture variants such as tiny-YOLO for embedded systems or for pretraining the model.

The **fully-connected** part requires a bit of reading.
> Image is divided into an S x S and for each grid cell predicts B bounding boxes, confidence for those boxes C class probabilities encoded as an S x S x (B x 5 + C) tensor.  
> [...]  
> For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor.

The prediction tensor for each cell is B x 5 + C, "5" corresponds to the coordinates of the bounding box plus the confidence score : c, x, y, w, h.
> A dropout layer with rate = .5 after the first connected layer prevents co-adaptation between layers.

We add a dropout between the two layers.

![image](https://github.com/user-attachments/assets/c216293e-bfa6-4953-8117-455846e1c7ad)

Leaky ReLu is used for all activations except for the output layer which is linear as it is considered a regression problem.

  
*Note that the model is different during pretraining, this part is better described in the "training" folder.*

## 2. Loss function
This was for me the hardest part about the project but ended up being an interesting challenge. YOLO's loss function is quite extensive, understanding and translating such a large mathematical expression and make it work was "fun".

## 3. Training 

At the moment I don't have access to the hardware necessary to train YOLO as it was done back in 2017. Detailed steps from the paper are described in the *training* folder README. Briefly, training was done in several times. First the feature extractor part of the model consisting in the first 20 layers, an average pooling layer and a fully connected one was trained on ImageNet (224x224) resolution. Then the weights were transfered to the full model and trained on Pascal-VOC (448x448).  

The feature extractor training lasted one week on a Titan X GPU. A quicker solution I aim at implementing and descibed on this [blogpost](https://medium.com/@m.khan/implementing-yolo-using-resnet-as-feature-extractor-5857f9da5014) is to use a pretrained resnet-50 as the feature extractor and train the YOLO classfier head with Pascal-VOC.  

This approach will be studied in an other repo as I want to keep this one as a "clean" YOLO v1 implementation, true to the original. Nonetheless, the implementation was checked by overfitting a 100 example sample from the Pascal-VOC dataset.



