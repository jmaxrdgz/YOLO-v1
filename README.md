# YOLO-from-paper
Full YOLO computer vision model annotated code from the [research paper](https://arxiv.org/pdf/1506.02640). 
Next versions once v1 is finished (will be keeping all versions).

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

