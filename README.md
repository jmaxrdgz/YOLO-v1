# YOLO-from-paper
Full YOLO computer vision model annotated code from the [research paper](https://arxiv.org/pdf/1506.02640). 
Next versions once v1 is finished (will be keeping all versions).

## 1. Implementing the model
Model implementation is quite straightforward as it is fully defined in the paper. 
Two parts can be considered seperatly : 
- the darknet (not sure why it is called this way, neither where I hold the information from)
- the fully-connected layers

Hence 2 separate functions are used for visibility issues.

One cool part I discovered while coding this project is using a list to instanciate the model's layers. Props to Aladdin Persson for showcasing it in his [YOLO implementation video](https://www.youtube.com/watch?v=n9_XyCGr-MI&t=24s). There are 24 convolutional layers wich makes it very redundant to define. Instead we create a list using tuples for layer arguments, lists for redundant blocks of layers and characters for Max-Pooling layers. We then define a *_create_layers* function acting as a parser based on instance types int the declaration list to instanciate the model. This will make it way easier when creating YOLO architecture variants such as tiny-YOLO for embedded systems.

## 2. Loss function
This was for me the hardest part about the project but ended up being an interesting challenge. YOLO's loss function is quite extensive, understanding and translating such a large mathematical expression and make it work was "fun".

