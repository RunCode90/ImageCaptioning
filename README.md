# ImageCaptioning

Requirements 

    python 3.6
    torch 0.4.1
    h5py 2.8
    tqdm 4.26
    nltk 3.3

Instructions

    Download the COCO 2014 dataset from here. In particualr, you'll need the 2014 Training, Validation and Testing images, as well as the 2014 Train/Val annotations.

    Download Karpathy's Train/Val/Test Split. You may download it from here.

    If you want to do evaluation on COCO, make sure to download the COCO API from here if your on Linux or from here if your on Windows. Then download the COCO caption toolkit from here and re-name the folder to cococaption. (This also requires java. Simply dowload it from here if you don't have it).

 Training

Simply run python main.py to start training.

    python3.6 train.py

The dataset used for learning and evaluation is the MSCOCO Image captioning challenge dataset. It is split into training, validation and test sets using the popular Karpathy splits. This split contains 113,287 training images with five captions each, and 5K images respectively for validation and testing. Teacher forcing is used to aid convergence during training. Teacher forcing is a method of training sequence based tasks on recurrent neural networks by using the actual or expected output from the training dataset at the current time step y(t) as input in the next time step X(t+1), rather than the output generated by the network. Teacher forcing addresses slow convergence and instability when training recurrent networks that use model output from a prior time step as an input.

Weight normalization was found to prevent the model from overfitting and is used liberally for all fully connected layers.
Gradients are clipped during training to prevent gradient explosion that is not uncommon with LSTMs. The attention dimensions, word embedding dimension and hidden dimensions of the LSTMs are set to 1024.


 Testing

Updating
