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

The dataset used for learning and evaluation is the MSCOCO Image captioning challenge dataset. It is split into training, validation and test sets using the popular Karpathy splits. This split contains 113,287 training images with five captions each, and 5K images respectively for validation and testing. 




 Testing

Updating
