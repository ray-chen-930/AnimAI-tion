# AnimAI-Tion

Creating machine learning-generated animations from Anime with the First Order Motion Model for Image Animation (https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation).

## Table of Contents
  * [Installation](#installation)
  * [Fine-tuning the First Order Motion Model](#fine-tuning-the-first-order-motion-model)
  * [Some results](#some-results)

## Installation

Clone into directory
```
git clone https://github.com/ray-chen-930/AnimAI-tion.git
cd AnimAI-tion
```

### Model Weights

The weights used for YOLOv5 small and First Order Motion Model can be downloaded from here:
https://drive.google.com/drive/folders/108Dbrx9URFKkt_07WX4lsrr5fdeqHeCM

Create a folder 'weights' in the directory, and save the weights there
├── AnimAI-tion
│   ├── Fomm
|   ├── yolov5
|   ├── weights             # save the weights in this folder!
|   ├── data_train_yolo
|   ├── demo_save_results.ipynb
│   └── ...

### Try your own image and video!

Open demo_save_results.ipynb and input a path to your own source image and driving video! Note that the weights used in the model work best on anime style faces and facial animations.

## Fine-tuning the First Order Motion Model

The First Order Motion Model has already been fine-tuned for facial animations, but only on real faces in the celebs dataset.  As a result, using the weights trained on the celebs data set does not translate well over to faces drawn in an anime style.  To produce better results, the First Order Motion Model must first be fine-tuned, which can be done using clips of anime.  Weights for two separately fine-tuned models can be found in the link above, one trained on a single episode and one trained on 5 episodes.

### Getting clips used to fine-tune the First Order Motion Model

The crop-vid-yolov5.py file is a script that can be run to first use YOLOv5 small (https://github.com/ultralytics/yolov5) to detect faces of anime characters in a video file and then crop the video to the bounding boxes and output each frame as a png file in a folder separated by trajectories.  A trajectory is a single bounding box over a scene, so there can be multiple trajectories in the same scene.  These files of images can be used to fine-tune the First Order Motion Model.

Along with yolo predicted bounding boxes, other methods can be used improve the crops outputted.  Such methods implemented include scene detection, checking for low resolution crops, and checking for scenes where a face isn't moving.

The crop-video.py file is similar to crop-vid-yolov5.py but instead uses Yolo version 3 and, unfortunately, the weights for this model are proprietary to Spellbrush.

### Fine-tuning the Yolo V5 model

A data set can be downloaded from https://www.kaggle.com/shihkuanchen/kon-characters that contains around 500 images and their respective bounding boxes in text files for faces.  Since these bounding boxes include classes for different characters, clean_kon_bbox_data.py can be run to change the classes in all the text files to a '0' which will represent the presence of a face.  The split_train_val.py file can then be run to randomly split the images and their respective text files into a training and validation set in their respective directories.  These files are in the data_train_yolo folder.

These files can then be used to train a yolov5 model with yolov5/train.py, which, as previously mentioned, is used in crop-vid-yolo5.py.  The fine-tuned weights for this model can be downloded from the google drive link above.

## Some Results:

From left to right: the source image, driving video, model fine-tuned on celebrity data set, model fine-tuned on 1 episode of K-On!, and model fine-tuned on 5 episodes of K-On!

![Screenshot](results/sig_better.gif)
![Screenshot](results/goodface_badmove.gif)
![Screenshot](results/head_turn.gif)
![Screenshot](results/eyes.gif)
