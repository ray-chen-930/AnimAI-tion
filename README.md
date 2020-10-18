# AnimAI-Tion

Creating machine learning-generated animations from Anime with the First Order Motion Model for Image Animation (https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation).

## Fine-tuning the First Order Motion Model

The First Order Motion Model has already been fine-tuned for facial animations, but only on real faces in the celebs dataset.  As a result, using the weights trained on the celebs data set does not translate well over to faces drawn in an anime style.  To produce better results, the First Order Motion Model must first be fine-tuned, which can be done using clips of anime.

### Getting these clips

The crop-video.py file is a script that can be run to first use a Yolo algorithm to detect faces of anime characters in a video file and then crop the video to the bounding boxes and output each frame as a png file in a folder separated by trajectories.  A trajectory is a single bounding box over a scene, so there can be multiple trajectories in the same scene.  These files of images can be used to fine-tune the First Order Motion Model.


## Some Results:

From left to right: the source image, driving video, model fine-tuned on celebrity data set, model fine-tuned on 1 episode of K-On!, and model fine-tuned on 5 episodes of K-On!

![Screenshot](results/sig_better.gif)
![Screenshot](results/goodface_badmove.gif)
![Screenshot](results/head_turn.gif)
![Screenshot](results/eyes.gif)
