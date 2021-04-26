# Head pose estimation
This repo is a head pose estimation on video from webcam based on [this tutorial](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/).

## How to run

You can run `main.py` directly using:

```python main.py```

or using docker:

Build the image using:

```docker build -t head-pose-estimation .```

Then run the image using:

```docker run head-pose-estimation```