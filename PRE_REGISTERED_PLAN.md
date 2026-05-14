# Pre-registered Plan

## Aim

Build an image-only classifier that predicts the position of a frame in a five-image visual story.

## Dataset Plan

The dataset folder contains 1,370 images. I will sort the image filenames, group every five consecutive images as one story, and assign labels 1 to 5 according to position inside the group. I will split by story so that images from the same story do not appear in both training and validation.

## Model Plan

The baseline model will be a small CNN with three convolution blocks and a five-class output layer. The loss function will be `CrossEntropyLoss`, and the evaluation metric will be accuracy.

## Hypothesis

The model may struggle because a single frame often has weak temporal information. I expect regularisation and normalization to help if the model overfits, but I expect increasing model size alone to have limited benefit.

## Experiments

1. Baseline CNN.
2. Add dropout.
3. Increase filter count.
4. Change kernel size from 3 to 5.
5. Add batch normalization.
6. Add a fourth convolution block.

Each variation changes exactly one model aspect compared with the baseline.
