# First year project 2023

## Purpose
The main purpose of the project is to predict if a lesion is cancerous or not.

## Features
The folder features contains a csv file with all the data we have gathered from a public data-set that can be found here: https://data.mendeley.com/datasets/zr7vgbcyr2/1

## Utilities
Our code uses another csv that contains various colors that can help us categorize each segment of the lesion by its color.

# How to use

You need to have a script that uses matlibplot library to read one image that represents a lesion and another one that represents its mask. After reading them you can print out the results of the function "classify(your_image, your_mask)" (change "your_image" and "your_mask" into your image and respectively your mask variables) from "03_evaluate_classifier.py". You will see two values, the first one representing the cancerous state(0 for negative and 1 for positive) and the other one is the probability of it being non-cancerous or cancerous. There is also the predicted label after setting the threshold for the probability of 0 to 0.6 (commented).
