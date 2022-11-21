import cv2
import numpy as np


import math


def check_even(number):
    if number % 2 == 0:
        return True
    else:
        return False


def process_image(img):
    w, h = img.shape

    new_w = 32
    new_h = int(h * (new_w / w))

    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype("float32")

    if w < 32:
        add_zeros = np.full((32 - 2, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    if h < 128:
        add_zeros = np.full((w, 128 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)

    img = img / 255
    return img
