

import os

import cv2
import torch

from model import UNET

model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("./my_checkpoint.pth.tar")["state_dict"])
model.eval()


def predict(img):
    with torch.no_grad():
        pred = model(img)
        cv2.imshow("Pred", pred)


img = cv2.imread("1315_m.PNG")
predict(img)