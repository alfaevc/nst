import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

import time
from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgsize = (480, 300)

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

def Img2Tensor(img):
    return toTensor(img).unsqueeze(0).to(device)

def Tensor2Img(tensor):
    return np.array(toPIL(tensor.squeeze().cpu()))

def loadImage(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, imgsize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def loadVid(path, numframes=-1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video file failed to load! Maybe check the path or file integrity.')
    ret, frame = cap.read()
    if not ret:
        raise Exception('Failed to read first frame')
    frames = []
    while cap.isOpened() and (numframes == -1 or len(frames) < numframes):
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, imgsize)
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        if len(frames) % 100 == 0: print('Progress: {} frames loaded'.format(len(frames)))
    cap.release()
    return np.array(frames)[::2]

def saveVid(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class ETATracker():

    def __init__(self, iters):
        self.iters = iters

    def start(self):
        self.times = []
        self.prevtime = time.time()
        print('Total # of frames: {}'.format(self.iters))

    def timestamp(self):
        assert(self.iters > 0)
        self.iters -= 1
        delta = time.time() - self.prevtime
        self.prevtime = time.time()
        self.times.append(delta)

        print('------')
        print('previous time: {:.2f}'.format(delta))
        avg = np.mean(self.times)
        print('average time: {:.2f}'.format(avg))
        print('eta: {}'.format(timedelta(seconds=int(avg)*self.iters)))
        print('------\n')
