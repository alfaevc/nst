import torch
import torch.nn
import torch.nn.functional as F
import cv2
import numpy as np

import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def none(*args, **kwargs):
    return torch.tensor(0.0).to(device)

def mse(*args, **kwargs):
    curr_frame = kwargs['curr_frame']
    prev_frame = kwargs['prev_frame']
    return F.mse_loss(prev_frame, curr_frame)

def mse_of_residual(*args, **kwargs):
    curr_frame = kwargs['curr_frame']
    prev_frame = kwargs['prev_frame']
    curr_content = kwargs['curr_content']
    prev_content = kwargs['prev_content']
    return F.mse_loss(curr_frame-prev_frame, curr_content-prev_content)

def scaled_residual_mse(*args, **kwargs):
    curr_frame = kwargs['curr_frame']
    prev_frame = kwargs['prev_frame']
    curr_content = kwargs['curr_content']
    prev_content = kwargs['prev_content']

    scale = torch.tanh(torch.exp(-F.mse_loss(curr_content, prev_content)))
    output_diff = mse(**kwargs)
    mse_resid = mse_of_residual(**kwargs)

    return scale * output_diff + mse_resid

# This function is borrowed from https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

saved_flow = [-1, None]

def optical_flow(*args, **kwargs):
    if saved_flow[0] != kwargs['frame_idx']:
        prev_frame = utils.Tensor2Img(kwargs['prev_frame'].clamp(0,1).cpu())
        curr_content = utils.Tensor2Img(kwargs['curr_content'].clamp(0,1).cpu())
        prev_content = utils.Tensor2Img(kwargs['prev_content'].clamp(0,1).cpu())

        curr_content = cv2.cvtColor(curr_content, cv2.COLOR_RGB2GRAY)
        prev_content = cv2.cvtColor(prev_content, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_content, curr_content, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        predicted = warp_flow(prev_frame, flow)
        saved_flow[0] = kwargs['frame_idx']
        saved_flow[1] = utils.Img2Tensor(predicted).to(device)
    
    curr_frame = kwargs['curr_frame']
    return F.mse_loss(saved_flow[1], curr_frame)