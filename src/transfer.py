import torch
import torch.optim as optim
import cv2
import numpy as np

import gatys, losses, utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfer_image(content_img, style_img, epochs=20, coh_loss_fn=losses.none,
                   content_weight=0.85, style_weight=6e5, coherence_weight=40, **kwargs):
    content_img = utils.Img2Tensor(content_img).to(device)
    style_img = utils.Img2Tensor(style_img).to(device)

    model = gatys.Model().to(device)
    model.set_content(content_img)
    model.set_style(style_img)

    input_img = content_img.clone()
    input_img.requires_grad_(True)

    optimizer = optim.LBFGS([input_img])
    checkpoints = []

    for i in range(epochs):

        loss_stats = [None]*4

        def closure():
            input_img.data.clamp_(0, 1)
            content_loss, style_loss = model(input_img)
            coherence_loss = coh_loss_fn(curr_frame=input_img, **kwargs)

            content_loss *= content_weight
            style_loss *= style_weight
            coherence_loss *= coherence_weight

            combined_loss = content_loss + style_loss + coherence_loss

            loss_stats[0] = content_loss
            loss_stats[1] = style_loss
            loss_stats[2] = coherence_loss
            loss_stats[3] = combined_loss

            optimizer.zero_grad()
            combined_loss.backward()

            return combined_loss

        optimizer.step(closure)

        checkpoints.append((loss_stats[3].item(), input_img.clone()))

        print("epoch {}:".format(i+1))
        print('Content Loss: {:4f}, Style Loss: {:4f}, Coherence Loss: {:4f}, Combined Loss: {:4f}'.format(
            loss_stats[0].item(), loss_stats[1].item(), loss_stats[2].item(), loss_stats[3].item()))
        print()

    best_img = min(checkpoints, key=lambda x:x[0])
    print('best loss = {:4f}'.format(best_img[0]))
    best_img = best_img[1]

    return utils.Tensor2Img(best_img.clamp(0,1).cpu())



def transfer_video(video_path, style_path, save_path, start_frame=None, end_frame=None, loss_fn=losses.mse, **kwargs):
    vid = utils.loadVid(video_path)
    h, w = vid.shape[2:]
    if start_frame is None: start_frame = 0
    if end_frame is None: end_frame = len(vid)
    vid = vid[start_frame:end_frame]

    style_raw = utils.loadImage(style_path)
    
    eta_tracker = utils.ETATracker(len(vid))
    eta_tracker.start()
    output_frames = []

    for i in range(len(vid)):
        print('frame: {}'.format(i+1))

        if i == 0:
            output = transfer_image(vid[i], style_raw, epochs=8, coh_loss_fn=losses.none, **kwargs)
        else:
            kwargs['frame_idx'] = i
            kwargs['prev_frame'] = utils.Img2Tensor(output_frames[i-1]).to(device)
            kwargs['curr_content'] = utils.Img2Tensor(vid[i]).to(device)
            kwargs['prev_content'] = utils.Img2Tensor(vid[i-1]).to(device)

            output = transfer_image(vid[i], style_raw, epochs=8, coh_loss_fn=loss_fn, **kwargs)

        output_frames.append(output)

        eta_tracker.timestamp()

    utils.saveVid(save_path, np.array(output_frames))
