import transfer, losses

video_path = ''
style_img_path = ''
output_path = ''

transfer.transfer_video(video_path, style_img_path, output_path, start_frame=0, end_frame=90,
                        loss_fn=losses.scaled_residual_mse, style_weight=600000, content_weight=0.85, coherence_weight=40)