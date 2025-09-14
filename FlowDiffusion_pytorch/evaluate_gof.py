import torch.nn.functional as F
import math
from core.datasets_return_dict import KITTI, MpiSintel
import torch.utils.data as data
from local_diffusers.pipelines.DDPM import DDPMPipeline
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from evaluate_diffusers_warprefine import compute_grid_indices, compute_weight, backwarp, save

def evaluate_gof(pipeline, img1, img2):
    os.makedirs("flow_images_gt", exist_ok=True)
    IMAGE_SIZE = None
    TRAIN_SIZE = [320, 448]
    min_overlap = 48
    min_overlap_h = 0
    sigma = 0.05
    pipeline.unet = pipeline.unet.to(torch.bfloat16)

    results = {}



    B, _, H, W = img1.shape
    if IMAGE_SIZE is None or H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
        print(f"replace {IMAGE_SIZE} with [{H}, {W}]")
        IMAGE_SIZE = [H, W]
        hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap=min_overlap,min_overlap_h=min_overlap_h)
        weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    img1 = 2 * (img1 / 255.0) - 1.0
    img2 = 2 * (img2 / 255.0) - 1.0

    resized_image1 = F.interpolate(img1, TRAIN_SIZE, mode='bicubic', align_corners=True)
    resized_image2 = F.interpolate(img2, TRAIN_SIZE, mode='bicubic', align_corners=True)
    inputs = torch.cat([resized_image1, resized_image2], dim=1).repeat(1,1,1,1)
    resized_flow = pipeline(
        inputs=inputs.to(torch.bfloat16),
        batch_size=inputs.shape[0],
        num_inference_steps=args.ddpm_num_steps,
        output_type="tensor",
        normalize=args.normalize_range,
    ).images.to(torch.float32)
    # print(np.mean(np.var(resized_flow.cpu().numpy(),axis=0)))
    resized_flow = F.interpolate(resized_flow, IMAGE_SIZE, mode='bicubic', align_corners=True) * \
                    torch.tensor([W / TRAIN_SIZE[1], H / TRAIN_SIZE[0]]).view(1, 2, 1, 1).cuda()

    warpimg1 = backwarp(img2, resized_flow)

    flows = 0
    flow_count = 0

    image1_tiles = []
    image2_tiles = []
    for idx, (h, w) in enumerate(hws):
        image1_tiles.append(img1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
        image2_tiles.append(warpimg1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])

    inputs = torch.cat([torch.cat(image1_tiles, dim=0), torch.cat(image2_tiles, dim=0)], dim=1)
    flow_pre_total = pipeline(
        inputs=inputs.to(torch.bfloat16),
        batch_size=inputs.shape[0],
        num_inference_steps=32,
        output_type="tensor",
        normalize=args.normalize_range,
    ).images

    for idx, (h, w) in enumerate(hws):
        flow_pre = flow_pre_total[idx * B:(idx + 1) * B]
        padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
        flows += F.pad(flow_pre * weights[idx], padding)
        flow_count += F.pad(weights[idx], padding)

    flow = flows / flow_count + resized_flow
    # print(epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt().item())
    return flow
    # return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_path', help="restore pipeline")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 448])
    parser.add_argument('--train_batch_size', type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument('--ddpm_num_steps', type=int, default=64)
    parser.add_argument("--normalize_range", action="store_true",
                        help="Whether to normalize the flow range into [-1,1].")
    parser.add_argument('--validation', type=str, nargs='+')
    args = parser.parse_args()
    # TODO
    # maybe need set clip_sample to True
    pipeline = DDPMPipeline.from_pretrained(args.pipeline_path).to('cuda')
    folder_path = "/home/xuhang/code/FlowDiffusion_pytorch/datasets/static_40k_png_1_of_4/static_40k_png_1_of_4"
    os.makedirs("flow_images_gof_none", exist_ok=True)
    for subfolder in sorted(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, subfolder)
        img1_path = os.path.join(scene_path, "im0.png")
        img2_path = os.path.join(scene_path, "im1.png")
        # flow_path = os.path.join(scene_path, "gyro_homo.npy")
        img1 = torch.from_numpy(np.array(Image.open(img1_path))).permute(2, 0, 1).unsqueeze(0).cuda()
        img2 = torch.from_numpy(np.array(Image.open(img2_path))).permute(2, 0, 1).unsqueeze(0).cuda()
        flow = evaluate_gof(pipeline, img1, img2)
        flow_image = save(flow.cpu().numpy())
        flow_image.save(f"flow_images_gof_none/{subfolder}.png")
        # flow = np.load(flow_path)
        # print(flow.shape)
        # flow = flow.transpose(1, 2, 0)
