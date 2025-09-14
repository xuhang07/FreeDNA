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
backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0],
                                                                                                           -1,
                                                                                                           tenFlow.shape[2],
                                                                                                           -1).cuda()
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0],
                                                                                                         -1, -1,
                                                                                                         tenFlow.shape[
                                                                                                             3]).cuda()

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bicubic', padding_mode='border', align_corners=True)


def compute_grid_indices(image_shape, patch_size, min_overlap=20, min_overlap_h=20):
    if min_overlap_h >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap_h))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))[:5]
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    # unique
    hs = np.unique(hs)
    # ws.append(32)
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])

    return patch_weights
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
def save(flow):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    u = flow[0][0]
    v = flow[0][1]
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return Image.fromarray(flow_image)

@torch.no_grad()
def validate_kitti(pipeline, args=None, sigma=0.05, start_t=4):
    IMAGE_SIZE = None
    TRAIN_SIZE = [320, 448]
    min_overlap = 250

    pipeline.unet = pipeline.unet.to(torch.bfloat16)
    val_dataset = KITTI(split='training')
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    out_list, epe_list = [], []
    i=0
    for batch in tqdm(val_loader):
        i+=1
        if i>5:
            break
        for k in batch:
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].cuda()

        B, _, H, W = batch["image0"].shape
        if IMAGE_SIZE is None or H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with [{H}, {W}]")
            IMAGE_SIZE = [H, W]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap=min_overlap)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        batch["image0"] = 2 * (batch["image0"] / 255.0) - 1.0
        batch["image1"] = 2 * (batch["image1"] / 255.0) - 1.0

        resized_image1 = F.interpolate(batch["image0"], TRAIN_SIZE, mode='bicubic', align_corners=True)
        resized_image2 = F.interpolate(batch["image1"], TRAIN_SIZE, mode='bicubic', align_corners=True)
        inputs = torch.cat([resized_image1, resized_image2], dim=1)
        resized_flow = pipeline(
            inputs=inputs.to(torch.bfloat16),
            batch_size=inputs.shape[0],
            num_inference_steps=args.ddpm_num_steps,
            output_type="tensor",
            normalize=args.normalize_range
        ).images.to(torch.float32)

        resized_flow = F.interpolate(resized_flow, IMAGE_SIZE, mode='bicubic', align_corners=True) * \
               torch.tensor([W / TRAIN_SIZE[1], H / TRAIN_SIZE[0]]).view(1, 2, 1, 1).cuda()

        warpimg1 = backwarp(batch['image1'], resized_flow)

        flows = 0
        flow_count = 0

        image1_tiles = []
        image2_tiles = []
        for idx, (h, w) in enumerate(hws):
            image1_tiles.append(batch["image0"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
            image2_tiles.append(warpimg1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])

        inputs = torch.cat([torch.cat(image1_tiles, dim=0), torch.cat(image2_tiles, dim=0)], dim=1)
        flow_pre_total = pipeline(
            inputs=inputs.to(torch.bfloat16),
            batch_size=inputs.shape[0],
            num_inference_steps=start_t,
            output_type="tensor",
            normalize=args.normalize_range,
        ).images

        for idx, (h, w) in enumerate(hws):
            flow_pre = flow_pre_total[idx*B:(idx+1)*B]
            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow = flows / flow_count + resized_flow

        epe = torch.sum((flow - batch['target']) ** 2, dim=1).sqrt()
        mag = torch.sum(batch['target'] ** 2, dim=1).sqrt()
        for index in range(B):
            epe_indexed = epe[index].view(-1)
            mag_indexed = mag[index].view(-1)
            val = batch['valid'][index].view(-1) >= 0.5
            out = ((epe_indexed > 3.0) & ((epe_indexed / mag_indexed) > 0.05)).float()
            epe_list.append(epe_indexed[val].mean().cpu().item())
            out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_sintel(pipeline, args=None, sigma=0.05, start_t=32):
    """ Peform validation using the Sintel (train) split """
    os.makedirs("flow_images_gt", exist_ok=True)
    IMAGE_SIZE = None
    TRAIN_SIZE = [320, 448]
    min_overlap = 304

    pipeline.unet = pipeline.unet.to(torch.bfloat16)

    results = {}
    for dstype in ['final']:
        val_dataset = MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=args.train_batch_size, pin_memory=True, shuffle=False,
                                     num_workers=4)

        epe_list = []
        i=0
        for batch in tqdm(val_loader):
            i+=1
            if i>100:
                break
            for k in batch:
                if type(batch[k]) == torch.Tensor:
                    batch[k] = batch[k].cuda()

            B, _, H, W = batch["image0"].shape
            if IMAGE_SIZE is None or H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
                print(f"replace {IMAGE_SIZE} with [{H}, {W}]")
                IMAGE_SIZE = [H, W]
                hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap=min_overlap)
                weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

            batch["image0"] = 2 * (batch["image0"] / 255.0) - 1.0
            batch["image1"] = 2 * (batch["image1"] / 255.0) - 1.0

            resized_image1 = F.interpolate(batch["image0"], TRAIN_SIZE, mode='bicubic', align_corners=True)
            resized_image2 = F.interpolate(batch["image1"], TRAIN_SIZE, mode='bicubic', align_corners=True)
            inputs = torch.cat([resized_image1, resized_image2], dim=1).repeat(1,1,1,1)
            resized_flow = pipeline(
                inputs=inputs.to(torch.bfloat16),
                batch_size=inputs.shape[0],
                num_inference_steps=args.ddpm_num_steps,
                output_type="tensor",
                normalize=args.normalize_range
            ).images.to(torch.float32)
            # print(np.mean(np.var(resized_flow.cpu().numpy(),axis=0)))

            resized_flow = F.interpolate(resized_flow, IMAGE_SIZE, mode='bicubic', align_corners=True) * \
                           torch.tensor([W / TRAIN_SIZE[1], H / TRAIN_SIZE[0]]).view(1, 2, 1, 1).cuda()

            warpimg1 = backwarp(batch['image1'], resized_flow)

            flows = 0
            flow_count = 0

            image1_tiles = []
            image2_tiles = []
            for idx, (h, w) in enumerate(hws):
                image1_tiles.append(batch["image0"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
                image2_tiles.append(warpimg1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])

            inputs = torch.cat([torch.cat(image1_tiles, dim=0), torch.cat(image2_tiles, dim=0)], dim=1)
            flow_pre_total = pipeline(
                inputs=inputs.to(torch.bfloat16),
                batch_size=inputs.shape[0],
                num_inference_steps=start_t,
                output_type="tensor",
                normalize=args.normalize_range,
            ).images

            for idx, (h, w) in enumerate(hws):
                flow_pre = flow_pre_total[idx * B:(idx + 1) * B]
                padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow = flows / flow_count + resized_flow
            # flow_image = save(flow.cpu().numpy())
            flow_image = save(batch['target'].cpu().numpy())
            flow_image.save(f"flow_images_gt/flow_{i}.png")
            epe = torch.sum((flow - batch['target']) ** 2, dim=1).sqrt()
            epe_list.append(epe.view(-1).cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)
        
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f"{dstype}"] = epe
    return results


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
    for val_dataset in args.validation:
        results = {}
        if val_dataset == 'kitti':
            results.update(validate_kitti(pipeline, args=args))
        elif val_dataset == 'sintel':
            results.update(validate_sintel(pipeline, args=args))
