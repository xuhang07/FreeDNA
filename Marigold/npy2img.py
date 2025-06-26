import numpy as np
from marigold.marigold_pipeline import colorize_depth_maps, chw2hwc
from PIL import Image
import os
npy_path = '/home/xuhang/code/Marigold/output/test/depth_npy'
img_path = '/home/xuhang/code/Marigold/output/test/depth_colored'
os.makedirs(img_path,exist_ok=True)
color_map = 'Spectral'
for i in os.listdir(npy_path)[:5]:
    depth_pred = np.load(os.path.join(npy_path,i))
    # depth_pred = np.array(Image.open(os.path.join(npy_path,i)))
    # depth_pred = np.clip(depth_pred/100,1e-5,80)/80
    # depth_pred = depth_pred/60
    img_name = os.path.splitext(i)[0] + '.png'
    # depth_pred = np.clip(depth_pred,1e-5,60)/60
    depth_colored = colorize_depth_maps(
                    depth_pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_colored_img = Image.fromarray(depth_colored_hwc)
    depth_colored_img.save(os.path.join(img_path,img_name))
    print(f'{img_name} done')