import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from glob import glob
from timm.data import constants
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from tqdm import tqdm
from clearml import Logger

TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
    ]
)


class VisualizeCallback(Callback):
    def __init__(self, args):
        args.visualize_output = os.path.join(args.visualize_output, args.desc)
        os.makedirs(args.visualize_output, exist_ok=True)
        self.args = args
        image_folder = os.path.join(args.visualize_input, "TrueOrtho")
        with open(os.path.join(args.visualize_input, "ImageSet", "test.txt"), "r") as f:
            image_names = f.readlines()
        self.img_paths = [os.path.join(image_folder, x.strip()) for x in image_names]
        self.out_path = args.visualize_output
        self.size = args.img_size[0]

    def on_train_epoch_end(self, trainer, pl_module):
        for path in tqdm(self.img_paths):
            img_name = path.split("/")[-1]
            out_path = os.path.join(self.out_path, img_name)
            big_img = cv2.imread(path)
            total_scence = np.zeros_like(big_img)
            big_mask = big_img > 0
            for x in range(0, big_img.shape[1] - self.size, self.size):
                for y in range(0, big_img.shape[0] - self.size, self.size):
                    img = big_img[y : y + self.size, x : x + self.size, :]
                    tensor_img = TRANSFORM(img).unsqueeze(0)
                    with torch.no_grad():
                        mask_pred = pl_module(tensor_img.cuda())
                    mask_pred = F.interpolate(mask_pred, scale_factor=4, mode="nearest")
                    mask_pred = mask_pred.squeeze(0).squeeze(0)
                    mask_pred = mask_pred > 0
                    mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
                    color = np.array([0, 255, 0], dtype="uint8")
                    masked_img = np.where(mask_pred[..., None], color, img)
                    colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
                    total_scence[
                        y : y + self.size, x : x + self.size, :
                    ] = colored_mask_img
            total_scence = total_scence * big_mask
            total_scence = cv2.resize(total_scence, (1444, 1444))
            total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
            Logger.current_logger().report_image(
                "Testing Image",
                img_name,
                iteration=trainer.current_epoch,
                image=total_scence,
            )
