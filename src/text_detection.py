import sys
sys.path.append(r'..\lib\MangaTextSegmentation\MangaCode')
sys.path.append(r'..\lib\MangaTextSegmentation')
sys.path.append(r'..\lib\ComicTextDetector')

from fastai.vision import load_learner
from lib.MangaTextSegmentation.MangaCode.dataset import open_image, unpad_tensor, image2np
import cv2 as cv
import numpy as np
import torch
from lib.MangaImageTranslator.textblockdetector import dispatch as dispatch_ctd_detection
from lib.MangaImageTranslator.textblockdetector.textblock import visualize_textblocks
import asyncio
from image_utils import show
import os
from lib.ComicTextDetector.inference import TextDetector
from lib.ComicTextDetector.utils.textmask import REFINEMASK_ANNOTATION
model_path_onnex = r'../lib/ComicTextDetector/data/comictextdetector.pt.onnx'
model_path = r'../lib/ComicTextDetector/data/comictextdetector.pt'

class TextDetectionSimple:
    def get_text(self, img_path):
        img = cv.imread(img_path)
        img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(img_g, 180, 255, cv.THRESH_BINARY_INV)
        return threshold


class TextDetectionCTD:
    def __init__(self):
        self.model = TextDetector(model_path=model_path_onnex, input_size=1024, device='cuda', act='leaky')

    def get_text(self, img_path):
        img = cv.imread(img_path)
        mask, mask_refined, blk_list = self.model(img, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True)
        #cv.imshow("mask_refined", mask_refined)
        return mask_refined


class TextDetectionMIT:
    def get_text(self, img_path):
        img = cv.imread(img_path)
        loop = asyncio.get_event_loop()
        mask, raw_mask = loop.run_until_complete(self._get_text(img))
        return mask

    async def _get_text(self, img):
        mask, final_mask, textlines = await dispatch_ctd_detection(img, False, model_path, model_path_onnex)
        #visualize_textblocks(img,textlines)
        return final_mask, mask


class TextDetectionMTS:
    def __init__(self):
        self.learner = load_learner('../models','model_mts.pkl')

    def get_text(self, img_path):

        # make text location prediction using Manga Text Segmentation
        img_raw = open_image(img_path)
        make_image_even(img_raw) # fastai will throw an error when given images where width or height is an odd number
        torch.cuda.empty_cache()
        pred = self.learner.predict(img_raw)[0]

        # create image from prediction
        mask = unpad_tensor(pred.px, img_raw.px.shape)[0]
        img_tensor = torch.ones(img_raw.px.shape) * 255
        img_tensor[1][mask == 1] = 0
        img_tensor[2][mask == 1] = 0
        img = image2np(img_tensor).astype(np.uint8)

        # convert to binary image
        img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_g = cv.bitwise_not(img_g) # invert colors

        return img_g


def make_image_even(img):
    width,height = img.size
    if(height % 2) == 1 and (width % 2 == 1):
        img.crop_pad((width + 1, height + 1), padding_mode="zeros")
    else:
        if(height % 2) == 1:
            img.crop_pad((width,height + 1), padding_mode="zeros")

        if(width % 2) == 1:
            img.crop_pad((width + 1,height), padding_mode="zeros")


def test_folder():
    td = TextDetectionMIT()
    folder = r"C:\code\FuriganaDetection\data\all\data"
    with os.scandir(folder) as it:
        for file in it:
            if file.name.endswith(".jpg"):
                img_path = os.path.join(folder, file.name)
                text = td.get_text(img_path)
                show("text", text)
                cv.waitKey(0)


if __name__ == "__main__":
    td = TextDetectionMIT()
    text = td.get_text(r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\test\data\t_c_18_misu_jpg.rf.48ceb7769b4c2658d8e679f1cc00b868.jpg")
    text2 = TextDetectionCTD().get_text(r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\test\data\t_c_18_misu_jpg.rf.48ceb7769b4c2658d8e679f1cc00b868.jpg")
    cv.imshow("mit",text)
    cv.imshow("ctd",text2)
    cv.waitKey(0)
