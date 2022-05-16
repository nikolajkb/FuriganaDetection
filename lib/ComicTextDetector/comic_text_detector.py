from inference import TextDetector
import cv2
from utils.textmask import REFINEMASK_ANNOTATION

def get_text_cdt(img_path):
    model_path = r'data/comictextdetector.pt'
    img = cv2.imread(img_path)
    model = TextDetector(model_path=model_path, input_size=1024, device='cuda', act='leaky')
    mask, mask_refined, blk_list = model(img, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True)
    cv2.imshow("mask",mask)
    cv2.imshow("mask_refined",mask_refined)
    cv2.waitKey(0)


if __name__ == "__main__":
    get_text_cdt(r"C:\code\comicTextDetector\comic-text-detector\data\examples\DLraw.net-1-008.jpg")
