import nltk
import os
from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image
import cv2 as cv
from image_utils import draw_rect, show, crop, remove_furigana
from geometry import to_wh
import xml.etree.ElementTree as ET
from detection import FuriganaDetector
from statistics import mean

td = r"C:\code\tessdata-main"

def get_text(img, clean=False):
    im_pil = Image.fromarray(img)
    with PyTessBaseAPI(path=td, lang="jpn_vert") as api:
        api.SetPageSegMode(PSM.SINGLE_BLOCK_VERT_TEXT)
        api.SetImage(im_pil)
        text = api.GetUTF8Text()
        if clean:
            return text.replace(" ","").replace("\n","")
        else:
            return text


def evaluate_manga109_ocr(parent_folder_path, annotation_folder):
    detector = FuriganaDetector()
    with os.scandir(parent_folder_path) as books_it, os.scandir(annotation_folder) as annotation_it:
        for book in books_it:
            annotation = next(annotation_it)
            book = book.name
            annotation = annotation.name
            print(book)
            print(annotation)
            if book != "AkkeraKanjinchou":
                continue

            annotation_path = os.path.join(annotation_folder, annotation)
            folder_path = os.path.join(parent_folder_path, book)

            evaluate_book(folder_path, annotation_path, detector)


def evaluate_book(folder_path, annotation_path, detector, gt_path=None):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    pages = root.find("pages")
    err_rate_wf_all = []
    err_rate_nf_all = []
    err_rate_gt_all = []
    gt_it = None
    img_gt = None
    if gt_path:
        gt_it = os.scandir(gt_path)
    with os.scandir(folder_path) as it:
        for page in pages:
            img = next(it, None)
            if img is None:
                break
            img_path = img.name
            full_path = os.path.join(folder_path, img_path)

            img = cv.imread(full_path)
            furigana = detector.detect(full_path)
            img_nf = remove_furigana(furigana, img)

            if gt_it is not None:
                img_gt = next(gt_it, None)
                full_path_gt = os.path.join(gt_path,img_gt)
                img_gt = cv.imread(full_path_gt)
            for obj in page:
                if obj.text:
                    xmin = obj.get("xmin")
                    ymin = obj.get("ymin")
                    xmax = obj.get("xmax")
                    ymax = obj.get("ymax")
                    location = to_wh([int(xmin), int(ymin), int(xmax), int(ymax)])
                    (err_rate_wf, err_rate_nf, err_rate_gt) = evaluate_area(obj.text, img, location, img_nf, img_gt=img_gt)
                    err_rate_wf_all.append(err_rate_wf)
                    err_rate_nf_all.append(err_rate_nf)
                    err_rate_gt_all.append(err_rate_gt)

    if gt_it:
        gt_it.close()
    print("")
    print(folder_path.split("\\")[-1])
    print("err rate wf:\n", mean(err_rate_wf_all), sep="")
    print("err rate nf:\n", mean(err_rate_nf_all), sep="")
    if gt_it:
        print("err rate gt:\n", mean(err_rate_gt_all), sep="")
    print("total text areas:\n", len(err_rate_wf_all), sep="")
    print("")


def evaluate_area(text, img, location, img_nf, img_gt=None):
    im_crop_wf = crop(img, location, 5)
    im_crop_nf = crop(img_nf, location, 5)
    try:
        text_wf = get_text(im_crop_wf, clean=True)
    except:
        im_crop_wf = crop(img, location, 0)
        im_crop_nf = crop(img_nf, location, 0)
        text_wf = get_text(im_crop_wf, clean=True)
    dist_wf = nltk.edit_distance(text, text_wf)  # Levenshtein edit-distance
    text_nf = get_text(im_crop_nf, clean=True)
    dist_nf = nltk.edit_distance(text, text_nf)

    dist_gt = None
    im_crop_gt = None
    text_gt = None
    if img_gt is not None:
        im_crop_gt = crop(img_gt, location, 5)
        text_gt = get_text(im_crop_gt, clean=True)
        dist_gt = nltk.edit_distance(text, text_gt)

    err_rate_wf = dist_wf / len(text)
    err_rate_nf = dist_nf / len(text)
    err_rate_gt = None
    if dist_gt is not None:
        err_rate_gt = dist_gt / len(text)

    if False:
        print("GT:      ", text)
        print("WF:      ", text_wf)
        print("NF:      ", text_nf)
        print("NFGT:    ", text_gt)
        print("GT/WF:   ", dist_wf)
        print("GT/NF:   ", dist_nf)
        print("GT/GTNF: ", dist_gt)
        print("CER_WF:  ", err_rate_wf)
        print("CER_NF:  ", err_rate_nf)
        print("CER_GT:  ", err_rate_gt)

        cv.imshow("wf", im_crop_wf)
        cv.imshow("nf", im_crop_nf)
        if im_crop_gt is not None:
            cv.imshow("gtnf", im_crop_gt)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("")

    return err_rate_wf, err_rate_nf, err_rate_gt


if __name__ == "__main__":
    #parse_hocr(r"C:\Users\nikol\AppData\Local\Programs\Tesseract-OCR\hocr.hocr")
    #evaluate_manga109_ocr(r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\Manga109_released_2021_12_30\images",
        # r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\Manga109_released_2021_12_30\annotations.v2020.12.18")
    evaluate_book(r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\Manga109_released_2021_12_30\images\AkkeraKanjinchou",
                  r"C:\Users\nikol\PycharmProjects\FuriganaDetection\data\Manga109_released_2021_12_30\annotations\AkkeraKanjinchou.xml", FuriganaDetector(),
                  gt_path=r"C:\Users\nikol\PycharmProjects\FuriganaDetection2\data\Akkera Kanjinchou - no furigana")