import numpy
from tesserocr import PyTessBaseAPI, RIL, iterate_level, PSM
from PIL import Image
import cv2 as cv
from image_utils import draw_rect, show
from geometry import to_wh
import xml.etree.ElementTree as ET
from statistics import mean



img_name = r"C:\Users\nikol\OneDrive\Speciale\data\book\b_1_kino.png"
td = r"C:\code\tessdata-main"
hiragana_katakana = "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑをんっゃゅょぃゔぇ" + \
                    "アイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヰヱヲンッャュョィヴェ"

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def text_is_furigana(img, tessdata):
    im_pil = Image.fromarray(img)
    width, height = im_pil.size
    horizontal = True if width > height else False
    square = True if 0.5 < height / width < 1.5 else False
    im_pil = im_pil.resize((width*2,height*2))
    im_pil = add_margin(im_pil,10,10,10,10,(255,255,255))
    lang = "jpn" if horizontal else "jpn_vert"
    seg = PSM.SINGLE_LINE if horizontal else PSM.SINGLE_BLOCK_VERT_TEXT
    if square:
        lang = "jpn"
        seg = PSM.SINGLE_CHAR

    with PyTessBaseAPI(tessdata, lang=lang) as api:
        api.SetVariable("tessedit_char_whitelist", hiragana_katakana)
        api.SetPageSegMode(seg)
        api.SetImage(im_pil)
        text = api.GetUTF8Text()
        confidences = api.AllWordConfidences()
        line_is_furigana = is_furigana(text,confidences)

    if False:
        print(lang)
        print("square: ",square)
        print(text)
        print(confidences)
        print("Is furigana: ",line_is_furigana)
        im_cv = numpy.array(im_pil)
        cv.imshow("original",img)
        cv.imshow("cropped",im_cv)
        print("---------")
        cv.waitKey(0)
        cv.destroyWindow("original")
        cv.destroyWindow("cropped")

    return line_is_furigana

def is_furigana(text, confidences):
    if len(confidences) == 0:
        return False
    elif mean(confidences) > 50:
        return True
    elif any([c >= 90 for c in confidences]):
        return True
    else:
        return False


def text_direction(img, tessdata):
    try:
        im_pil = Image.fromarray(img)
    except:
        return False  # TODO this error happens when the crop touches the edges of the image
    im_pil = add_margin(im_pil, 20, 20, 20, 20, (255, 255, 255))
    with PyTessBaseAPI(path=tessdata, lang="jpn", oem=0) as api:
        api.SetImage(im_pil)
        api.Recognize()

        ri = api.GetIterator()
        level = RIL.TEXTLINE
        nr_hor = 0
        nr_ver = 0
        for r in iterate_level(ri, level):
            if r.BoundingBox(level) is None:
                continue
            wh = to_wh(r.BoundingBox(level))
            if wh[2] > wh[3]:
                nr_hor += 1
            else:
                nr_ver += 1
        if nr_hor > nr_ver:
            return True
        else:
            return False


def parse_hocr(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    body = root.find("{http://www.w3.org/1999/xhtml}body")
    for page in body:
        img_path = page.get("title").split(";")[0][7:-1]
        img = cv.imread(img_path)
        for block in page:
            for paragraph in block:
                for line in paragraph:
                    bbox = [int(c) for c in line.get("title").split(";")[0].split(" ")[1:]]
                    draw_rect(img, to_wh(bbox), (0, 0, 255), 3)
                    for word in line:
                        break
                        bbox = [int(c) for c in word.get("title").split(";")[0].split(" ")[1:]]
                        print(word.text)
                        print(word.get("title").split(";")[1])
                        print("--------------------")
                        draw_rect(img,to_wh(bbox),(0,0,255))
                        show("box",img)
                        cv.waitKey(0)
                        draw_rect(img,to_wh(bbox),(0,0,255))

    show("imag", img)
    cv.waitKey(0)



