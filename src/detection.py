import json
import math
import cv2 as cv
import numpy as np
import os
from image_utils import show, draw_rect
from src import geometry
from text_detection import TextDetectionMTS, TextDetectionMIT
from statistics import mean
from geometry import rect_distance, bounding_rect
import ocr
os.environ['NUMEXPR_MAX_THREADS'] = '16'

left = 0
top = 1
width = 2
height = 3


class FuriganaDetector:
    def __init__(self, screen_height=1080, verbose=False, debug_areas=False, tessdata=None, validate=False, config="config.json", text_detector=TextDetectionMIT()):
        self.binary = None
        self.img_w = None
        self.img_h = None
        self.img = None
        self.screen_height = screen_height
        self.text_detector = text_detector
        self.verbose = verbose
        self.debug_areas = debug_areas
        self.tessdata = tessdata
        self.validate = validate
        with open(config) as config_file:
            self.config = json.load(config_file)

    def c(self, name):
        return self.config.get(name)

    def detect(self, file_path):
        print("processing file: " + file_path)

        img = cv.imread(file_path)
        self.img = img
        self.img_w = img.shape[1]
        self.img_h = img.shape[0]

        self.binary = self.text_detector.get_text(file_path)

        text_areas, area_horizontal = self.find_text_areas(self.binary, verbose=self.verbose)

        furigana_rects = []

        morph_all = np.zeros(self.binary.shape, np.uint8)
        th_color = cv.cvtColor(self.binary,cv.COLOR_GRAY2BGR)
        watershed = False
        self.debug_areas = False
        for i, area in enumerate(text_areas):
            img_area = self.remove_other_text(area, self.binary)

            horizontal = area_horizontal[i]

            if watershed:
                kernel = np.ones((1, self.rel(15)), np.uint8) if horizontal else np.ones((self.rel(15), 1), np.uint8)
                morph = cv.morphologyEx(img_area, cv.MORPH_CLOSE, kernel, iterations=3)

                rects = self.watershed(morph, th_color)
            else:
                morph = self.morph_lines(img_area, horizontal)

                contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                rects = bounding_rects(contours)
            rects = grow_rects(rects, 2, 2, 0, 0)  # since we are eroding the text slightly, we readjust here.

            furigana_rects_area = filter_furigana(rects, horizontal)

            if self.debug_areas:
                show("Only this area", img_area)
                show("After morphology", morph, self.screen_height)
                img_rects = self.draw_rects(img, rects)
                show("bounding boxes", img_rects, self.screen_height)
                img_furigana_rects = self.draw_rects(img, furigana_rects_area)
                show("bounding boxes furigana", img_furigana_rects, self.screen_height)
                cv.waitKey(0)

            morph_all = cv.bitwise_or(morph_all, morph)
            furigana_rects += furigana_rects_area

        merge_close_rects(furigana_rects)
        furigana_rects = self.split_furigana(furigana_rects)

        furigana_rects = grow_rects(furigana_rects, 2, 2, 0, 0)
        if self.validate:
            furigana_rects = self.validate_detection(furigana_rects)

        if self.verbose:
            img_furigana_rects = self.draw_rects(img, furigana_rects)
            show("bounding boxes furigana merged", img_furigana_rects, self.screen_height)
            show("morph all", morph_all)
            show("only text", self.binary, self.screen_height)

            show(file_path, img, self.screen_height)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return furigana_rects

    def validate_detection(self, rects):
        return [r for r in rects if ocr.text_is_furigana(self.crop(self.img, r), self.tessdata)]

    def morph_lines(self, threshold, horizontal):
        kernel = np.ones((2, 1), np.uint8) if horizontal else np.ones((1, 2), np.uint8)
        morph = cv.morphologyEx(threshold, cv.MORPH_ERODE, kernel)

        kernel = np.ones((1, self.rel(self.c("lineKernel"))), np.uint8) if horizontal else np.ones((self.rel(self.c("lineKernel")), 1), np.uint8)
        morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel, iterations=2)
        return morph

    def split_furigana(self, furigana):
        split = []
        for rect in furigana:
            only_this = self.remove_other_text(rect, self.binary)
            kernel = np.ones((self.rel(self.c("furiganaSplitKernel")), self.rel(self.c("furiganaSplitKernel"))), np.uint8)
            morph = cv.morphologyEx(only_this, cv.MORPH_CLOSE, kernel)

            contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            rects = bounding_rects(contours)

            split += rects

        return split

    def remove_other_text(self, area, binary):
        mask = np.zeros(binary.shape, dtype="uint8")
        draw_rect(mask, area, (255,255,255), fill=True)

        area_only = cv.bitwise_and(mask, binary)

        return area_only

    def watershed(self, threshold, th_color):
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(threshold, kernel, iterations=6)

        dist_transform = cv.distanceTransform(threshold, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv.subtract(sure_bg, sure_fg)

        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv.watershed(cv.cvtColor(threshold, cv.COLOR_GRAY2BGR), markers)

        markers[markers == -1] = 0
        unique = np.unique(markers)
        unique = np.delete(unique, [0, 1])

        for i in unique:
            th_color[markers == i] = list(np.random.choice(range(256), size=3))

        rects_all = []
        for i in unique:
            mask = np.zeros(threshold.shape, dtype="uint8")
            mask[markers == i] = 255
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            rects = bounding_rects(contours)
            rects_all.append(rects[0])

        to_remove = []
        for r1 in rects_all:
            for r2 in rects_all:
                if r1 == r2:
                    continue
                ioa = geometry.get_ioa_wh(r1, r2)
                if ioa > 0.9:
                    to_remove.append(r1)

        rects_all = [r for r in rects_all if r not in to_remove]

        return rects_all

    # defines a general way of getting a size relative to the size of the image
    def rel(self, size):
        return round(((self.img_h + self.img_w) / 1000) * size)

    def rel_q(self, size):
        return round(((self.img_h * self.img_w) / 100000) * size)

    def infer_text_direction(self, rects):
        horizontal = []
        vertical = []
        for r in rects:
            max_ratio = max(r[width] / r[height], r[height] / r[width])
            if True: #max_ratio > 2.5 or (r[width] * r[height]) < self.rel_q(40) or not self.tessdata:
                ratio = r[width] / r[height]
                if ratio > 1:
                    horizontal.append(r)
                else:
                    vertical.append(r)
            else:
                is_horizontal = ocr.text_direction(self.crop(self.img, r, 2), tessdata=self.tessdata)
                if is_horizontal:
                    horizontal.append(r)
                else:
                    vertical.append(r)

        return horizontal, vertical

    def find_text_areas(self, threshold, verbose=True):
        kernel = np.ones((self.rel(self.c("textAreaKernel")), self.rel(self.c("textAreaKernel"))), np.uint8)
        morph = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        rects = bounding_rects(contours)

        horizontal, vertical = self.infer_text_direction(rects)

        vertical = grow_rects(vertical, 5, 5, 5, 5)
        horizontal = grow_rects(horizontal, 5, 5, 5, 5)
        merge_close_rects(vertical, max_dist=self.rel(self.c("textAreaMergeSize")))
        merge_close_rects(horizontal, max_dist=self.rel(self.c("textAreaMergeSize")))

        areas = horizontal + vertical
        area_horizontal = [(True if r in horizontal else False) for r in areas]
        merge_close_rects(areas, max_dist=0, groups=area_horizontal)  # sometimes small blobs appear inside larger text areas, we merge these into the larger area

        if verbose:
            img = cv.cvtColor(threshold, cv.COLOR_GRAY2BGR)
            img_final = self.draw_rects(img, areas)
            show("Morph text areas", morph)
            show("bounding boxes", img_final)

        return areas, area_horizontal

    def draw_rects(self, img, rects, thickness=0):
        if thickness == 0:
            thickness = (1 if self.img_w < 1000 else 2)

        img_rects = img.copy()
        for rect in rects:
            cv.rectangle(img_rects, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), thickness)

        return img_rects

    def crop(self, img, rect, margin=0):
        x, y, w, h = rect
        return img[y - margin:y + h + margin, x - margin:x + w + margin]



def filter_wide(rect):
    return (rect[width] / rect[height]) > 1


def merge_close_rects(rects, max_dist=5, groups=None):
    did_merge = False
    for i1, r1 in enumerate(rects):
        if did_merge:
            break
        for i2, r2 in enumerate(rects):
            if i1 != i2 and rect_distance(r1, r2) <= max_dist:
                merge = bounding_rect(r1, r2)
                rects.append(merge)
                rects.remove(r1)
                rects.remove(r2)

                if groups:
                    i1_orig = groups[i1]
                    i2_orig = groups[i2]
                    del groups[i2]
                    del groups[i1]
                    if r1[2]*r1[3] > r2[2]*r2[3]:
                        groups.append(i1_orig)
                    else:
                        groups.append(i2_orig)

                did_merge = True
                break

    if did_merge:
        merge_close_rects(rects, max_dist=max_dist, groups=groups)


def grow_rects(rects, l, t, r, b):
    new_rects = []
    for rect in rects:
        rect_l = list(rect)
        rect_l[0] = rect_l[0] - l
        rect_l[1] = rect_l[1] - t
        rect_l[2] = rect_l[2] + l + r
        rect_l[3] = rect_l[3] + t + b
        if rect_l[0] < 0: rect_l[0] = 0
        if rect_l[1] < 0: rect_l[1] = 0
        new_rects.append(rect_l)
    return new_rects


def calc_font_size(rects, bin_size, horizontal):
    line_thickness = [rect[3] for rect in rects] if horizontal else [rect[2] for rect in rects]

    min_thickness = min(line_thickness)
    max_thickness = max(line_thickness)
    if min_thickness == max_thickness:
        return min_thickness  # all the thicknesses are the same, simply return one of them

    font_size = 0
    max_area = 0
    for i in range(min_thickness, max_thickness):
        area = sum([r[2] * r[3] for r in rects if i <= (r[3] if horizontal else r[2]) <= (i + bin_size)])
        if area >= max_area:
            max_area = area
            font_size = mean([(r[3] if horizontal else r[2]) for r in rects if
                              i <= (r[3] if horizontal else r[2]) <= (i + bin_size)])

    return font_size


# attempts to filter away rectangles that are not furigana
def filter_furigana(rects, horizontal):
    if len(rects) <= 1:
        return []
    font_size = calc_font_size(rects, 5, horizontal)
    furigana_size = font_size/2
    furigana_min = 3
    furigana_max = math.ceil(furigana_size + 2)

    filtered = filter(lambda r: furigana_min < (r[3] if horizontal else r[2]) < furigana_max, rects)

    return list(filtered)


def bounding_rects(contours):
    rects = []
    for c in contours:
        rects.append(cv.boundingRect(c))
    return rects


def test_detect_folder(path):
    with os.scandir(path) as it:
        for file in it:
            if (file.name.endswith(".jpg") or file.name.endswith(".JPEG")) and ("jpfor" in file.name or "tobira" in file.name):
                img_path = os.path.join(path, file.name)
                FuriganaDetector(verbose=True).detect(img_path)


if __name__ == "__main__":
    FuriganaDetector(verbose=True).detect(r"../data/example.jpg")





