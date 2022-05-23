import cv2 as cv
import numpy as np
import geometry
from src.image_utils import show


def plot_iou(r1, r2):
    r1 = [int(r1[0]), int(r1[1]), int(r1[2]), int(r1[3])]
    r2 = [int(r2[0]), int(r2[1]), int(r2[2]), int(r2[3])]
    print(r1)
    print(r2)

    img = np.zeros((2100,1300,3), np.uint8)
    img.fill(255)
    cv.rectangle(img, (r1[0],r1[1]), (r1[2],r1[3]),(255,0,0), 2)
    cv.rectangle(img, (r2[0], r2[1]), (r2[2], r2[3]), (0, 0, 255), 2)

    iou = geometry.get_iou(r1, r2)

    print(iou)
    cv.putText(img, "IOU: "+str(iou), (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

    show("rects", img)
    cv.waitKey(0)


def plot_n_iou(r1, r2_lst):
    r1 = [int(r1[0]),int(r1[1]),int(r1[2]),int(r1[3])]
    r2_lst = [[int(r2[0]), int(r2[1]), int(r2[2]), int(r2[3])] for r2 in r2_lst]
    print(r1)
    print(r2_lst)

    img = np.zeros((2100,1300,3), np.uint8)
    img.fill(255)
    cv.rectangle(img, (r1[0],r1[1]), (r1[2],r1[3]),(255,0,0), 2)
    for r2 in r2_lst:
        cv.rectangle(img, (r2[0], r2[1]), (r2[2], r2[3]), (0, 0, 255), 2)

    iou = geometry.get_n_iou(r1,r2_lst)

    print(iou)
    cv.putText(img, "n-IOU: "+str(round(iou,4)), (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

    show("rects", img)
    cv.waitKey(0)

def plot_predictions(ground_truths, predictions):
    img = np.zeros((896,1290,3), np.uint8)
    img.fill(255)
    for r in ground_truths:
        cv.rectangle(img, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]),(0,255,0), 2)

    for r in predictions:
        cv.rectangle(img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255), 2)

    cv.imshow("rects", img)
    cv.waitKey(0)

if __name__ == "__main__":
    #plot_iou([100,100,200,200],[100,100,200,200])
    #plot_n_iou([95, 95, 205, 325], [[100,100,200,200],[100,210,200,320]])
    plot_n_iou([100, 100, 200, 200], [[110,102,150,198],[160,110,200,160]])