import cv2 as cv

def show(window_name, image, screen_height=1080):
    (h, w) = image.shape[:2]
    r = (screen_height - 40) / float(h)
    dim = (int(w * r), screen_height - 40)

    cv.imshow(window_name, cv.resize(image, dim))


def remove_furigana(furigana, img):
    no_furigana = img.copy()
    for f in furigana:
        draw_rect(no_furigana, f, (255, 255, 255), fill=True)
    return no_furigana


def draw_rect(img, rect, color, thickness=1, fill=False):
    if fill:
        thickness = -1
    cv.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness)


def crop(img, rect, margin=0):
    x, y, w, h = rect
    return img[y - margin:y + h + margin, x - margin:x + w + margin]