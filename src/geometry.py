from math import sqrt


def dist(x1,y1,x2,y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# https://stackoverflow.com/a/26178015
def rect_distance(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    x1b = x1 + w1
    y1b = y1 + h1

    x2b = x2 + w2
    y2b = y2 + h2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0


def bounding_rect(r1,r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    x3 = min(x1,x2)
    y3 = min(y1,y2)
    w3 = max(x1 + w1, x2 + w2) - x3
    h3 = max(y1 + h1, y2 + h2) - y3

    return x3,y3,w3,h3


def to_wh(rect):
    return [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]


def to_bbox(rect):
    return [rect[0],rect[1],rect[0] + rect[2], rect[1] + rect[3]]


def to_bbox_n(rects):
    return [[rect[0],rect[1],rect[0] + rect[2], rect[1] + rect[3]] for rect in rects]


# gets the iou between a and b_lst which is a list of rects
def get_n_iou(a, b_lst, epsilon=1e-5):
    total_overlap = 0
    total_b_area = 0
    for b in b_lst:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        width = (x2 - x1)
        height = (y2 - y1)

        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height

        area_b = (b[2] - b[0]) * (b[3] - b[1])
        total_b_area += area_b

        total_overlap += area_overlap

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_combined = total_b_area + area_a - total_overlap

    iou = total_overlap/ (area_combined+epsilon)

    return iou


def get_n_iou_wh(a, b_lst):
    return get_n_iou(to_bbox(a), to_bbox_n(b_lst))


# source: http://ronny.rest/tutorials/module/localization_001/iou/
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def get_iou_wh(a, b):
    return get_iou(to_bbox(a), to_bbox(b))


# intersection over area
def get_ioa(a,b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    area_a = (a[0] - a[2]) * (a[1] - a[3])

    return area_overlap / area_a


def get_ioa_wh(small,big):
    return get_ioa(to_bbox(small), to_bbox(big))


if __name__ == "__main__":
    print(bounding_rect((0,0,100,100),(50,50,150,150)))