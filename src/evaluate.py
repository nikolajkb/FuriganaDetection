import json
from statistics import mean
import plots
from geometry import get_n_iou_wh, get_ioa_wh

def evaluate(label_path: str, prediction_path: str, iou_threshold=0.5, ioa_threshold=0.6):
    with open(label_path, 'r') as label_file, open(prediction_path, 'r') as prediction_file:
        label_json = json.load(label_file)
        prediction_json = json.load(prediction_file)

        image_count = len(label_json.get('images'))

        predictions_by_image = get_annotation_by_image(prediction_json, image_count)
        labels_by_image = get_annotation_by_image(label_json.get('annotations'), image_count)

        precisions = []
        recalls = []
        f1s = []

        for i in range(image_count):
            im_name = get_image_name(i,label_json)

            #if not im_name.startswith("t_b"): continue                                                      # book
            #if (not im_name.startswith("t_b")) or ("tobira" in im_name) or ("jpfor" in im_name): continue    # books (no textbook)
            #if ("tobira" not in im_name) and ("jpfor" not in im_name): continue                               # only textbooks
            #if not im_name.startswith("t_c"): continue                                                       # comics


            print("Filename:",im_name)

            predictions = predictions_by_image[i]
            labels = labels_by_image[i]
            used_labels = []
            true_positive = 0
            false_positive = 0
            if len(labels) == 0:
                print("No labels")
                print()
                continue
            else:
                for prediction in predictions:
                    ioas = [[get_ioa_wh(label,prediction),i] for i,label in enumerate(labels)]
                    ioa_l = list(filter(lambda ioa: ioa[0] > ioa_threshold and ioa[1] not in used_labels, ioas))
                    ioa_rects = [labels[i[1]] for i in ioa_l]
                    if len(ioa_l) == 1:
                        #plots.plot_iou(to_bbox(prediction), to_bbox(ioa_rects[0]))
                        iou = get_n_iou_wh(prediction, [ioa_rects[0]])
                        if iou > iou_threshold:
                            true_positive += 1
                            used_labels.append(ioa_l[0][1])
                    elif len(ioa_l) >= 2:
                        #plots.plot_n_iou(to_bbox(prediction), geometry.to_bbox_n(ioa_rects))
                        niou = get_n_iou_wh(prediction,ioa_rects)
                        if niou > iou_threshold:
                            true_positive += len(ioa_rects)
                            used_labels.extend([i[1] for i in ioa_l])
                    else:
                        false_positive += 1

                if true_positive + false_positive == 0:
                    continue

                precision = true_positive / (true_positive + false_positive)
                precisions.append(precision)

                recall = true_positive / len(labels)
                recalls.append(recall)

                f1 = 0
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
            f1s.append(f1)
            print("TP:",true_positive)
            print("FP:",false_positive)
            print("Recall", recall)
            print("Precision",precision)
            print("F1-score", f1)
            print()

        print("Averages")
        print("Recall",mean(recalls))
        print("Precision:",mean(precisions))
        print("F1-score", mean(f1s))


def get_image_name(index,label_json):
    return next(filter(lambda img: int(img.get('id')) == index, label_json.get('images'))).get('file_name')


def get_annotation_by_image(annotations, image_count):
    by_image = [[] for _ in range(image_count)]
    for annotation in annotations:
        by_image[int(annotation.get('image_id'))].append(annotation.get('bbox'))

    return by_image
