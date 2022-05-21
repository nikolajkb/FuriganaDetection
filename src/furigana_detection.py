from typing import Dict, List

import detection
import os
import json
import argparse
from evaluate import evaluate

os.environ['NUMEXPR_MAX_THREADS'] = '16'


def detect_folder(folder_path, labels=None, out="predictions.json", debug=False, debug_areas=False):
    file_names = []
    with os.scandir(folder_path) as it:
        for file in it:
            if file.name.endswith(".jpg") or file.name.endswith(".png") or file.name.endswith(".JPEG"):
                file_names.append(file.name)

    label_dict: Dict[str, List[Dict[str, str]]] = {}
    if labels:
        with open(labels,"r") as label_file:
            label_dict = json.load(label_file)

    predictions = []
    for i,file in enumerate(file_names):
        file_path = os.path.join(folder_path, file)
        detections = detection.FuriganaDetector(verbose=debug, debug_areas=debug_areas).detect(file_path)

        img_id = i
        if label_dict:
            img_id = int(list(filter(lambda entry: entry.get("file_name") == file, label_dict.get("images")))[0].get("id"))

        detection_to_json(detections,predictions,img_id)

    with open(os.path.join(folder_path,out), 'w') as file:
        json.dump(predictions, file)


def detection_to_json(detections, predictions, img_id):
    for d in detections:
        predictions.append({
            'image_id': img_id,
            'category_id': 1,  # furigana
            'bbox': d,
            'score': 1  # no confidence is given so score is set to 1
        })
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Show debug images (result)", action="store_true")
    parser.add_argument("--debug_area", help="Show debug images for each individual text area", action="store_true")
    parser.add_argument("--folder", help="Detect all images in a folder", type=str, default=None)
    parser.add_argument("--image", help="An image to detect furigana in", type=str, default=None)
    parser.add_argument("--config", help="Path to a config file with advanced configurations", type=str, default="config.json")
    parser.add_argument("--out", help="name of output file", default="predictions.json")
    parser.add_argument("--labels", help="Path to ground truth labels", default=None)
    parser.add_argument("--predictions", help="Path to predictions for evaluation (labels must also be specified)", default=None)
    parser.add_argument("--validate", help="validate detections using ocr (tessdata must be specified)", action="store_true")
    parser.add_argument("--eval", help="run evaluation. If a folder is detected, these results will be evaluated", action="store_true")

    args = parser.parse_args()
    cwd = os.getcwd()

    if args.validate and not args.tessdata:
        print("you must specify tessdata location in order to validate detections with OCR")
        exit()

    if args.eval and not args.labels:
        print("labels must be specified in order to evaluate")
        exit()

    if args.image:
        furigana = detection.FuriganaDetector(verbose=args.debug, debug_areas=args.debug_area, tessdata=args.tessdata,
                                              validate=args.validate, config=args.config).detect(args.image)
        with open(os.path.join(cwd, args.out), 'w') as file:
            json.dump(detection_to_json(furigana, [], 0), file)
    elif args.folder:
        detect_folder(args.folder, labels=args.labels, out=args.out)
        if args.eval and args.labels:
            evaluate(args.labels, os.path.join(args.folder, args.out))
    elif args.predictions and args.labels and args.eval:
        evaluate(args.labels, args.predictions)
    else:
        print("No image, folder or predictions specified")

