import json
import os.path
import sys

import fiftyone as fo
import fiftyone.utils.coco as fouc

def coco_evaluation():
    dataset = fo.Dataset.from_dir(
        dataset_dir="../data/furigana",
        dataset_type=fo.types.COCODetectionDataset,
    )

    fouc.add_coco_labels(
        dataset,
        "predictions",
        "../data/furigana/data/predictions.json",
    )

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
    )

    results.print_report(classes=["furigana"])

    result_map = results.metrics(classes=["furigana"])
    print(result_map)

    return result_map

def start_session(dataset_path):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_path,
        dataset_type=fo.types.COCODetectionDataset,
    )

    print(dataset)

    prediction_path = os.path.join(dataset_path,"data","predictions.json")
    if not os.path.isfile(prediction_path):
        print("predictions.json missing in " + prediction_path)

    fix_prediction_index(os.path.join(dataset_path,"data"),'predictions.json')

    fouc.add_coco_labels(
        dataset,
        "predictions",
        os.path.join(dataset_path,"data","fo_predictions.json"),
    )

    session = fo.launch_app(dataset)
    session.wait()


def fix_prediction_index(path,filename):
    with open(os.path.join(path,filename),"r") as old, open(os.path.join(path,"fo_predictions.json"), 'w') as new:
        old_json = json.load(old)
        for detection in old_json:
            detection['image_id'] = detection.get('image_id') + 1
        json.dump(old_json,new)


if __name__ == "__main__":
    start_session("../data/test")