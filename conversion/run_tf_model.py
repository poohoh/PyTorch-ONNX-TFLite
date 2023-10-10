import argparse

import tensorflow as tf
import cv2
import numpy as np

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
from utils import vis
from postprocess import get_detections

def get_parser():
    parser = argparse.ArgumentParser(description='Run TF Model')
    parser.add_argument('--tf_model', default='model', type=str, required=True)
    parser.add_argument('--names', default='names.names', type=str)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--score', default=0.3, type=float)
    parser.add_argument('--nms', default=0.45, type=float)
    parser.add_argument('--out', default='out/result.png', type=str)

    return parser.parse_args()

def read_names(names_path):
    if names_path == "":
        return None

    class_names = []
    with open(names_path, "r") as f:
        for line in f:
            class_names.append(line.strip())
    return class_names

def main(args):
    model = tf.saved_model.load(args.tf_model)
    model.trainable = False

    # read image
    img = cv2.imread(args.img)
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)  # expand dim
    input_data = img


    # input_tensor = tf.random.uniform([1, 3, 416, 416])

    # out = model(**{'input': input_tensor})
    out = model(**{'input': input_data})
    # boxes = np.squeeze(out['boxes'])
    # confs = np.squeeze(out['confs'])

    final_boxes, final_scores, final_cls_inds = get_detections(out, args.score, args.nms)

    class_names = read_names(args.names)
    result = vis(
        args.img,
        final_boxes,
        final_scores,
        final_cls_inds,
        conf=0.5,
        class_names=class_names,
        out_img=args.out,
        print_bbox=True,
    )

    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
    args = get_parser()
    main(args)