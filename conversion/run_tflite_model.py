import argparse
import cv2
import numpy as np
import tensorflow as tf

from utils import vis
from postprocess import get_detections

def get_parser():
    parser = argparse.ArgumentParser(description='Run TF Model')
    parser.add_argument('--tflite_model', default='model', type=str, required=True)
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
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data
    # input_shape = input_details[0]['shape']

    # read image
    img = cv2.imread(args.img)
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)  # expand dim
    input_data = img

    # set input image
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor

    # print(output_details)
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)

    output = interpreter.get_tensor(output_details[0]['index'])
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    confs = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    out = {}
    out['boxes'] = boxes
    out['confs'] = confs

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

    # cv2.imwrite('./data/test5_result.png', result)

    # print_tflite_model_tensor_types(interpreter)

def print_tflite_model_tensor_types(interpreter):
    # get tensor details
    tensor_details = interpreter.get_tensor_details()

    # print all tensors
    for tensor in tensor_details:
        name = tensor['name']
        dtype = tensor['dtype']
        print(f'Tensor Name: {name}, Data Type: {dtype}')

if __name__=='__main__':
    args = get_parser()
    main(args)