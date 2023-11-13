import argparse
import time

import cv2
import numpy as np
import onnxruntime

from postprocess import get_detections
from utils import vis

# Use more steps to get more stable inference speed measurement
WARMUP_STEPS = 1  # 30
INFERENCE_STEPS = 1  # 30
# CUDAExecutionProvider, CPUExecutionProvider
PROVIDERS = ["CPUExecutionProvider"]
OUTPUT_IMG = "onnx_predictions.jpg"


def get_parser():
    parser = argparse.ArgumentParser(description="Darknet to ONNX")
    parser.add_argument(
        "--img",
        "-i",
        type=str,
        required=True,
        help="RGB image (.jpg/.png...) for visualization",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        type=int,
        help=(
            "If batch size > 0, ONNX model will be static. If batch size <= 0, "
            "ONNX model will be dynamic."
        ),
    )
    parser.add_argument(
        "--to-float16",
        action="store_true",
        help="Use onnxmltools to convert to float16 model",
    )
    parser.add_argument(
        "--onnx_model", "-o", default="model.onnx", help="Output file path"
    )
    parser.add_argument("--score", default=0.3, type=float)
    parser.add_argument("--nms", default=0.45, type=float)
    parser.add_argument("--names", "-n", default="", type=str)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument('--out', default='out/onnx_result.png', type=str)
    return parser.parse_args()


def detect(
    session, image_path, score_thresh=0.1, nms_thresh=0.45, to_float16=False
):
    # preprocess
    t1 = time.time()
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    image_src = cv2.imread(image_path)
    resized = cv2.resize(
        image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR
    )
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)  # HWC to CHW
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)
    if to_float16:
        img_in = img_in.astype(np.float16)

    # warm-up
    t2 = time.time()
    input_name = session.get_inputs()[0].name
    for _ in range(WARMUP_STEPS):
        _ = session.run(
            None, {input_name: img_in}
        )  # output = [[batch, num, 4 + num_classes]]

    # inference
    t3 = time.time()
    for _ in range(INFERENCE_STEPS):
        # output = [[batch, num, 4 + num_classes]]
        outputs = session.run(None, {input_name: img_in})

    # because postprocessing only supports float32
    outputs = [output.astype(np.float32) for output in outputs]

    # for testing confidence threshold
    raw_boxes = np.squeeze(outputs[0])
    raw_scores = np.squeeze(outputs[1])

    # postprocess
    t4 = time.time()
    final_boxes, final_scores, final_cls_inds = get_detections(
        outputs, score_thresh, nms_thresh
    )
    t5 = time.time()

    # time analysis
    print(f"Preprocessing : {t2 - t1:.4f}s")
    print(f"Inference     : {(t4 - t3) / INFERENCE_STEPS:.4f}s")
    print(f"Postprocessing: {t5 - t4:.4f}s")
    print(
        f"Total         : "
        f"{t2 - t1 + (t4 - t3) / INFERENCE_STEPS + t5 - t4:.4f}s"
    )
    return final_boxes, final_scores, final_cls_inds, raw_boxes, raw_scores


def read_names(names_path):
    if names_path == "":
        return None

    class_names = []
    with open(names_path, "r") as f:
        for line in f:
            class_names.append(line.strip())
    return class_names


def main(args):
    # load ONNX model
    session = onnxruntime.InferenceSession(args.onnx_model, providers=PROVIDERS)

    # detect 1 image
    final_boxes, final_scores, final_cls_inds, raw_boxes, raw_scores = detect(
        session, args.img, args.score, args.nms, args.to_float16
    )

    # visualization
    class_names = read_names(args.names)

    # for testing confidence threshold
    cls_inds = []
    scores = []
    for i in range(len(raw_scores)):
        cls_ind = np.argmax(raw_scores[i])
        cls_inds.append(cls_ind)
        scores.append(max(raw_scores[i]))
    result_tmp = vis(
        args.img,
        raw_boxes,
        scores,
        cls_inds,
        conf=0.0,
        class_names=class_names,
        out_img='out/onnx_result_raw.png',
        print_bbox=True,
    )

    result = vis(
        args.img,
        final_boxes,
        final_scores,
        final_cls_inds,
        conf=args.score,
        class_names=class_names,
        out_img=args.out,
        print_bbox=True,
    )

    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_parser()
    for k, v in vars(args).items():
        print(f"{k:10}: {v}")
    main(args)
