import numpy as np

"""
nms & multiclass_nms_class_aware are borrowed from YOLOX
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
"""


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    if len(scores.shape) < 2:
        scores = scores.reshape((scores.shape[0], 1))
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def get_detections(predictions, score_thresh=0.5, nms_thresh=0.45):
    # boxes = predictions[:, :4]
    # scores = predictions[:, 4:]

    # darknet_onnx를 사용하여 변환한 onnx 파일
    boxes = np.squeeze(predictions[0])
    scores = np.squeeze(predictions[1])
    dets = multiclass_nms_class_aware(
        boxes, scores, nms_thr=nms_thresh, score_thr=score_thresh
    )
    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls_inds = dets[:, 5]
        return final_boxes, final_scores, final_cls_inds
    else:
        return None, None, None
