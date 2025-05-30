import os

import numpy as np
import torch
from PIL import Image
from scipy.stats import linregress

from ultralytics import YOLO
import cv2
import copy
import re


det_model = YOLO("D:/User/Desktop/ultralytics/checkpoint/det/yolov8_det.pt")  # yolov8 : map 0.7
seg_model = YOLO("D:/User/Desktop/runs/segment/train14/weights/best.pt")


def expand_bounding_boxes(boxes, img_path, expansion_ratio=1.8):
    expanded_boxes = []
    cls_conf = []
    for box in boxes:
        if box.cls.item() == 0:  # 只处理类别ID为0的边界框
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            new_width, new_height = width * expansion_ratio, height * expansion_ratio
            new_x1, new_y1 = max(0, center_x - new_width / 2), max(0, center_y - new_height / 2)
            new_x2, new_y2 = min(1, center_x + new_width / 2), min(1,
                                                                   center_y + new_height / 2)
            expanded_boxes.append((new_x1, new_y1, new_x2, new_y2))
            cls_conf.append(box.conf)
    return expanded_boxes, cls_conf


def scale_image(masks, im0_shape, ratio_pad=None):
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def mask_to_seg_img(pred_mask, pred_cls, img_path, crop_img_shape, xy_list, pred_conf, pred_bbox,
                    alpha=0.2):
    seg_img = cv2.imread(img_path)
    pd_nums = len(pred_mask)
    masks = np.zeros_like(seg_img, dtype=np.uint8)
    fontsize = seg_img.shape[0] / 1500
    fontscale = int(fontsize * 2)
    for i in range(pd_nums):
        x1, y1, x2, y2 = xy_list[i]
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        pd_mask = scale_image(pred_mask[i], crop_img_shape[i])
        pd_mask = pd_mask.squeeze(-1)
        if pred_cls[i] == 0:
            masks[y1:y2, x1:x2][pd_mask > 0] = (0, 0, 255)
        else:
            masks[y1:y2, x1:x2][pd_mask > 0] = (255, 0, 0)

    res = cv2.addWeighted(masks, alpha, seg_img, 1 - alpha, 0)

    if len(pred_bbox) != 0:
        box_nms = [tensor.cpu().numpy() for tensor in pred_bbox]
        box_nms = np.array(box_nms)
        conf_nms = [tensor.cpu().numpy() for tensor in pred_conf]
        conf_nms = np.array(conf_nms)
        xy_nms = np.array(xy_list)
        cls_nms = np.array(pred_cls)
        box_nms[:, :1] += xy_nms[:, :1]
        box_nms[:, 2:3] += xy_nms[:, :1]
        box_nms[:, 1:2] += xy_nms[:, 1:2]
        box_nms[:, 3:] += xy_nms[:, 1:2]
        res_bbox, res_conf, res_cls = nms(box_nms, conf_nms, 0.5, cls_nms)
        for i in range(len(res_cls)):
            minx, miny = res_bbox[i][:2]
            maxx, maxy = res_bbox[i][2:]
            minx, miny = int(minx), int(miny)
            maxx, maxy = int(maxx), int(maxy)
            conf = res_conf[i]
            if res_cls[i] == 0:
                txt = f"{'strawberry'} {conf:.2f}"
                res = cv2.rectangle(res, (minx, miny), (maxx, maxy), (0, 0, 255), 4)
                res = cv2.putText(res, txt, (minx, miny - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 3)
            else:
                txt = f"{'stem'} {conf:.2f}"
                res = cv2.rectangle(res, (minx, miny), (maxx, maxy), (255, 0, 0), 4)
                res = cv2.putText(res, txt, (minx, miny - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 0, 0), 3)
    return res


def calculate_min_distance(point_set_a, point_set_b):
    min_distance = float('inf')

    for point_a in point_set_a:
        for point_b in point_set_b:
            # 计算当前两个点之间的距离
            distance = 0
            for i in range(len(point_a)):
                distance += (point_a[i] - point_b[i]) ** 2
            distance = distance ** 0.5  # 取平方根

            if distance < min_distance:
                min_distance = distance

    return min_distance


def find_farthest_point_and_line_equation(points):
    """找到离质心最远的点，并计算线的方程的A、B、C"""
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    farthest_index = np.argmax(distances)
    farthest_point = points[farthest_index]

    # 计算直线方程的A、B、C
    A = farthest_point[1] - centroid[1]  # y2 - y1
    B = centroid[0] - farthest_point[0]  # x1 - x2
    C = A * centroid[0] + B * centroid[1]  # Ax + By = C

    return centroid, farthest_point, A, B, C


def calculate_average_distance(points, A, B, C):
    """计算点集到直线的平均距离"""
    distances = np.abs(A * points[:, 0] + B * points[:, 1] - C) / np.sqrt(A ** 2 + B ** 2)
    average_distance = np.mean(distances)
    return average_distance


# 匹配最近距离的草莓与草莓茎
def find_nearest_stems(pred_mask_xy, pred_cls, pred_mask, crop_img_shape, xyxy_list, pred_conf,
                       pred_bbox):
    # 分离草莓和茎的点集
    strawberries = [pred_mask_xy[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems = [pred_mask_xy[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_mask = [pred_mask[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_mask = [pred_mask[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_xyxy = [xyxy_list[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_xyxy = [xyxy_list[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_conf = [pred_conf[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_conf = [pred_conf[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_shape = [crop_img_shape[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_shape = [crop_img_shape[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_cls = [0 for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_cls = [1 for i in range(len(pred_cls)) if pred_cls[i] == 1]
    strawberries_bbox = [pred_bbox[i] for i in range(len(pred_cls)) if pred_cls[i] == 0]
    stems_bbox = [pred_bbox[i] for i in range(len(pred_cls)) if pred_cls[i] == 1]
    # 用于存储每个草莓及其最近的茎
    nearest_stems = []
    res_crop_img_shape = []
    res_pred_mask = []
    res_pred_cls = []
    res_xyxy_list = []
    res_pred_conf = []
    res_pred_bbox = []
    for i in range(len(strawberries)):
        min_distance = float('inf')
        nearest_stem = None
        inds = -1
        strawberry_centroid = np.mean(strawberries[i], axis=0)
        centroid, farthest_point, A, B, C = find_farthest_point_and_line_equation(strawberries[i])
        for j in range(len(stems)):
            stem_centroid = np.mean(stems[j], axis=0)
            if stem_centroid[-1] < strawberry_centroid[-1]:
                if strawberries_shape[i] == stems_shape[j]:
                    distance_A = calculate_min_distance(strawberries[i], stems[j])
                    distance_B = calculate_average_distance(stem_centroid.reshape(1, 2), A, B, C)
                    distance = distance_A + distance_B
                    if distance < min_distance:
                        min_distance = distance
                        nearest_stem = stems[j]
                        inds = j
            else:
                continue
        res_crop_img_shape.append(strawberries_shape[i])
        res_pred_mask.append(strawberries_mask[i])
        res_pred_cls.append(strawberries_cls[i])
        res_xyxy_list.append(strawberries_xyxy[i])
        res_pred_conf.append(strawberries_conf[i])
        res_pred_bbox.append(strawberries_bbox[i])
        if min_distance != float('inf'):
            res_crop_img_shape.append(stems_shape[inds])
            res_pred_mask.append(stems_mask[inds])
            res_pred_cls.append(stems_cls[inds])
            res_xyxy_list.append(stems_xyxy[inds])
            res_pred_conf.append(stems_conf[inds])
            res_pred_bbox.append(stems_bbox[inds])
        nearest_stems.append(strawberries[i])
        nearest_stems.append(nearest_stem)

    return res_pred_mask, res_pred_cls, res_crop_img_shape, res_xyxy_list, res_pred_conf, res_pred_bbox


def resize_expand_box(expaned_bbox, img_path, ct):
    img_name = re.search(r'\d+', img_path).group()
    print(img_name)
    pred_mask = []
    pred_cls = []
    crop_img_shape = []
    xyxy_list = []
    pred_mask_xyn = []
    pred_conf = []
    pred_mask_xy = []
    pred_bbox = []
    for box in expaned_bbox:
        x1, y1, x2, y2 = box
        # 加载原图
        seg_img = cv2.imread(img_path)
        # 原图的高宽
        seg_img_height, seg_img_width = seg_img.shape[:2]
        # 计算 det_bbox 在原图上的点
        x1, x2 = x1 * seg_img_width, x2 * seg_img_width
        y1, y2 = y1 * seg_img_height, y2 * seg_img_height
        xyxy = (x1, y1, x2, y2)
        # rescaled_cropped_image
        crop_img = seg_img[round(y1):round(y2), round(x1):round(x2)]
        # crop_img_height, crop_img_width = crop_img.shape[:2]

        # 得到 rescaled_cropped_image 的 result
        seg_res = seg_model(crop_img)
        rescaled_img_pred_mask = img_path.replace('images', 'images_mask/rescaled_cropped_image_mask/')
        seg_res[0].save(rescaled_img_pred_mask)
        # 获取 掩码图像 掩码坐标 预测得分 预测类别'
        if seg_res[0].masks is None:
            return

        nums = len(seg_res[0].boxes.conf)
        for i in range(nums):
            if seg_res[0].boxes.cls[i] == 1:
                if seg_res[0].boxes.conf[i] > 0.1:
                    print(f"Box {i} conf: {seg_res[0].boxes.conf[i]}")
                    pred_mask.append(seg_res[0].masks.data.cpu().numpy()[i])
                    crop_img_shape.append(crop_img.shape[:2])
                    xyxy_list.append(xyxy)
                    pred_mask_xy.append(seg_res[0].masks.xy[i])
                    pred_cls.append(seg_res[0].boxes.cls[i])
                    pred_conf.append(seg_res[0].boxes.conf[i])
                    pred_bbox.append(seg_res[0].boxes.xyxy[i])
            if seg_res[0].boxes.cls[i] == 0:
                if seg_res[0].boxes.conf[i] > 0.6:
                    print(f"Box {i} conf: {seg_res[0].boxes.conf[i]}")
                    pred_mask.append(seg_res[0].masks.data.cpu().numpy()[i])
                    crop_img_shape.append(crop_img.shape[:2])
                    xyxy_list.append(xyxy)
                    pred_mask_xy.append(seg_res[0].masks.xy[i])
                    pred_cls.append(seg_res[0].boxes.cls[i])
                    pred_conf.append(seg_res[0].boxes.conf[i])
                    pred_bbox.append(seg_res[0].boxes.xyxy[i])


    pred_scale_mask_xy = copy.deepcopy(pred_mask_xy)
    # 根据crop_img_shape的大小 筛选 属于同一张crop_image的预测结果，对单个crop_image的预测结果筛选 预测草莓与草莓茎之间的最短距离
    # 最后将所有的结果保存到同一个列表下，可视化
    # 待优化
    pred_mask, pred_cls, crop_img_shape, xyxy_list, pred_conf, pred_bbox = find_nearest_stems(
        pred_scale_mask_xy,
        pred_cls, pred_mask,
        crop_img_shape,
        xyxy_list,
        pred_conf,
        pred_bbox)

    output_path = img_path.replace('images', 'images_mask/full_size_img_mask')
    # output_path = output_path.replace('test', 'full_size_img_mask/test')

    res = mask_to_seg_img(pred_mask, pred_cls, img_path, crop_img_shape, xyxy_list, pred_conf, pred_bbox)

    cv2.imwrite(output_path, res)

    return crop_img


def calculate_iou(boxA, boxB):
    # 计算交集
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算并集
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    # 计算 IoU
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou


def nms(boxes, scores, threshold, cls=None):
    n = len(boxes)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:  # 跳过已被剔除的框
            continue

        for j in range(i + 1, n):
            if not keep[j]:  # 跳过已被剔除的框
                continue

            iou = calculate_iou(boxes[i], boxes[j])
            if iou > threshold:
                # 保留置信度高的框
                if scores[i] > scores[j]:
                    keep[j] = False  # 剔除 j
                else:
                    keep[i] = False  # 剔除 i
                    break  # 一旦 i 被剔除，停止检查
    if cls is not None:
        return [boxes[i] for i in range(n) if keep[i]], [scores[i] for i in range(n) if keep[i]], [cls[i] for i in
                                                                                                   range(n) if keep[i]]
    else:
        return [boxes[i] for i in range(n) if keep[i]], [scores[i] for i in range(n) if keep[i]]

if __name__ == '__main__':
    results = det_model("D:/User/Desktop/ultralytics/data/test/images/16.jpg", save=False)
    no_obj_count = 0
    for result in results:
        # if result.boxes.xyxy.numel() == 0:
        #     no_obj_count += 1
        #     continue
        strawberry_exist = torch.eq(result.boxes.cls, 0).any()

        # 是否检测出草莓
        if len(result.boxes) == 0:
            no_obb_path = result.path.replace('images', 'images_mask/no_od_img')
            result.save(no_obb_path)
            no_obj_count += 1
        else:
            obb_path = result.path.replace('images', 'images_mask/od_img')
            result.save(obb_path)
            boxes = result.boxes
            expand_boxes, cls_conf = expand_bounding_boxes(boxes, result.path, expansion_ratio=2.0)
            nms_boxes = nms(expand_boxes, cls_conf, 0.7)
            expand_boxes, cls_conf = nms_boxes[0], nms_boxes[1]
            rescaled_obb_path = result.path.replace('images', 'images_mask/rescaled_cropped_img')
            rescaled_img_bbox = cv2.imread(result.path)
            for xx in range(len(expand_boxes)):
                x1, y1, x2, y2 = expand_boxes[xx]
                x1, x2 = round(x1 * result.orig_img.shape[1]), round(x2 * result.orig_img.shape[1])
                y1, y2 = round(y1 * result.orig_img.shape[0]), round(y2 * result.orig_img.shape[0])
                # 获得检测模型的预测得分
                pred_cls_conf = float(cls_conf[xx])
                txt = f"{'strawberry'} {pred_cls_conf:.2f}"
                rescaled_img_bbox = cv2.rectangle(rescaled_img_bbox, (x1, y1), (x2, y2), (0, 0, 255), 6)
                rescaled_img_bbox = cv2.putText(rescaled_img_bbox, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 4)
            cv2.imwrite(rescaled_obb_path, rescaled_img_bbox)
            count = 0
            seg_img_input = resize_expand_box(expand_boxes, result.path, count)

    print(no_obj_count)
