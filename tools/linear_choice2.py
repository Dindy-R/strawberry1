import numpy as np

def find_longest_axis(mask):
    y, x = np.where(mask)
    centroid_x, centroid_y = np.mean(x), np.mean(y)
    max_distance = 0
    direction = None

    for i in range(len(x)):
        distance = np.sqrt((x[i] - centroid_x) ** 2 + (y[i] - centroid_y) ** 2)
        if distance > max_distance:
            max_distance = distance
            direction = (x[i], y[i])

    return direction, (centroid_x, centroid_y)

def extend_line(direction, centroid, length=50):
    centroid_x, centroid_y = centroid
    dx = direction[0] - centroid_x
    dy = direction[1] - centroid_y

    if dx == 0:
        dx = 1e-6
    if dy == 0:
        dy = 1e-6

    line_start = (int(centroid_x - dx * length), int(centroid_y - dy * length))
    line_end = (int(centroid_x + dx * length), int(centroid_y + dy * length))

    return line_start, line_end

def calculate_distance(point, line_start, line_end):
    a = line_end[1] - line_start[1]
    b = line_start[0] - line_end[0]
    c = line_end[0] * line_start[1] - line_end[1] * line_start[0]
    distance = abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return distance

def calculate_angle_between_lines(line1, line2):
    """
    计算两条直线之间的夹角。

    :param line1: 第一条直线的两个端点 ((x1, y1), (x2, y2))
    :param line2: 第二条直线的两个端点 ((x3, y3), (x4, y4))
    :return: 夹角（弧度）
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    vec1 = (x2 - x1, y2 - y1)
    vec2 = (x4 - x3, y4 - y3)

    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    angle_rad = np.arccos(dot_product / (norm1 * norm2))
    return angle_rad

def process_strawberries_and_stems2(strawberry_masks, stem_masks):
    selected_stem_mask = None
    min_angle = float('inf')
    extend_lines = []

    for mask in strawberry_masks:
        direction, centroid = find_longest_axis(mask)
        line_start, line_end = extend_line(direction, centroid, length=50)
        extend_lines.append((line_start, line_end))

        centroid_strawberry = centroid
        for stem_mask in stem_masks:
            centroid_stem = calculate_centroid(stem_mask)
            stem_line_start = centroid_stem
            stem_line_end = centroid_strawberry

            angle = calculate_angle_between_lines((line_start, line_end), (stem_line_start, stem_line_end))

            if angle < min_angle:
                min_angle = angle
                selected_stem_mask = stem_mask

    return selected_stem_mask
