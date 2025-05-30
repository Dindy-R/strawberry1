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

    return direction

def extend_line(direction, length=50):
    centroid_x, centroid_y = np.mean(np.where(direction[0])), np.mean(np.where(direction[1]))
    if centroid_x == direction[0]:
        dx = 0
    else:
        dx = direction[0] - centroid_x

    if centroid_y == direction[1]:
        dy = 0
    else:
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

def process_strawberries_and_stems(strawberry_masks, stem_masks):

    selected_stem_masks = []

    for mask in strawberry_masks:
        direction = find_longest_axis(mask)
        centroid_x, centroid_y = np.mean(np.where(mask)[1]), np.mean(np.where(mask)[0])
        line_start, line_end = extend_line(direction, length=50)

        min_distance = float('inf')
        closest_stem_mask = None

        for stem_mask in stem_masks:
            y, x = np.where(stem_mask)
            centroid_x_stem, centroid_y_stem = np.mean(x), np.mean(y)
            distance = calculate_distance((centroid_x_stem, centroid_y_stem), line_start, line_end)

            if distance < min_distance or (distance == 0 and line_start[0] <= centroid_x_stem <= line_end[0]):
                min_distance = distance
                closest_stem_mask = stem_mask

        if closest_stem_mask is not None:
            selected_stem_masks.append(closest_stem_mask)

    return selected_stem_masks