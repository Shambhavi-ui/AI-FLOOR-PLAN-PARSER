import cv2
import numpy as np
import math


def crop_to_paper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.15 * h * w:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if best is None or area > best[0]:
            best = (area, x, y, cw, ch)

    if best is None:
        return img, gray, 0, 0

    _, x, y, cw, ch = best
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    cw = min(w - x, cw + pad * 2)
    ch = min(h - y, ch + pad * 2)
    cropped = img[y:y+ch, x:x+cw]
    return cropped, gray[y:y+ch, x:x+cw], x, y


def make_edge_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    return edges


def normalize_line(x1, y1, x2, y2, x_off, y_off):
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 12:
        x2 = x1
    if abs(dy) < 12:
        y2 = y1

    x1 += x_off
    y1 += y_off
    x2 += x_off
    y2 += y_off

    if (x1, y1) > (x2, y2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return int(x1), int(y1), int(x2), int(y2)


def merge_intervals(intervals, tolerance=8):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def merge_collinear_lines(lines, tolerance=8):
    vertical = {}
    horizontal = {}

    def find_group(key, groups):
        for existing in groups:
            if abs(existing - key) <= tolerance:
                return existing
        return key

    for x1, y1, x2, y2 in lines:
        if abs(x1 - x2) < tolerance:
            x = int(round((x1 + x2) / 2))
            group = find_group(x, vertical)
            start, end = sorted([y1, y2])
            vertical.setdefault(group, []).append((start, end))
        elif abs(y1 - y2) < tolerance:
            y = int(round((y1 + y2) / 2))
            group = find_group(y, horizontal)
            start, end = sorted([x1, x2])
            horizontal.setdefault(group, []).append((start, end))
        else:
            # Keep any non-axis-aligned line as-is.
            pass

    merged = []
    for x, intervals in vertical.items():
        for start, end in merge_intervals(intervals, tolerance):
            merged.append((x, start, x, end))
    for y, intervals in horizontal.items():
        for start, end in merge_intervals(intervals, tolerance):
            merged.append((start, y, end, y))
    return merged


def snap_wall_endpoints_to_grid(wall_lines, tolerance=12):
    if not wall_lines:
        return wall_lines

    def cluster_positions(values):
        sorted_values = sorted(values)
        clusters = []
        for value in sorted_values:
            placed = False
            for cluster in clusters:
                if abs(cluster[0] - value) <= tolerance:
                    cluster.append(value)
                    placed = True
                    break
            if not placed:
                clusters.append([value])
        return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]

    vertical_x = cluster_positions({x for x1, y1, x2, y2 in wall_lines if abs(x1 - x2) < tolerance for x in (x1, x2)})
    horizontal_y = cluster_positions({y for x1, y1, x2, y2 in wall_lines if abs(y1 - y2) < tolerance for y in (y1, y2)})

    def snap_value(value, candidates):
        if not candidates:
            return value
        best = min(candidates, key=lambda c: abs(c - value))
        return best if abs(best - value) <= tolerance else value

    snapped = set()
    for x1, y1, x2, y2 in wall_lines:
        if abs(y1 - y2) < tolerance:
            x1 = snap_value(x1, vertical_x)
            x2 = snap_value(x2, vertical_x)
            y1 = snap_value(y1, horizontal_y)
            y2 = y1
        elif abs(x1 - x2) < tolerance:
            y1 = snap_value(y1, horizontal_y)
            y2 = snap_value(y2, horizontal_y)
            x1 = snap_value(x1, vertical_x)
            x2 = x1
        else:
            x1 = snap_value(x1, vertical_x)
            x2 = snap_value(x2, vertical_x)
            y1 = snap_value(y1, horizontal_y)
            y2 = snap_value(y2, horizontal_y)

        if (x1, y1) > (x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        snapped.add((x1, y1, x2, y2))

    return snapped


def split_axis_aligned_intersections(wall_lines, tolerance=12):
    vertical = []
    horizontal = []
    others = []

    for x1, y1, x2, y2 in wall_lines:
        if abs(x1 - x2) < tolerance:
            y_start, y_end = sorted([y1, y2])
            vertical.append((x1, y_start, x2, y_end))
        elif abs(y1 - y2) < tolerance:
            x_start, x_end = sorted([x1, x2])
            horizontal.append((x_start, y1, x_end, y2))
        else:
            others.append((x1, y1, x2, y2))

    normalized = []
    for x, y1, x2, y2 in vertical:
        ys = {y1, y2}
        for hx1, hy, hx2, _ in horizontal:
            if hx1 - tolerance <= x <= hx2 + tolerance and y1 - tolerance <= hy <= y2 + tolerance:
                ys.add(hy)
        ys = sorted(ys)
        for a, b in zip(ys, ys[1:]):
            if b - a >= 4:
                normalized.append((x, a, x, b))

    for x1, y, x2, y2 in horizontal:
        xs = {x1, x2}
        for vx, vy1, vx2, vy2 in vertical:
            if vy1 - tolerance <= y <= vy2 + tolerance and x1 - tolerance <= vx <= x2 + tolerance:
                xs.add(vx)
        xs = sorted(xs)
        for a, b in zip(xs, xs[1:]):
            if b - a >= 4:
                normalized.append((a, y, b, y))

    return normalized + others


def normalize_wall_lines(wall_lines, tolerance=12):
    if not wall_lines:
        return wall_lines
    wall_lines = set(merge_collinear_lines(wall_lines, tolerance=tolerance))
    wall_lines = snap_wall_endpoints_to_grid(wall_lines, tolerance=tolerance)
    wall_lines = set(split_axis_aligned_intersections(wall_lines, tolerance=tolerance))
    wall_lines = set(merge_collinear_lines(wall_lines, tolerance=tolerance))
    return wall_lines


def remove_border_contours(mask, padding=4):
    # Remove unwanted contours that touch the image border,
    # such as false bottom strips or paper edges.
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = mask.copy()
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if x <= padding or y <= padding or x + cw >= w - padding or y + ch >= h - padding:
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=cv2.FILLED)
    return cleaned


def clean_edges(edges, min_area=100):
    # Filter out very small noisy contours after morphological cleanup.
    cleaned = np.zeros_like(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255)
    return cleaned


def build_structural_wall_mask(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    mask = np.zeros_like(dark)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area < 1500:
            continue
        if max(w, h) < 20:
            continue
        if area < 4000 and min(w, h) > 0.5 * max(w, h):
            continue
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def filter_openings_by_overlap(rects, overlap_mask, x_off, y_off, min_ratio=0.05):
    filtered = []
    h, w = overlap_mask.shape
    for rect in rects:
        x, y, rect_w, rect_h = rect
        rx = int(round(x - x_off))
        ry = int(round(y - y_off))
        if rx < 0 or ry < 0 or rx + rect_w > w or ry + rect_h > h:
            continue
        sub = overlap_mask[ry:ry + rect_h, rx:rx + rect_w]
        if sub.size == 0:
            continue
        if cv2.countNonZero(sub) / float(sub.size) >= min_ratio:
            filtered.append(rect)
    return filtered


def watershed_room_segments(wall_mask, cropped, x_off, y_off, min_area=10000):
    # Split a single connected interior region into separate rooms.
    interior = cv2.bitwise_not(wall_mask)
    if cv2.countNonZero(interior) == 0:
        return []

    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist <= 0:
        return []

    _, peaks = cv2.threshold(dist, max_dist * 0.35, 255, cv2.THRESH_BINARY)
    peaks = np.uint8(peaks)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    if cv2.countNonZero(peaks) == 0:
        return []

    _, markers = cv2.connectedComponents(peaks)
    if markers.max() <= 0:
        return []

    markers = markers + 1
    markers[wall_mask == 255] = 0
    markers = markers.astype(np.int32)

    image = cropped.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(image, markers)

    rooms = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        if label == -1:
            continue
        region = np.uint8(markers == label) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        region = clean_edges(region, min_area=500)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue
        rooms.append({
            'bounds': [
                [int(max(0, x + x_off)), int(max(0, y + y_off))],
                [int(min(cropped.shape[1], x + w) + x_off), int(min(cropped.shape[0], y + h) + y_off)]
            ],
            'label': classify_room_area(area),
            'area': int(area)
        })

    rooms.sort(key=lambda r: r['area'], reverse=True)
    return rooms


def preprocess_floor_plan(image_path):
    # Read the floor plan, crop the page region, and build a cleaned edge mask.
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0, 0

    cropped, _, x_off, y_off = crop_to_paper(img)
    edges = make_edge_mask(cropped)
    edges = remove_border_contours(edges)
    edges = clean_edges(edges)
    return cropped, edges, x_off, y_off


def parse_floor_plan(image_path):
    cropped, edges, x_off, y_off = preprocess_floor_plan(image_path)
    if edges is None:
        return []

    wall_lines = set()
    hough_input = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hough_input = cv2.dilate(hough_input, np.ones((2, 2), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(hough_input, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=40)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 60:
                continue
            x1, y1, x2, y2 = normalize_line(x1, y1, x2, y2, x_off, y_off)
            bw = abs(x2 - x1) + 1
            bh = abs(y2 - y1) + 1
            if bw <= 40 and bh <= 40:
                continue
            wall_lines.add((x1, y1, x2, y2))

    if wall_lines:
        wall_lines = normalize_wall_lines(set(wall_lines), tolerance=10)

    if not wall_lines:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 40 and h <= 40:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for i in range(len(approx)):
                x1, y1 = approx[i][0]
                x2, y2 = approx[(i + 1) % len(approx)][0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = math.hypot(dx, dy)
                if length < 60:
                    continue
                if not (dx < 12 or dy < 12):
                    continue
                x1, y1, x2, y2 = normalize_line(x1, y1, x2, y2, x_off, y_off)
                bw = abs(x2 - x1) + 1
                bh = abs(y2 - y1) + 1
                if bw <= 40 and bh <= 40:
                    continue
                wall_lines.add((x1, y1, x2, y2))

    if wall_lines:
        wall_lines = normalize_wall_lines(set(wall_lines), tolerance=10)

    walls = [[[x1, y1], [x2, y2]] for x1, y1, x2, y2 in sorted(wall_lines)]
    return walls


def get_edge_preview(image_path):
    cropped, edges, _, _ = preprocess_floor_plan(image_path)
    if edges is None:
        return None
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def find_doors(image_path):
    # Detect door footprints as moderate-size horizontal edge blobs.
    cropped, edges, x_off, y_off = preprocess_floor_plan(image_path)
    if edges is None:
        return []

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120 or area > 4000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if (30 < w < 120 and 10 < h < 40) or (10 < w < 40 and 30 < h < 120):
            doors.append([int(x + x_off), int(y + y_off), int(w), int(h)])
    return doors


def rects_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def find_windows(image_path):
    # Detect window candidates from small edge contours.
    cropped, edges, x_off, y_off = preprocess_floor_plan(image_path)
    if edges is None:
        return []

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_windows = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 2000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if 25 < w < 120 and 5 < h < 35:
            raw_windows.append([int(x + x_off), int(y + y_off), int(w), int(h)])

    # Exclude any candidate that overlaps a detected door.
    doors = find_doors(image_path)
    windows = []
    for win in raw_windows:
        if any(rects_overlap(win, door) for door in doors):
            continue
        windows.append(win)

    return windows


def _point_inside_room(x, y, room, margin=16):
    [x1, y1], [x2, y2] = room['bounds']
    return x >= x1 + margin and y >= y1 + margin and x <= x2 - margin and y <= y2 - margin


def _find_room_for_point(rooms, x, y):
    for room in rooms:
        if _point_inside_room(x, y, room, margin=16):
            return room
    return None


def _classify_furniture_shape(room_label, width, height, area):
    label = room_label.lower()
    aspect = float(width) / float(height) if height else 1.0

    if 'bath' in label:
        if area <= 3500 and max(width, height) <= 140:
            return 'toilet'
        return None

    if 'bed' in label or 'master' in label:
        if area >= 1800 and aspect >= 1.1:
            return 'bed'
        if area >= 1000 and 0.7 <= aspect <= 1.4:
            return 'bed'

    if 'living' in label or 'lounge' in label:
        if area >= 2000 and 0.8 <= aspect <= 1.6:
            return 'sofa'
        if area >= 1200 and 0.7 <= aspect <= 1.3:
            return 'table'

    if 'kitchen' in label or 'dining' in label:
        if area >= 1500 and 0.7 <= aspect <= 1.7:
            return 'table'

    return None


def find_furniture(image_path):
    cropped, edges, x_off, y_off = preprocess_floor_plan(image_path)
    if edges is None:
        return []

    rooms = find_rooms(image_path)
    if not rooms:
        return []

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    furniture = []
    seen = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 8000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 24 or h < 24:
            continue
        if x <= 10 or y <= 10 or x + w >= cropped.shape[1] - 10 or y + h >= cropped.shape[0] - 10:
            continue

        center_x = int(x + w / 2 + x_off)
        center_y = int(y + h / 2 + y_off)
        room = _find_room_for_point(rooms, center_x, center_y)
        if room is None:
            continue

        furniture_type = _classify_furniture_shape(room['label'], w, h, area)
        if furniture_type is None:
            continue

        is_duplicate = any(abs(x - ex) <= 20 and abs(y - ey) <= 20 and abs(w - ew) <= 20 and abs(h - eh) <= 20 for ex, ey, ew, eh in seen)
        if is_duplicate:
            continue

        seen.append((x, y, w, h))
        furniture.append({
            'type': furniture_type,
            'bounds': [int(x + x_off), int(y + y_off), int(w), int(h)],
            'room': room['label']
        })

    return furniture


def classify_room_area(area, bounds=None, image_shape=None):
    center_x = center_y = 0
    aspect = 1.0
    image_w = image_h = None
    if bounds is not None and image_shape is not None:
        x1, y1 = bounds[0]
        x2, y2 = bounds[1]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        aspect = float(width) / float(height)
        image_w, image_h = image_shape

    if area > 140000:
        if image_h and center_y < image_h * 0.35:
            return 'Hall'
        return 'Living Room'
    if area > 90000:
        if image_w and center_x > image_w * 0.55:
            return 'Kitchen'
        if image_h and center_y < image_h * 0.35:
            return 'Master Bedroom'
        return 'Bedroom'
    if area > 60000:
        if image_w and center_x > image_w * 0.55:
            return 'Dining Room'
        return 'Bedroom'
    if area > 30000:
        if aspect > 1.4:
            return 'Dining Room'
        return 'Bedroom'
    if area > 8000:
        return 'Bathroom'
    return 'Storage'


def find_rooms(image_path):
    cropped, edges, x_off, y_off = preprocess_floor_plan(image_path)
    if edges is None:
        return []

    walls = parse_floor_plan(image_path)
    if not walls:
        return []

    doors = find_doors(image_path)
    windows = find_windows(image_path)

    wall_mask = np.zeros_like(edges)
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        rx1 = int(round(x1 - x_off))
        ry1 = int(round(y1 - y_off))
        rx2 = int(round(x2 - x_off))
        ry2 = int(round(y2 - y_off))
        cv2.line(wall_mask, (rx1, ry1), (rx2, ry2), 255, thickness=18)

    for rect in doors + windows:
        x, y, w, h = rect
        rx = int(round(x - x_off))
        ry = int(round(y - y_off))
        if rx < 0 or ry < 0:
            continue
        left = max(0, rx - 6)
        top = max(0, ry - 6)
        right = min(wall_mask.shape[1], rx + w + 6)
        bottom = min(wall_mask.shape[0], ry + h + 6)
        cv2.rectangle(wall_mask, (left, top), (right, bottom), 255, thickness=-1)

    wall_mask = cv2.bitwise_or(wall_mask, edges)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    wall_mask = cv2.dilate(wall_mask, np.ones((7, 7), np.uint8), iterations=1)

    interior = cv2.bitwise_not(wall_mask)
    flood = interior.copy()
    h, w = flood.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 0)
    room_mask = cv2.bitwise_and(interior, cv2.bitwise_not(flood))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    room_mask = clean_edges(room_mask, min_area=500)

    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rooms = []
    image_area = cropped.shape[0] * cropped.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue

        border_touch = (x <= 15 or y <= 15 or x + w >= cropped.shape[1] - 15 or y + h >= cropped.shape[0] - 15)
        if border_touch and len(contours) > 1 and area < image_area * 0.4:
            continue

        rooms.append({
            'bounds': [
                [int(max(0, x + x_off)), int(max(0, y + y_off))],
                [int(min(cropped.shape[1], x + w) + x_off), int(min(cropped.shape[0], y + h) + y_off)]
            ],
            'label': classify_room_area(area),
            'area': int(area)
        })

    if not rooms and contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area > max(12000, image_area * 0.05):
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 40 and h >= 40:
                rooms.append({
                    'bounds': [
                        [int(max(0, x + x_off)), int(max(0, y + y_off))],
                        [int(min(cropped.shape[1], x + w) + x_off), int(min(cropped.shape[0], y + h) + y_off)]
                    ],
                    'label': classify_room_area(area),
                    'area': int(area)
                })

    rooms.sort(key=lambda r: r['area'], reverse=True)
    if len(rooms) <= 1:
        watershed_rooms = watershed_room_segments(wall_mask, cropped, x_off, y_off, min_area=10000)
        if len(watershed_rooms) > 1:
            return watershed_rooms
    return rooms