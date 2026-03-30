import cv2
import numpy as np

def process_floorplan(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # EDGE IMAGE (NEW)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite("static/edges.png", edges)

    # CLEANING (NEW)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    elements = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 🔴 NEW: NOISE REMOVAL
        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 🔴 NEW: BORDER FILTER
        if y < 10 or (y+h) > (H-10):
            continue

        # 🔴 NEW: CLASSIFICATION
        if area > 6000:
            t = "wall"
        elif area > 2000:
            t = "door"
        else:
            t = "window"

        elements.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "type": t
        })

    return elements