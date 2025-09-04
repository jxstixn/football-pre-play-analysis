import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def extract_bright_features(img, tophat_kernel_size=50, thresh=70):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel_size, tophat_kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Threshold to get binary image
    _, binary = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)

    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small, iterations=1)

    return binary


def extend_and_cluster(yard_lines, img_height, eps=20, min_samples=1):
    """
    Extend each segment to full height, compute top/bottom intercepts,
    then cluster them via DBSCAN in the (x_top, x_bot) space.
    """
    intercepts = []
    # 1) for each segment compute its (x_top, x_bot)
    for (x1, y1, x2, y2) in yard_lines:
        # line in ax + by + c = 0 form
        # but easier: parametric slope/intercept
        if x2 == x1:
            # perfect vertical
            x_top = x_bot = x1
        else:
            m = (y2 - y1) / (x2 - x1)  # slope dy/dx
            # invert to dx/dy for x(y) = x1 + (y - y1)/m
            inv_m = (x2 - x1) / (y2 - y1)  # dx/dy
            x_top = x1 + (0 - y1) * inv_m
            x_bot = x1 + (img_height - y1) * inv_m
        intercepts.append([x_top, x_bot])

    intercepts = np.array(intercepts)

    # 2) cluster via DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intercepts)
    labels = clustering.labels_
    unique_labels = sorted(set(labels))  # -1 is noise if any

    # 3) for each cluster, average the intercepts → yield one “master” line
    masters = []
    for lbl in unique_labels:
        if lbl < 0:
            continue
        pts = intercepts[labels == lbl]
        top_avg, bot_avg = pts.mean(axis=0)
        masters.append((top_avg, bot_avg))

    return masters


def detect_yard_lines(img_path: str, output_img_path: str):
    img = cv2.imread(img_path)

    # Resize image if not already 720x1280
    if img.shape != (720, 1280, 3):
        img = cv2.resize(img, (720, 1280, 3))

    # Step 1: Extract bright features
    binary = extract_bright_features(img, tophat_kernel_size=50, thresh=70)

    # Step 2: Edge detection on the bright‐feature binary
    edges = cv2.Canny(binary,
                      threshold1=50,  # lower hysteresis threshold
                      threshold2=150,  # upper hysteresis threshold
                      apertureSize=3,  # Sobel kernel size
                      L2gradient=True)

    # Step 3: Morphological closing to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 4: Hough Line Transform to detect lines
    lines_p = cv2.HoughLinesP(
        edges_closed,
        rho=1,
        theta=np.pi / 180,
        threshold=200,  # lower threshold for line detection
        minLineLength=int(img.shape[0] * 0.45),  # vertical lines are long
        maxLineGap=200
    )

    # Step 5: Filter out only near-vertical lines
    yard_lines = []
    if lines_p is not None:
        for x1, y1, x2, y2 in lines_p[:, 0]:
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle - np.pi / 2) < np.deg2rad(45):  # within ±
                yard_lines.append((x1, y1, x2, y2))

    # Step 6: Extend and cluster lines
    H, W = img.shape[:2]
    masters = extend_and_cluster(yard_lines, H, eps=30)

    output_img = img.copy()

    for x_top, x_bot in masters:
        pt1 = (int(round(x_top)), 0)
        pt2 = (int(round(x_bot)), H)
        cv2.line(output_img, pt1, pt2, (0, 0, 255), 3)

    cv2.imwrite(output_img_path, output_img)

    return masters


if __name__ == "__main__":
    img_path = "C:/Users\Justin.Getzke\AppData\Roaming\com.uoc.football-pre-play-analysis\extracted/2108 BOISE STATE OFF vs COLORADO STATE\snaps\play_005_snap2.jpg"
    output_img_path = "yard_lines.jpg"
    detect_yard_lines(img_path, output_img_path)
    print(f"Output written to {output_img_path}")