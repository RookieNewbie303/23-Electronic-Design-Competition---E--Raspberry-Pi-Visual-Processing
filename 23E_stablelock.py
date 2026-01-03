# -*- coding: utf-8 -*-

"""
Integrated laser-robust rectangle lock + sampling + FPS + serial sender.
Includes:
 - A: remove_laser_pixels (inpaint / median)
 - B: auto_canny
 - C: is_rect_contained (geometric checks)
 - D: stable_lock (multi-frame consensus)
Drop-in replacement for your earlier main loop.
"""
import cv2
import numpy as np
import serial
import time
import math
from collections import deque

# =============================================================================
# 1. Configuration
# =============================================================================

class Config:
    # --- Camera settings ---
    CAMERA_ID = 0
    FRAME_WIDTH = 800
    FRAME_HEIGHT = 600

    # crop & zoom
    CROP_Y_START = 0.1
    CROP_Y_END = 0.6
    CROP_X_START = 0.30
    CROP_X_END = 0.70
    ZOOM_FACTOR = 1.0

    # rectangle detection
    RECT_MIN_AREA_SET = 15000         # can be made relative
    PERCENT_OF_IMAGE = 0.03
    CANNY_TH1 = 10
    CANNY_TH2 = 30
    LOCK_TIME_WINDOW = 100.0

    # laser detection
    LASER_MIN_AREA = 5  
    RED_DIFF_THRESH = 60    
    VALUE_CLAHE_CLIP = 2.0  
    TOP_PERCENT = 0.0003
    
    # serial
    SERIAL_PORT = '/dev/ttyAMA0'
    BAUD_RATE = 9600
    SEND_INTERVAL = 0.050
    SAMPLE_COUNT = 1
    SAMPLE_INTERVAL = SEND_INTERVAL / float(SAMPLE_COUNT)

    # laser removal options (A)
    LASER_REMOVE_METHOD = 'median'   # 'inpaint' or 'median'
    LASER_DILATE_PX = 20              # expand mask before removing

    # auto canny (B)
    SIGMA = 0.33

    # stability (D)
    STABLE_FRAMES = 8                 # queue length to consider stable
    STABLE_MIN_COUNT = 3              # minimal valid entries required
    STABLE_MAX_DISP = 30.0            # max centroid dispersion in px

    # containment geometry (C)
    MIN_INNER_OUTER_RATIO = 0.6
    MAX_INNER_OUTER_RATIO = 0.9
    VERTEX_TOL = 30                   # minimal vertex separation tolerance
    
# =============================================================================
# 2. Utility functions
# =============================================================================
def open_camera(idx, width, height, retries=6, wait_s=1.0):
    for i in range(retries):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened():
            return cap
        else:
            try:
                cap.release()
            except:
                pass
            time.sleep(wait_s)
    return None

def order_points(pts):
    """ Order 4 points: top-left, top-right, bottom-right, bottom-left """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def point_segment_distance(a, b, p):
    """ Compute shortest distance from point p to segment ab """
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return np.linalg.norm(p - a)
    ap = p - a
    t = np.dot(ap, ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# =============================================================================
# A. Remove laser pixels (inpaint or median)
# =============================================================================
def remove_laser_pixels(img_bgr, laser_mask, method='inpaint', dilate_px=5):
    """
    Remove/replace bright laser pixels before edge detection.
    - laser_mask: binary mask (255 for laser pixels)
    - method: 'inpaint' (cv2.inpaint) or 'median' (local median fill)
    - dilate_px: expand mask to cover halo
    Returns cleaned BGR image (copy).
    """
    
    """Remove bright laser region from image using mask (255)"""
    if laser_mask is None:
        return img_bgr.copy()
    mask = (laser_mask > 0).astype('uint8') * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)
    if method == 'inpaint':
        try:
            cleaned = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return cleaned
        except Exception:
            return img_bgr.copy()
    else:
        # median fill fallback
        med = cv2.medianBlur(img_bgr, 7)
        cleaned = img_bgr.copy()
        ys, xs = np.where(mask > 0)
        if ys.size > 0:
            cleaned[ys, xs] = med[ys, xs]
        return cleaned

# =============================================================================
# B. auto_canny
# =============================================================================
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower == upper:
        lower = max(0, upper - 1)
    return cv2.Canny(gray, lower, upper)

# =============================================================================
# C. geometry checks for inner/outer rectangle
# =============================================================================
def rect_area(pts):
    return abs(cv2.contourArea(pts))

def is_rect_contained(outer_pts, inner_pts, min_ratio=Config.MIN_INNER_OUTER_RATIO, max_ratio=Config.MAX_INNER_OUTER_RATIO, vertex_tol=Config.VERTEX_TOL):
    """
    Validate that inner is truly inside outer and area ratio sensible.
    - min_ratio: inner_area / outer_area must be >= min_ratio
    - max_ratio: inner_area / outer_area must be <= max_ratio
    - vertex_tol: check that corresponding vertices aren't too close (to avoid near-coincident)
    """
    # outer_pts/inner_pts expected as (4,2) float32 or similar
    A_out = rect_area(outer_pts)
    A_in = rect_area(inner_pts)
    if A_out <= 0 or A_in <= 0:
        return False
    ratio = A_in / A_out
    if not (min_ratio <= ratio <= max_ratio):
        return False
    # inner centroid must be inside outer
    pt = tuple(inner_pts[0].ravel())
    if cv2.pointPolygonTest(outer_pts.astype(np.float32), pt, False) < 0:
        return False
    # centroids not coincident
    c_out = np.mean(outer_pts.reshape(-1,2), axis=0)
    c_in = np.mean(inner_pts.reshape(-1,2), axis=0)
    if np.linalg.norm(c_out - c_in) < 3.0:
        # if practically identical, suspicious (maybe overlapping)
        return False
    # vertex-wise distance: ensure vertices aren't almost identical
    close_count = 0
    for i in range(4):
        d = np.linalg.norm(outer_pts[i].ravel() - inner_pts[i].ravel())
        if d < vertex_tol:
            close_count += 1
    # if 3 or more vertices too close -> reject (likely coincident)
    if close_count >= 3:
        return False
    return True

# =============================================================================
# D. stable lock: multi-frame consensus
# =============================================================================
def stable_lock(queue, max_disp=Config.STABLE_MAX_DISP, min_count=Config.STABLE_MIN_COUNT):
    """
    queue: deque of items (outer_pts, inner_pts, timestamp)
    Return True if queue indicates stable detection.
    """
    if len(queue) < min_count:
        return False
    centers = [np.mean(item[0].reshape(-1,2), axis=0) for item in queue]
    centers = np.array(centers)
    mean_center = centers.mean(axis=0)
    max_disp_actual = float(np.max(np.linalg.norm(centers - mean_center, axis=1)))
    return max_disp_actual <= max_disp

# =============================================================================
# Laser centroid detection
# =============================================================================

def create_combined_laser_mask(img_bgr):
    """
    Return a binary mask of candidate laser pixels using:
      - HSV red ranges (wider)
      - R - max(G,B) red-diff threshold
      - CLAHE-enhanced V channel + percentile threshold (top-k)
    """
    # params (tune in Config instead if you prefer)
    red_s_min = 30      # allow lower saturation (black tape reduces S)
    red_v_min = 30      # allow lower brightness
    red_diff_thresh = getattr(Config, "RED_DIFF_THRESH", 40)   # R - max(G,B)
    top_percent = getattr(Config, "TOP_PERCENT", 0.0005)        # 0.05% brightest
    clahe_clip = getattr(Config, "VALUE_CLAHE_CLIP", 2.0)

    # 1) HSV red mask (wider than before)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, red_s_min, red_v_min])
    upper1 = np.array([12, 255, 255])
    lower2 = np.array([150, red_s_min, red_v_min])
    upper2 = np.array([180, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # 2) Red-difference mask (works when red channel stands out)
    b, g, r = cv2.split(img_bgr)
    max_gb = np.maximum(g, b).astype(np.int16)
    r_int = r.astype(np.int16)
    red_diff = (r_int - max_gb).astype(np.int16)
    _, mask_rdiff = cv2.threshold(red_diff, red_diff_thresh, 255, cv2.THRESH_BINARY)
    mask_rdiff = mask_rdiff.astype(np.uint8)

    # 3) CLAHE on V channel + percentile threshold (adaptive brightness)
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)

    # compute percentile threshold: top X fraction of pixels
    flat = v_clahe.flatten()
    # avoid zero-length
    if flat.size > 0:
        kth = int(max(1, flat.size * (1.0 - top_percent)))
        # use numpy partition for speed: threshold = value at kth largest
        thr_val = np.partition(flat, kth)[kth]
    else:
        thr_val = 255
    _, mask_top = cv2.threshold(v_clahe, int(thr_val), 255, cv2.THRESH_BINARY)

    # combine masks (OR)
    mask_combined = cv2.bitwise_or(mask_hsv, mask_rdiff)
    mask_combined = cv2.bitwise_or(mask_combined, mask_top)

    # morphology: open then close to remove speckle and fill core
    kernel = np.ones((3,3), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask_combined
    
def find_laser_centroid(mask, img_bgr=None, min_area=None):
    """
    Improved centroid finder:
      - uses connectedComponentsWithStats to get area and mean intensity
      - prefers the blob with largest mean red intensity (or largest area if tie)
      - returns (cx, cy, best_contour) or (None, None, None)
    If img_bgr provided, compute mean red channel for ranking.
    """
    if min_area is None:
        min_area = getattr(Config, "LASER_MIN_AREA", 2)

    # find connected components (faster and robust)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, None, None

    best_idx = -1
    best_score = -1.0

    # if BGR provided, get red intensity per component
    red_channel = None
    if img_bgr is not None:
        red_channel = img_bgr[:, :, 2]

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cx, cy = centroids[i]
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # compute circularity (optional)
        # approximate perimeter by bounding box
        if area > 0:
            circularity = 4 * math.pi * area / ( (w+h)**2 + 1e-6 )
        else:
            circularity = 0

        # score: prioritize red intensity mean, then area, then circularity
        if red_channel is not None:
            mask_comp = (labels[y:y+h, x:x+w] == i)
            mean_red = float(np.mean(red_channel[y:y+h, x:x+w][mask_comp])) if np.any(mask_comp) else 0.0
        else:
            mean_red = float(area)

        # combine metrics into score (weights tunable)
        score = mean_red * 2.0 + area * 0.5 + circularity * 5.0

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        return None, None, None

    # build contour of best component for drawing (approx)
    mask_best = (labels == best_idx).astype(np.uint8) * 255
    contours = cv2.findContours(mask_best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) == 0:
        return None, None, None

    best_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(best_contour)
    if M.get("m00", 0) != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy, best_contour

    return None, None, None
    
# =============================================================================
# FPS counter
# =============================================================================
class FPSCounter:
    """ Exponential-smoothed FPS estimator. Call update() once per frame. """
    def __init__(self, smoothing=0.9):
        self.smoothing = float(smoothing)
        self.last_time = None
        self.fps = 0.0

    def update(self):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return 0.0
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0:
            return self.fps
        inst = 1.0 / dt
        if self.fps == 0.0:
            self.fps = inst
        else:
            alpha = 1.0 - self.smoothing
            self.fps = self.smoothing * self.fps + alpha * inst
        return self.fps
 
# =============================================================================
# Display manager (drawing helpers)
# =============================================================================

class DisplayManager:
    def __init__(self):
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_PURPLE = (255, 0, 255)
        self.COLOR_ORANGE = (0, 165, 255)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(self, img, is_locked, time_left, result_code, edge_lbl, vert_lbl):
        """ Draw top-left status text """
        if not is_locked:
            text = f"Locking in {time_left:.1f}s"
            cv2.putText(img, text, (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)
        else:
            cv2.putText(img, "LOCKED", (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)

            # result text
            if result_code == '0':
                status_text = "NO LASER"
                color = self.COLOR_RED
            elif result_code == '1':
                status_text = "INSIDE"
                color = self.COLOR_GREEN
            elif result_code == '2':
                status_text = "BETWEEN"
                color = self.COLOR_ORANGE
            else:
                status_text = "OUTSIDE"
                color = self.COLOR_ORANGE

            cv2.putText(img, status_text, (20, 30), self.FONT, 0.7, color, 2)

            # show extra info
            if vert_lbl != '0':
                cv2.putText(img, f"At vertex: {vert_lbl}", (20, 90), self.FONT, 0.7, (155, 65, 200), 2)
            else:
                cv2.putText(img, f"Closest edge: {edge_lbl}", (20, 90), self.FONT, 0.7, (155, 65, 200), 2)

    def draw_rectangles(self, img, outer, inner, mid_warped, inv_matrix):
        """ Draw outer, inner and mid rectangles """
        if outer is not None:
            cv2.drawContours(img, [outer], -1, self.COLOR_BLUE, 2)
        if inner is not None:
            cv2.drawContours(img, [inner], -1, self.COLOR_GREEN, 2)

        if mid_warped is not None and inv_matrix is not None:
            try:
                mid_orig = cv2.perspectiveTransform(mid_warped.reshape(1,4,2), inv_matrix)[0]
                mid_orig_int = mid_orig.astype(np.int32)
                cv2.drawContours(img, [mid_orig_int], -1, self.COLOR_PURPLE, 2)
            except:
                pass

    def draw_detailed_labels(self, img, mid_warped, inv_matrix):
        """ Draw edge labels, lengths and vertex labels """
        if mid_warped is None or inv_matrix is None:
            return

        edge_display_map = {0: '1', 1: '2', 2: '3', 3: '4'}
        vertex_display_map = {0: '1', 1: '2', 2: '3', 3: '4'}
        wm = mid_warped
        center = np.mean(wm, axis=0)

        for i in range(4):
            p1 = wm[i]
            p2 = wm[(i+1)%4]
            midpoint = (p1 + p2) / 2.0
            length = np.linalg.norm(p1 - p2)

            dir_vec = midpoint - center
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                norm = 1
            dir_unit = dir_vec / norm
            offset_dist = 30

            txt_pos_w = midpoint + dir_unit * offset_dist
            len_pos_w = midpoint + dir_unit * (offset_dist + 15)

            pts_src = np.array([txt_pos_w, len_pos_w, p1], dtype=np.float32).reshape(-1, 1, 2)
            pts_dst = cv2.perspectiveTransform(pts_src, inv_matrix)

            txt_xy = (int(pts_dst[0][0][0]), int(pts_dst[0][0][1]))
            len_xy = (int(pts_dst[1][0][0]), int(pts_dst[1][0][1]))
            vert_xy = (int(pts_dst[2][0][0]), int(pts_dst[2][0][1]))

            lbl = edge_display_map.get(i, str(i))
            cv2.putText(img, lbl, txt_xy, self.FONT, 0.9, self.COLOR_ORANGE, 2)
            cv2.putText(img, f"{int(length)}", len_xy, self.FONT, 0.6, self.COLOR_CYAN, 1)
            v_lbl = vertex_display_map.get(i, str(i))
            cv2.circle(img, vert_xy, 5, self.COLOR_YELLOW, -1)
            cv2.putText(img, v_lbl, (vert_xy[0]-10, vert_xy[1]-10), self.FONT, 0.7, self.COLOR_GREEN, 2)

    def draw_laser(self, img, pt):
        if pt is not None:
            cv2.circle(img, pt, 5, self.COLOR_RED, -1)
            x, y = pt
            cv2.line(img, (x-10, y), (x+10, y), self.COLOR_RED, 1)
            cv2.line(img, (x, y-10), (x, y+10), self.COLOR_RED, 1)

# =============================================================================
# Serial manager (sampling & sending)
# =============================================================================

class SerialManager:
    def __init__(self):
        self.ser = None
        self.dist_samples = []
        self.last_sample_time = time.time()
        self.last_send_time = time.time()

        try:
            self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=1)
            time.sleep(2)
            print("Serial Connected.")
        except Exception as e:
            print(f"Serial Failed: {e}")

    def update(self, current_dist, res, edge, vert, curr_time):
        """
        curr_time: current system time
        current_dist: current measured distance (None means no laser)
        """
        # sampling logic
        if curr_time - self.last_sample_time >= Config.SAMPLE_INTERVAL:
            if current_dist is not None:
                self.dist_samples.append(current_dist)
            self.last_sample_time = curr_time

        # sending logic
        if curr_time - self.last_send_time >= Config.SEND_INTERVAL:
            final_dist_int = 0
            if len(self.dist_samples) > 0:
                avg_dist = sum(self.dist_samples) / len(self.dist_samples)
                final_dist_int = int(round(avg_dist))
                final_dist_int = min(999, max(0, final_dist_int))
            else:
                final_dist_int = 0

            send_str = f"{res}{edge}{vert}{final_dist_int}A"
            print(f"{send_str}")

            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(send_str.encode())
                except Exception as e:
                    print(f"Serial Error: {e}")

            self.dist_samples = []
            self.last_send_time = curr_time

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

# =============================================================================
# Main
# =============================================================================

def main():
    cap = open_camera(Config.CAMERA_ID, Config.FRAME_WIDTH, Config.FRAME_HEIGHT, retries=8, wait_s=1.0)
    if cap is None:
        print("Camera Error - cannot open after retries")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Camera Error")
        return

    serial_mgr = SerialManager()
    display_mgr = DisplayManager()
    fps_counter = FPSCounter(smoothing=0.9)   # create FPS counter

    start_time = time.time()
    rectangles_locked = False
    perspective_mtx = None
    inverse_mtx = None
    warped_mid_poly = None
    draw_outer_cnt = None
    draw_inner_cnt = None

    # stability queue for (outer, inner, t)
    stable_q = deque(maxlen=Config.STABLE_FRAMES)
    
    while True:
        success, raw_img = cap.read()
        if not success:
            print("Frame read fail")
            break

        h, w = raw_img.shape[:2]
        
        #print(f"h: {h}")
        #print(f"w: {w}")
        
        img = raw_img[int(h*Config.CROP_Y_START):int(h*Config.CROP_Y_END),
                      int(w*Config.CROP_X_START):int(w*Config.CROP_X_END)]
        img = cv2.resize(img, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)

            # --- Laser extraction ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        laser_mask = create_combined_laser_mask(img)
        laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        # --- A: remove laser pixels before edge detection ---
        clean = remove_laser_pixels(img, laser_mask, method=Config.LASER_REMOVE_METHOD, dilate_px=Config.LASER_DILATE_PX)

        # --- B: edges (auto or fixed) ---
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        edges = auto_canny(gray, sigma=Config.SIGMA)

            # --- Rectangle lock (first few seconds) ---
        curr_time = time.time()

        if not rectangles_locked and (curr_time - start_time) < Config.LOCK_TIME_WINDOW:
            # ---------- DEBUG-ROBUST LOCK (replace your lock block with this) ----------
            # expects: img, clean, edges, Config, stable_q, order_points, is_rect_contained, stable_lock

            curr_time = time.time()

            # robust findContours across OpenCV versions
            cnts_res = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # contours are usually at index -2
            cnts = cnts_res[-2] if len(cnts_res) >= 2 else []
            if cnts is None:
                cnts = []

            rect_candidates = []
            for c in cnts:
                peri = cv2.arcLength(c, True)
                # try a tighter epsilon first (preserve quads), fallback to 0.02
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                if len(approx) != 4:
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    area = abs(cv2.contourArea(approx))
                    # compute dynamic threshold
                    img_area = img.shape[0] * img.shape[1]
                    RECT_MIN_AREA = max(Config.RECT_MIN_AREA_SET, int(img_area * Config.PERCENT_OF_IMAGE))
                    # ??????????wtf??
                    if area > RECT_MIN_AREA:
                        pts = approx.reshape(4,2).astype(np.float32)
                        rect = order_points(pts)
                        rect_candidates.append((rect, area, approx))

            # debug print: how many candidates & top areas
            areas = [int(a) for (_, a, _) in rect_candidates]
            print("[LOCK DEBUG] candidates:", len(rect_candidates), "areas:", areas[:6])

            locked_this_frame = False
            if len(rect_candidates) >= 2:
                # sort by area desc
                rect_candidates.sort(key=lambda x: x[1], reverse=True)

                # try several pairing strategies (robust against swapped order / ties)
                tried_pairs = []
                accepted_pair = None
                # primary: largest outer with smallest inner
                pairs_to_try = [(0, -1)] + [(0, i) for i in range(1, min(4, len(rect_candidates)))] + [(i, i+1) for i in range(len(rect_candidates)-1)]
                for oi, ii in pairs_to_try:
                    try:
                        outer = rect_candidates[oi][0]
                        inner = rect_candidates[ii][0]
                    except Exception:
                        continue
                    key = (oi, ii)
                    if key in tried_pairs: continue
                    tried_pairs.append(key)

                    # quick visual check: inner centroid inside outer?
                    c_in = np.mean(inner.reshape(-1,2), axis=0)
                    inside = cv2.pointPolygonTest(outer.astype(np.float32), (float(c_in[0]), float(c_in[1])), False)
                    if inside < 0:
                        # inner centroid outside outer -> reject
                        continue

                    # geometric validation (less strict at first - allow broader ratio)
                    area_out = abs(cv2.contourArea(outer))
                    area_in = abs(cv2.contourArea(inner))
                    if area_out <= 0: continue
                    ratio = area_in / area_out
                    # accept wider ratio range initially, but reject near 1.0 (coincident)
                    if ratio > 0.995:
                        continue

                    # vertex spacing check (avoid nearly identical polygons)
                    vclose = sum(1 for k in range(4) if np.linalg.norm(outer[k] - inner[k]) < 5.0)
                    if vclose >= 3:
                        # likely coincident -> skip
                        continue

                    # passed simple tests -> push into stability queue and evaluate
                    stable_q.append((outer.copy(), inner.copy(), curr_time))
                    if stable_lock(stable_q,Config.STABLE_MAX_DISP,Config.STABLE_MIN_COUNT):
                        accepted_pair = (outer, inner)
                        locked_this_frame = True
                        break

                if accepted_pair is not None:
                    outer, inner = accepted_pair
                    # finalize transform
                    try:
                        dst_pts = np.float32([[0,0],[400,0],[400,500],[0,500]])
                        perspective_mtx = cv2.getPerspectiveTransform(outer.astype(np.float32), dst_pts)
                        inverse_mtx = cv2.getPerspectiveTransform(dst_pts, outer.astype(np.float32))
                        in_warp = cv2.perspectiveTransform(inner.reshape(1,4,2), perspective_mtx)[0]
                        # reorder inner to match dst correspondence
                        in_reorder = np.zeros_like(in_warp)
                        used = set()
                        for i in range(4):
                            dists = [np.linalg.norm(in_warp[j]-dst_pts[i]) for j in range(4)]
                            for u in used: dists[u] = 9e9
                            best = int(np.argmin(dists))
                            in_reorder[i] = in_warp[best]
                            used.add(best)
                        warped_mid_poly = (dst_pts + in_reorder) / 2.0
                        draw_outer_cnt = outer.reshape(4,1,2).astype(np.int32)
                        draw_inner_cnt = inner.reshape(4,1,2).astype(np.int32)
                        rectangles_locked = True
                        print("[LOCK] locked ok - outer_area:", int(area_out), "inner_area:", int(area_in))
                    except Exception as e:
                        print("[LOCK ERROR] finalize failed:", e)
                        rectangles_locked = False

            # if not locked, show debug overlays to help
            if not rectangles_locked:
                dbg = img.copy()
                for (r,a,c) in rect_candidates:
                    cv2.polylines(dbg, [c.astype(np.int32)], True, (0,180,255), 2)
                    cx,cy = np.mean(r, axis=0).astype(int)
                    cv2.putText(dbg, f"A:{int(a)}", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),1)
                cv2.imshow("lock_debug_candidates", dbg)
                # additionally print a short hint if no candidates
                if len(rect_candidates) == 0:
                    print("[LOCK HINT] no quad candidates above RECT_MIN_AREA =", RECT_MIN_AREA,
                          "-> try lowering RECT_MIN_AREA_SET or PERCENT_OF_IMAGE, or reduce LASER_DILATE_PX")
            # ---------- end replacement ----------

            # default outputs
        res, edge, vert = '0', '0', '0'
        dist = None
        
        lx, ly, laser_cnt = find_laser_centroid(laser_mask, img, min_area=Config.LASER_MIN_AREA)
        laser_pt = (lx, ly) if lx is not None else None

        if rectangles_locked and laser_pt:
            pt_w = cv2.perspectiveTransform(np.array([[[lx, ly]]], dtype=np.float32), perspective_mtx)[0][0]

            dists = [point_segment_distance(warped_mid_poly[i], warped_mid_poly[(i+1)%4], pt_w) for i in range(4)]
            min_d = min(dists)
            dist = float(min_d)

            if min_d <= 4.5:
                res = '2'
            elif cv2.pointPolygonTest(warped_mid_poly, (float(pt_w[0]), float(pt_w[1])), False) >= 0:
                res = '1'
            else:
                res = '3'

            idx = np.argmin(dists)
            edge_map = {0: '1', 1: '2', 2: '3', 3: '4'}
            edge = edge_map.get(idx, '0')

            v_dists = [np.linalg.norm(pt_w - v) for v in warped_mid_poly]
            if min(v_dists) < 10 and min(v_dists) < min_d * 2:
                vert_map = {0: '1', 1: '2', 2: '3', 3: '4'}
                vert = vert_map.get(np.argmin(v_dists), '0')
        else:
                # keep defaults when no laser
            pass

            # update serial manager
        serial_mgr.update(dist, res, edge, vert, curr_time)

            # display drawing
        display_mgr.draw_status(img, rectangles_locked, Config.LOCK_TIME_WINDOW - (curr_time - start_time), res, edge, vert)
        display_mgr.draw_laser(img, laser_pt)
        if rectangles_locked:
            display_mgr.draw_rectangles(img, draw_outer_cnt, draw_inner_cnt, warped_mid_poly, inverse_mtx)
            display_mgr.draw_detailed_labels(img, warped_mid_poly, inverse_mtx)

            # --- FPS: update and draw on top-right corner ---
        cur_fps = fps_counter.update()
        fps_text = f"FPS:{cur_fps:.1f}"
        (text_w, text_h), baseline = cv2.getTextSize(fps_text, display_mgr.FONT, 0.7, 2)
        x_pos = img.shape[1] - text_w - 10
        y_pos = 30  # top margin
        cv2.putText(img, fps_text, (x_pos, y_pos), display_mgr.FONT, 0.7, display_mgr.COLOR_CYAN, 2)

        #cv2.imshow("edges", edges)
        #cv2.imshow("mask_combined", laser_mask)
        cv2.imshow("Laser Detector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #finally:
    cap.release()
    serial_mgr.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
