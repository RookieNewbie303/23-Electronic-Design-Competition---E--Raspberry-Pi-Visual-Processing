# -*- coding: utf-8 -*-

"""
Integrated laser-robust rectangle lock + sampling + FPS + serial sender.
Modified based on user request:
 - C: Simplified to ONLY check area ratio.
 - D: Removed stable_lock (immediate lock).
 - Optimization: Rectangle logic (Canny/Inpaint) only runs until locked.
"""
import cv2
import numpy as np
import serial
import time
import math
#from collections import deque

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
    RECT_MIN_AREA_SET = 1500         
    PERCENT_OF_IMAGE = 0.001
    CANNY_TH1 = 10
    CANNY_TH2 = 30
    LOCK_TIME_WINDOW = 5.0

    # laser detection
    LASER_MIN_AREA = 1
    RED_DIFF_THRESH = 50    
    VALUE_CLAHE_CLIP = 2.0  
    TOP_PERCENT = 0.0002
    
    # serial
    SERIAL_PORT = '/dev/ttyAMA0'
    BAUD_RATE = 9600
    SEND_INTERVAL = 0.050
    SAMPLE_COUNT = 1
    SAMPLE_INTERVAL = SEND_INTERVAL / float(SAMPLE_COUNT)

    # laser removal options (A)
    LASER_REMOVE_METHOD = 'median'   # 'inpaint' or 'median'
    LASER_DILATE_PX = 5             

    # auto canny (B)
    SIGMA = 0.33

    # containment geometry (C) - Only Ratio kept
    MIN_INNER_OUTER_RATIO = 0.6
    MAX_INNER_OUTER_RATIO = 0.9
    
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

def compute_numbering_offset(mid_warped, inverse_mtx, angle_thresh_deg=12.0):
    """
    Decide which original edge index (0..3) should be labeled '1'.

    Strategy:
      1) Map canonical dst corners back to original image coords using inverse_mtx,
         producing outer_orig (4,2) in SAME order as dst (i.e. corresponds to warped_mid_poly order).
      2) Use compute_mean_edge_angle_correct on outer_orig to get mean long-edge angle to horizontal.
      3) If mean_angle <= angle_thresh_deg (nearly horizontal long edges),
         pick the top edge (smallest edge-midpoint y) as start index.
      4) Otherwise (rotated case), pick the edge whose midpoint maximizes (x - y)
         to prefer the right-top edge as edge 1 (your earlier rule).

    Returns:
      start_idx (int in 0..3)
    """
    try:
        if mid_warped is None or inverse_mtx is None:
            return 0

        # canonical dst corner coordinates used elsewhere in your pipeline
        dst_pts = np.float32([[0,0], [400,0], [400,500], [0,500]]).reshape(-1,1,2)

        # map canonical dst corners back to original image using inverse_mtx
        outer_orig = cv2.perspectiveTransform(dst_pts, inverse_mtx).reshape(-1,2)  # shape (4,2)

        # compute mean angle using the correct helper
        mean_angle, raw_angles, lengths = compute_mean_edge_angle_correct(outer_orig)

        # compute midpoints (in original-image coords) for selection rules
        mids_orig = []
        for i in range(4):
            p1 = outer_orig[i]
            p2 = outer_orig[(i+1) % 4]
            mids_orig.append(((p1 + p2) / 2.0))
        mids_orig = np.array(mids_orig)

        # If rectangle is nearly un-rotated (long edges horizontal), choose top edge
        if mean_angle <= angle_thresh_deg:
            ys = [float(p[1]) for p in mids_orig]
            start_idx = int(np.argmin(ys))  # top edge has smallest midpoint y
            return start_idx

        # otherwise, use right-top preference: maximize (x - y)
        scores = [float(p[0]) - float(p[1]) for p in mids_orig]
        start_idx = int(np.argmax(scores))
        return start_idx

    except Exception as e:
        # safe fallback
        #print("[compute_numbering_offset] exception:", e)
        return 0

def compute_mean_edge_angle_correct(outer_pts):
    """
    outer_pts: np.array shape (4,2) in ORIGINAL image coordinates, ordered by order_points -> [TL, TR, BR, BL]
    Returns:
      mean_angle_deg  - mean angle (degrees) of the LONG edges to horizontal (0..90)
      raw_angles      - list of raw angles (degrees, -180..180) for edges [0..3]
      lengths         - list of lengths for edges [0..3]
    """
    pts = np.asarray(outer_pts, dtype=np.float64).reshape(4,2)
    # compute edge vectors p_i -> p_{i+1}
    edge_vecs = [pts[(i+1)%4] - pts[i] for i in range(4)]
    lengths = [float(np.linalg.norm(v)) for v in edge_vecs]

    # raw atan2 angles in degrees (-180..180)
    raw_angles = [math.degrees(math.atan2(v[1], v[0])) for v in edge_vecs]

    # convert to angle-to-horizontal in [0..90]
    ang_to_h = []
    for a in raw_angles:
        a_abs = abs(a)
        if a_abs > 90:
            a_abs = 180.0 - a_abs
        ang_to_h.append(a_abs)

    # choose "long edges" for orientation (>= 0.8 * max_length)
    maxL = max(lengths) if len(lengths) > 0 else 0.0
    if maxL <= 0:
        mean_angle = 0.0
    else:
        long_mask = [L >= 0.8 * maxL for L in lengths]
        long_angles = [ang_to_h[i] for i in range(4) if long_mask[i]]
        if len(long_angles) == 0:
            mean_angle = float(np.mean(ang_to_h))
        else:
            mean_angle = float(np.mean(long_angles))

    # debug printing (remove or comment if noisy)
    """
    print("---- angle debug ----")
    for i in range(4):
        print(f"edge {i}: raw_ang={raw_angles[i]:.3f}, to_h={ang_to_h[i]:.3f}, len={lengths[i]:.1f}, long={lengths[i]>=0.8*maxL}")
    print(f"mean_angle (long edges) = {mean_angle:.3f}")
    print("---------------------")
    """

    return mean_angle, raw_angles, lengths
    
def compute_mean_angle_both(mid_warped, inverse_mtx):
    """
    Compute mean edge angle by mapping both canonical outer corners and
    derived inner corners back to original image coordinates, computing
    per-rectangle mean angle (via compute_mean_edge_angle_correct) and
    returning their average.
    mid_warped: (4,2) in dst space (warped mid poly)
    inverse_mtx: 3x3 matrix mapping dst->orig
    """
    try:
        if mid_warped is None or inverse_mtx is None:
            return 0.0

        dst_pts = np.float32([[0,0], [400,0], [400,500], [0,500]]).reshape(-1,2)  # (4,2)
        dst_pts_t = dst_pts.reshape(-1,1,2)

        # outer_orig: map canonical dst corners back to original coords
        outer_orig = cv2.perspectiveTransform(dst_pts_t, inverse_mtx).reshape(-1,2)

        # reconstruct in_reorder (inner corners in dst) from mid_warped:
        # mid_warped = (dst_pts + in_reorder) / 2 --> in_reorder = 2*mid_warped - dst_pts
        mid_arr = np.asarray(mid_warped, dtype=np.float32).reshape(4,2)
        in_reorder_dst = (2.0 * mid_arr) - dst_pts  # (4,2)
        in_reorder_t = in_reorder_dst.reshape(-1,1,2)

        # inner_orig: map inner corners (dst) back to original coords
        inner_orig = cv2.perspectiveTransform(in_reorder_t, inverse_mtx).reshape(-1,2)

        # compute mean angles for both rectangles
        mean_outer, _, _ = compute_mean_edge_angle_correct(outer_orig)
        mean_inner, _, _ = compute_mean_edge_angle_correct(inner_orig)

        # average (simple mean)
        mean_both = float((mean_outer + mean_inner) / 2.0)
        return mean_both

    except Exception as e:
        # fallback
        #print("[compute_mean_angle_both] exception:", e)
        return 0.0
        
# =============================================================================
# A. Remove laser pixels (inpaint or median)
# =============================================================================
def remove_laser_pixels(img_bgr, laser_mask, method='inpaint', dilate_px=5):
    """
    Remove/replace bright laser pixels before edge detection.
    """
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
# C. geometry checks (Simplified: Only Area Ratio)
# =============================================================================
def rect_area(pts):
    return abs(cv2.contourArea(pts))

def is_rect_contained(outer_pts, inner_pts, min_ratio=Config.MIN_INNER_OUTER_RATIO, max_ratio=Config.MAX_INNER_OUTER_RATIO):
    """
    Only checks if the area ratio is within bounds.
    Removes: Centroid check, PointPolygonTest, Vertex tolerance.
    """
    A_out = rect_area(outer_pts)
    A_in = rect_area(inner_pts)
    
    if A_out <= 0 or A_in <= 0:
        return False
        
    ratio = A_in / A_out
    
    # Only return True if ratio is valid
    if min_ratio <= ratio <= max_ratio:
        return True
        
    return False

# Function D (stable_lock) has been removed.

# =============================================================================
# Laser centroid detection
# =============================================================================

def create_combined_laser_mask(img_bgr):
    """
    Return a binary mask of candidate laser pixels.
    """
    # params (tune in Config instead if you prefer)
    red_s_min = 30      
    red_v_min = 30      
    red_diff_thresh = getattr(Config, "RED_DIFF_THRESH", 40)   
    top_percent = getattr(Config, "TOP_PERCENT", 0.0005)        
    clahe_clip = getattr(Config, "VALUE_CLAHE_CLIP", 2.0)

    # 1) HSV red mask 
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, red_s_min, red_v_min])
    upper1 = np.array([12, 255, 255])
    lower2 = np.array([150, red_s_min, red_v_min])
    upper2 = np.array([180, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # 2) Red-difference mask 
    b, g, r = cv2.split(img_bgr)
    max_gb = np.maximum(g, b).astype(np.int16)
    r_int = r.astype(np.int16)
    red_diff = (r_int - max_gb).astype(np.int16)
    _, mask_rdiff = cv2.threshold(red_diff, red_diff_thresh, 255, cv2.THRESH_BINARY)
    mask_rdiff = mask_rdiff.astype(np.uint8)

    # 3) CLAHE on V channel 
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)

    flat = v_clahe.flatten()
    if flat.size > 0:
        kth = int(max(1, flat.size * (1.0 - top_percent)))
        thr_val = np.partition(flat, kth)[kth]
    else:
        thr_val = 255
    _, mask_top = cv2.threshold(v_clahe, int(thr_val), 255, cv2.THRESH_BINARY)

    # combine masks (OR)
    mask_combined = cv2.bitwise_or(mask_hsv, mask_rdiff)
    mask_combined = cv2.bitwise_or(mask_combined, mask_top)

    # morphology
    kernel = np.ones((3,3), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask_combined
    
def find_laser_centroid(mask, img_bgr=None, min_area=None):
    if min_area is None:
        min_area = getattr(Config, "LASER_MIN_AREA", 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, None, None

    best_idx = -1
    best_score = -1.0
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

        if area > 0:
            circularity = 4 * math.pi * area / ( (w+h)**2 + 1e-6 )
        else:
            circularity = 0

        if red_channel is not None:
            mask_comp = (labels[y:y+h, x:x+w] == i)
            mean_red = float(np.mean(red_channel[y:y+h, x:x+w][mask_comp])) if np.any(mask_comp) else 0.0
        else:
            mean_red = float(area)

        score = mean_red * 2.0 + area * 0.5 + circularity * 5.0

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        return None, None, None

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
            text = f"Scanning... {time_left:.1f}s"
            cv2.putText(img, text, (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)
        else:
            cv2.putText(img, "LOCKED", (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)

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
    """
    def draw_detailed_labels(self, img, mid_warped, inv_matrix):
        # Draw edge labels, lengths and vertex labels 
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
    """
    def draw_detailed_labels(self, img, mid_warped, inv_matrix, numbering_offset=None):
        """ Draw edge labels and vertex labels using numbering_offset.
        numbering_offset: int index (0..3) indicating which original edge index becomes label '1'.
        If numbering_offset is None, defaults to 0 (legacy behavior).
        """
        if mid_warped is None or inv_matrix is None:
            return

        # determine numbering offset if not provided
        if numbering_offset is None:
            numbering_offset = compute_numbering_offset(mid_warped, inv_matrix)

        # wm is in warped/dst coordinates
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

            # compute label numbers using numbering_offset: clockwise numbering starting at offset
            edge_label = ((i - numbering_offset) % 4) + 1  # 1..4
            # the vertex corresponding to wm[i] will be labeled similarly: vertex index i -> label ((i - offset)%4)+1
            vert_label = ((i - numbering_offset) % 4) + 1

            cv2.putText(img, str(edge_label), txt_xy, self.FONT, 0.9, self.COLOR_ORANGE, 2)
            cv2.putText(img, f"{int(length)}", len_xy, self.FONT, 0.6, self.COLOR_CYAN, 1)
            cv2.circle(img, vert_xy, 5, self.COLOR_YELLOW, -1)
            cv2.putText(img, str(vert_label), (vert_xy[0]-10, vert_xy[1]-10), self.FONT, 0.7, self.COLOR_GREEN, 2)    
    
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
    fps_counter = FPSCounter(smoothing=0.9)   

    start_time = time.time()
    rectangles_locked = False
    perspective_mtx = None
    inverse_mtx = None
    warped_mid_poly = None
    draw_outer_cnt = None
    draw_inner_cnt = None

    numbering_offset = 0  # default until locked
    angle_sent = False

    while True:
        success, raw_img = cap.read()
        if not success:
            print("Frame read fail")
            break

        h, w = raw_img.shape[:2]
        
        img = raw_img[int(h*Config.CROP_Y_START):int(h*Config.CROP_Y_END),
                      int(w*Config.CROP_X_START):int(w*Config.CROP_X_END)]
        img = cv2.resize(img, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)
        curr_time = time.time()

        # --- Laser extraction (Always running) ---
        laser_mask = create_combined_laser_mask(img)
        # Optional small cleanup for laser mask
        laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

        # --- RECTANGLE LOCK LOGIC (Only runs initially) ---
        if not rectangles_locked and (curr_time - start_time) < Config.LOCK_TIME_WINDOW:
            
            # 1. Prepare image for edge detection (Moved inside for optimization)
            # Remove laser pixels from image to avoid false edges
            clean = remove_laser_pixels(img, laser_mask, method=Config.LASER_REMOVE_METHOD, dilate_px=Config.LASER_DILATE_PX)
            gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
            edges = auto_canny(gray, sigma=Config.SIGMA)

            # 2. Find contours
            cnts_res = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts_res[-2] if len(cnts_res) >= 2 else []
            if cnts is None: cnts = []

            # 3. Filter candidates
            rect_candidates = []
            img_area = img.shape[0] * img.shape[1]
            RECT_MIN_AREA = max(Config.RECT_MIN_AREA_SET, int(img_area * Config.PERCENT_OF_IMAGE))

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                if len(approx) != 4:
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                if len(approx) == 4:
                    area = abs(cv2.contourArea(approx))
                    if area > RECT_MIN_AREA:
                        pts = approx.reshape(4,2).astype(np.float32)
                        rect = order_points(pts)
                        rect_candidates.append((rect, area, approx))

            # 4. Try pairing (Sort by area desc)
            if len(rect_candidates) >= 2:
                rect_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Pair strategies
                pairs_to_try = [(0, -1)] + [(0, i) for i in range(1, min(4, len(rect_candidates)))] + [(i, i+1) for i in range(len(rect_candidates)-1)]
                
                for oi, ii in pairs_to_try:
                    try:
                        outer = rect_candidates[oi][0]
                        #mean_angle = compute_mean_edge_angle_correct(outer)
                        inner = rect_candidates[ii][0]
                        #mean_angle = compute_mean_edge_angle_correct(inner)
                    except Exception:
                        continue
                    
                    # Modified C: Only Ratio Check
                    if is_rect_contained(outer, inner):
                        # --- Immediate Lock (Removed D) ---
                        try:
                            # Finalize transform
                            dst_pts = np.float32([[0,0],[400,0],[400,500],[0,500]])
                            perspective_mtx = cv2.getPerspectiveTransform(outer.astype(np.float32), dst_pts)
                            inverse_mtx = cv2.getPerspectiveTransform(dst_pts, outer.astype(np.float32))
                            
                            in_warp = cv2.perspectiveTransform(inner.reshape(1,4,2), perspective_mtx)[0]
                            # Reorder inner to match dst correspondence
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
                            # --- compute numbering offset once and store ---
                            numbering_offset = compute_numbering_offset(warped_mid_poly, inverse_mtx)
                            
                            if (not angle_sent):
                                try:
                                    mean_angle = compute_mean_angle_both(warped_mid_poly, inverse_mtx)
                                    angle_str = f"B{mean_angle:.1f}C"
                                    print(f"[ANGLE SEND] preparing -> {angle_str}")

                                    ser = getattr(serial_mgr, "ser", None)
                                    if ser is not None and getattr(ser, "is_open", False):
                                        try:
                                          
                                            n = ser.write(angle_str.encode('ascii'))
                                            try:
                                                ser.flush()   
                                            except Exception:
                                                pass
                                            print(f"[ANGLE SEND] wrote {n} bytes")
                                            angle_sent = True   
                                        except Exception as e:
                                            print(f"[ANGLE SEND ERROR] write failed: {e}")
                                    else:
                                        print("[ANGLE SEND] serial not available or not open")
                                except Exception as e:
                                    print(f"[ANGLE SEND ERROR] compute/send failed: {e}")

                        except Exception as e:
                            print(f"[LOCK ERROR] Calculation failed: {e}")
                
            # If still not locked, optional debug draw
            if not rectangles_locked:
                for (r,a,c) in rect_candidates:
                    cv2.polylines(img, [c.astype(np.int32)], True, (0,100,100), 1)

        # --- Main Measurement Loop ---
        res, edge, vert = '0', '0', '0'
        dist = None
        
        lx, ly, laser_cnt = find_laser_centroid(laser_mask, img, min_area=Config.LASER_MIN_AREA)
        laser_pt = (lx, ly) if lx is not None else None

        if rectangles_locked and laser_pt:
            # Map laser point to warped space
            pt_w = cv2.perspectiveTransform(np.array([[[lx, ly]]], dtype=np.float32), perspective_mtx)[0][0]

            dists = [point_segment_distance(warped_mid_poly[i], warped_mid_poly[(i+1)%4], pt_w) for i in range(4)]
            min_d = min(dists)
            dist = float(min_d)

            # Logic: 2=Between, 1=Inside, 3=Outside
            if min_d <= 3:
                res = '2'
            elif cv2.pointPolygonTest(warped_mid_poly, (float(pt_w[0]), float(pt_w[1])), False) >= 0:
                res = '1'
            else:
                res = '3'

            """
            idx = np.argmin(dists)
            edge_map = {0: '1', 1: '2', 2: '3', 3: '4'}
            edge = edge_map.get(idx, '0')
            
            v_dists = [np.linalg.norm(pt_w - v) for v in warped_mid_poly]
            if min(v_dists) < 30 and min(v_dists) < min_d * 10:
                vert_map = {0: '1', 1: '2', 2: '3', 3: '4'}
                vert = vert_map.get(np.argmin(v_dists), '0')
            """
                            
            idx = int(np.argmin(dists))  # original index in warped_mid_poly order
            edge_label = ((idx - numbering_offset) % 4) + 1
            edge = str(edge_label)
            
            v_dists = [np.linalg.norm(pt_w - v) for v in warped_mid_poly]
            if min(v_dists) < 20 and min(v_dists) < min_d * 10:
                v_idx = int(np.argmin(v_dists))
                vert_label = ((v_idx - numbering_offset) % 4) + 1
                vert = str(vert_label)

        # update serial manager
        serial_mgr.update(dist, res, edge, vert, curr_time)

        # display drawing
        display_mgr.draw_status(img, rectangles_locked, Config.LOCK_TIME_WINDOW - (curr_time - start_time), res, edge, vert)
        display_mgr.draw_laser(img, laser_pt)
        if rectangles_locked:
            display_mgr.draw_rectangles(img, draw_outer_cnt, draw_inner_cnt, warped_mid_poly, inverse_mtx)
            #display_mgr.draw_detailed_labels(img, warped_mid_poly, inverse_mtx)
            display_mgr.draw_detailed_labels(img, warped_mid_poly, inverse_mtx, numbering_offset)

        # --- FPS ---
        cur_fps = fps_counter.update()
        fps_text = f"FPS:{cur_fps:.1f}"
        (text_w, text_h), baseline = cv2.getTextSize(fps_text, display_mgr.FONT, 0.7, 2)
        x_pos = img.shape[1] - text_w - 10
        y_pos = 30 
        cv2.putText(img, fps_text, (x_pos, y_pos), display_mgr.FONT, 0.7, display_mgr.COLOR_CYAN, 2)

        #cv2.imshow("edges", edges)
        #cv2.imshow("mask_combined", laser_mask)
        cv2.imshow("Laser Detector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    serial_mgr.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
