# -*- coding: gbk -*-
import cv2
import numpy as np
import serial
import time
import math

# =============================================================================
# 1. 配置与参数模块 (Configuration)
# =============================================================================

class Config:
    # --- 摄像头设置 ---
    CAMERA_ID = 0
    FRAME_WIDTH = 1080
    FRAME_HEIGHT = 720
    
    # --- 图像裁剪区域 ---
    CROP_Y_START = 0.1
    CROP_Y_END = 0.6
    CROP_X_START = 0.30
    CROP_X_END = 0.70
    ZOOM_FACTOR = 1.2

    # --- 矩形检测参数 (1米外检测建议) ---
    RECT_MIN_AREA = 1000 
    CANNY_TH1 = 50
    CANNY_TH2 = 150
    LOCK_TIME_WINDOW = 3.0

    # --- 激光检测参数 ---
    # 保持较低的值，以适应层级检测。原代码最小面积为 3，此处采用 2。
    LASER_MIN_AREA = 2 

    # --- 串口与采样设置 ---
    SERIAL_PORT = '/dev/ttyAMA0' # Windows请改为 'COM3' 等
    BAUD_RATE = 9600
    
    # 发送策略：每隔 SEND_INTERVAL 秒发送一次，期间每隔 SAMPLE_INTERVAL 取样一次
    SEND_INTERVAL = 10.0  # 发送周期 (秒)
    SAMPLE_COUNT = 10     # 目标采样次数
    SAMPLE_INTERVAL = SEND_INTERVAL / float(SAMPLE_COUNT)

# =============================================================================
# 2. 辅助工具函数模块 (Utils)
# =============================================================================

def order_points(pts):
    """ 排序坐标：左上，右上，右下，左下 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def point_segment_distance(a, b, p):
    """ 计算点p到线段ab的最短距离 """
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0: return np.linalg.norm(p - a)
    ap = p - a
    t = np.dot(ap, ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# =============================================================================
# 3. 显示管理模块 (Display Manager)
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
        """ 绘制左上角的状态文字 """
        if not is_locked:
            text = f"Locking in {time_left:.1f}s"
            cv2.putText(img, text, (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)
        else:
            cv2.putText(img, "LOCKED", (20, 60), self.FONT, 0.7, self.COLOR_GREEN, 2)
            
            # 显示检测结果文字
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

            # 显示详细信息
            if vert_lbl != '0':
                cv2.putText(img, f"At vertex: {vert_lbl}", (20, 90), self.FONT, 0.7, (155, 65, 200), 2)
            else:
                cv2.putText(img, f"Closest edge: {edge_lbl}", (20, 90), self.FONT, 0.7, (155, 65, 200), 2)

    def draw_rectangles(self, img, outer, inner, mid_warped, inv_matrix):
        """ 绘制三个矩形框 """
        if outer is not None:
            cv2.drawContours(img, [outer], -1, self.COLOR_BLUE, 2)
        if inner is not None:
            cv2.drawContours(img, [inner], -1, self.COLOR_GREEN, 2)
        
        if mid_warped is not None and inv_matrix is not None:
            try:
                mid_orig = cv2.perspectiveTransform(mid_warped.reshape(1,4,2), inv_matrix)[0]
                mid_orig_int = mid_orig.astype(np.int32)
                cv2.drawContours(img, [mid_orig_int], -1, self.COLOR_PURPLE, 2)
            except: pass

    def draw_detailed_labels(self, img, mid_warped, inv_matrix):
        """ 绘制边框编号、长度和顶点编号 """
        if mid_warped is None or inv_matrix is None: return

        edge_display_map = {0: '1', 1: '4', 2: '3', 3: '2'} 
        vertex_display_map = {0: '1', 1: '4', 2: '3', 3: '2'}
        wm = mid_warped
        center = np.mean(wm, axis=0)
        
        for i in range(4):
            p1 = wm[i]
            p2 = wm[(i+1)%4]
            midpoint = (p1 + p2) / 2.0
            length = np.linalg.norm(p1 - p2)
            
            dir_vec = midpoint - center
            norm = np.linalg.norm(dir_vec)
            if norm == 0: norm = 1
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
# 4. 串口通信管理模块 (Serial Manager)
# =============================================================================

class SerialManager:
    def __init__(self):
        self.ser = None
        self.dist_samples = []
        self.last_sample_time = time.time()
        self.last_send_time = time.time()
        
        try:
            self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=1)
            self.ser.setDTR(False)
            time.sleep(0.5)
            print("Serial Connected.")
        except Exception as e:
            print(f"Serial Failed: {e}")

    def update(self, current_dist, res, edge, vert, curr_time):
        """
        [逻辑] 负责定时采样和强制定时发送。
        """
        # 1. 采样逻辑 (时间驱动)
        if curr_time - self.last_sample_time >= Config.SAMPLE_INTERVAL:
            if current_dist is not None:
                self.dist_samples.append(current_dist)
            self.last_sample_time = curr_time

        # 2. 发送逻辑 (时间驱动)
        if curr_time - self.last_send_time >= Config.SEND_INTERVAL:
            
            final_dist_int = 0
            
            # 无激光/空样本处理：如果列表为空（即这 SEND_INTERVAL 秒内没采样到激光），发送 0
            if len(self.dist_samples) > 0:
                avg_dist = sum(self.dist_samples) / len(self.dist_samples)
                final_dist_int = int(round(avg_dist))
                final_dist_int = min(999, max(0, final_dist_int))
            else:
                final_dist_int = 0
            
            send_str = f"{res}{edge}{vert}{final_dist_int}A"
            
            print(f"[Serial] Sending 10s Avg: {send_str} (Samples: {len(self.dist_samples)})")
            
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(send_str.encode())
                    self.ser.flush()
                except Exception as e:
                    print(f"Serial Error: {e}")
            
            self.dist_samples = [] 
            self.last_send_time = curr_time

    def close(self):
        if self.ser and self.ser.is_open: self.ser.close()

# =============================================================================
# 5. 主程序逻辑 (Main)
# =============================================================================

# --- [修改] 使用原始的层级轮廓查找函数，以提高稳定性 ---
def find_laser_centroid(mask):
    """
    分层级优先的激光检测：
    - 使用 cv2.RETR_TREE 获取轮廓层级
    - 优先选择有子轮廓的（内层轮廓），从中选择面积最大的子轮廓
    - 否则退回到面积最大的轮廓
    返回 (cx, cy, best_contour) 或 (None, None, None)
    """

    found = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2] if len(found) >= 2 else []
    hierarchy = found[-1] if len(found) >= 2 else None

    best_contour = None
    largest_area = 0

    if hierarchy is not None and len(contours) > 0:
        
        for idx, cnt in enumerate(contours):
            # hierarchy[0][idx] = [next, prev, first_child, parent]
            child_idx = int(hierarchy[0][idx][2])
            
            if child_idx != -1:
                # 如果有子轮廓，检查子轮廓的面积
                inner_cnt = contours[child_idx]
                inner_area = cv2.contourArea(inner_cnt)
               
                if inner_area > largest_area:
                    largest_area = inner_area
                    best_contour = inner_cnt
            else:
                # 如果没有子轮廓，检查自身的面积
                area = cv2.contourArea(cnt)
                if area > largest_area:
                    largest_area = area
                    best_contour = cnt
    else:
        # 如果没有层级信息，简单查找最大轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                best_contour = cnt

    if best_contour is not None and largest_area >= Config.LASER_MIN_AREA:
        M = cv2.moments(best_contour)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, best_contour

    return None, None, None
# --- [修改结束] ---


def main():
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Camera Error")
        return

    serial_mgr = SerialManager()
    display_mgr = DisplayManager() 
    
    start_time = time.time()
    rectangles_locked = False
    perspective_mtx = None
    inverse_mtx = None
    warped_mid_poly = None 
    draw_outer_cnt = None
    draw_inner_cnt = None

    while True:
        success, raw_img = cap.read()
        if not success: break

        h, w = raw_img.shape[:2]
        img = raw_img[int(h*Config.CROP_Y_START):int(h*Config.CROP_Y_END), 
                      int(w*Config.CROP_X_START):int(w*Config.CROP_X_END)]
        img = cv2.resize(img, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)
        
        # --- 激光提取 ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.add(cv2.inRange(hsv, (0,40,40), (10,255,255)), 
                       cv2.inRange(hsv, (156,40,40), (180,255,255)))
        # 形态学操作
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 调用复杂查找逻辑
        lx, ly, laser_cnt = find_laser_centroid(mask)
        laser_pt = (lx, ly) if lx is not None else None

        # --- 矩形锁定 (前3秒) ---
        curr_time = time.time()
        if not rectangles_locked and (curr_time - start_time) < Config.LOCK_TIME_WINDOW:
            img_clean = img.copy()
            img_clean[mask > 0] = (0,0,0)
            edges = cv2.Canny(cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY), Config.CANNY_TH1, Config.CANNY_TH2)
            cnts, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            valid = []
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx)==4 and cv2.contourArea(approx) > Config.RECT_MIN_AREA:
                    valid.append(approx.reshape(4,2))
            
            if len(valid) >= 2:
                valid.sort(key=cv2.contourArea, reverse=True)
                outer = order_points(valid[0])
                inner = order_points(valid[-1])
                dst_pts = np.float32([[0,0], [400,0], [400,500], [0,500]])
                
                try:
                    perspective_mtx = cv2.getPerspectiveTransform(outer, dst_pts)
                    inverse_mtx = cv2.getPerspectiveTransform(dst_pts, outer)
                    
                    in_warp = cv2.perspectiveTransform(inner.reshape(1,4,2), perspective_mtx)[0]
                    in_reorder = np.zeros_like(in_warp)
                    used = set()
                    for i in range(4): 
                        dists = [np.linalg.norm(in_warp[j]-dst_pts[i]) for j in range(4)]
                        for u in used: dists[u] = 9e9
                        best = np.argmin(dists)
                        in_reorder[i] = in_warp[best]
                        used.add(best)
                    
                    warped_mid_poly = (dst_pts + in_reorder) / 2.0
                    draw_outer_cnt = outer.reshape(4,1,2).astype(np.int32)
                    draw_inner_cnt = inner.reshape(4,1,2).astype(np.int32)
                    rectangles_locked = True
                    print("Locked!")
                except: pass

        # --- 计算逻辑：确保无激光时返回 0 ---
        res, edge, vert = '0', '0', '0'
        dist = None 

        if rectangles_locked and laser_pt:
            # 绘制激光轮廓和中心点 (恢复原始代码的显示逻辑)
            if laser_cnt is not None:
                cv2.drawContours(img, [laser_cnt], -1, (0, 0, 255), 3)
            
            pt_w = cv2.perspectiveTransform(np.array([[[lx, ly]]], dtype=np.float32), perspective_mtx)[0][0]
            
            dists = [point_segment_distance(warped_mid_poly[i], warped_mid_poly[(i+1)%4], pt_w) for i in range(4)]
            min_d = min(dists)
            dist = float(min_d)
            
            if min_d <= 3.0: res = '2'
            elif cv2.pointPolygonTest(warped_mid_poly, (float(pt_w[0]), float(pt_w[1])), False) >= 0: res = '1'
            else: res = '3'
            
            idx = np.argmin(dists)
            edge_map = {0: '1', 3: '2', 2: '3', 1: '4'} 
            edge = edge_map.get(idx, '0')
            
            v_dists = [np.linalg.norm(pt_w - v) for v in warped_mid_poly]
            if min(v_dists) < 20.0 and min(v_dists) < min_d * 2:
                vert_map = {0: '1', 3: '2', 2: '3', 1: '4'} 
                vert = vert_map.get(np.argmin(v_dists), '0')
        else:
            # 无激光，保持 res, edge, vert 为 '0'，dist 为 None
            pass 

        # --- 串口更新 ---
        serial_mgr.update(dist, res, edge, vert, curr_time)

        # --- 界面显示 ---
        display_mgr.draw_status(img, rectangles_locked, Config.LOCK_TIME_WINDOW - (curr_time - start_time), res, edge, vert)
        display_mgr.draw_laser(img, laser_pt)
        if rectangles_locked:
            display_mgr.draw_rectangles(img, draw_outer_cnt, draw_inner_cnt, warped_mid_poly, inverse_mtx)
            display_mgr.draw_detailed_labels(img, warped_mid_poly, inverse_mtx)

        cv2.imshow("Laser Detector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    serial_mgr.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
