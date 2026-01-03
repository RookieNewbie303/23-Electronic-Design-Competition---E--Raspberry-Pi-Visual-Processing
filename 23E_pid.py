# -*- coding: utf-8 -*-
import cv2
import numpy as np
import serial
import time
import math

# ---------------- Config ----------------
class Config:
    CAMERA_ID = 0
    FRAME_WIDTH = 1080
    FRAME_HEIGHT = 720

    CROP_Y_START = 0.1
    CROP_Y_END   = 0.6
    CROP_X_START = 0.30
    CROP_X_END   = 0.70
    ZOOM_FACTOR = 1.2

    RECT_MIN_AREA = 2000
    CANNY_TH1 = 50
    CANNY_TH2 = 150
    LOCK_TIME_WINDOW = 3.0

    LASER_MIN_AREA = 2

    # 串口：发送给二维云台的控制器（按你硬件改）
    CONTROL_PORT = '/dev/ttyUSB0'   # 如果用 USB-Serial，请填 ttyUSBx；若用 GPIO TTL，请填 serial0/AMA0 并确认接线
    CONTROL_BAUD = 115200

    # 控制周期（s）
    CTRL_DT = 0.02  # 控制循环 50 Hz

    # 相机视角：用来把像素误差换成角度（度/像素）
    H_FOV_DEG = 62.0   # 水平视角（度），按你镜头填；PiCamera ~62° 是常见值
    V_FOV_DEG = 40.0   # 垂直视角，可按比例设置

    # PID 初始参数（需要调）
    PAN_KP = 0.6
    PAN_KI = 0.01
    PAN_KD = 0.03

    TILT_KP = 0.6
    TILT_KI = 0.01
    TILT_KD = 0.03

    # 最大角速度限制（deg/s）
    MAX_PAN_VEL = 40.0
    MAX_TILT_VEL = 40.0

    # 轨迹参数
    POINTS_PER_EDGE = 20  # 每条边等分多少个采样点（轨迹密度）
    TARGET_REACH_PIX = 8  # 到达目标的像素阈值（到达后切下一个点）

    # 若丢激光超过 N 秒则停止云台
    LOST_TIMEOUT = 0.5

# ---------------- helpers ----------------
def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def point_segment_distance(a,b,p):
    ab = b-a
    ab2 = np.dot(ab,ab)
    if ab2==0: return np.linalg.norm(p-a)
    ap = p-a
    t = np.dot(ap,ab)/ab2
    t = np.clip(t,0.0,1.0)
    closest = a + t*ab
    return np.linalg.norm(p-closest)

class PID:
    def __init__(self, kp, ki, kd, out_min=-1e6, out_max=1e6):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integrator = 0.0
        self.prev_err = 0.0
        self.prev_time = None

    def reset(self):
        self.integrator = 0.0
        self.prev_err = 0.0
        self.prev_time = None

    def update(self, err, dt):
        if self.prev_time is None:
            self.prev_time = time.time()
            de = 0.0
        else:
            de = (err - self.prev_err) / dt if dt>0 else 0.0
        self.integrator += err * dt
        out = self.kp * err + self.ki * self.integrator + self.kd * de
        out = max(self.out_min, min(self.out_max, out))
        self.prev_err = err
        return out

# ---------------- Serial control helper ----------------
class GimbalSerial:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(0.1)
            # try to avoid DTR reset if present
            try:
                self.ser.setDTR(False)
            except:
                pass
            print("Gimbal serial opened:", self.port, self.baud)
        except Exception as e:
            print("Open serial failed:", e)
            self.ser = None

    def send_cmd(self, pan_vel_deg_s, tilt_vel_deg_s):
        """发送角速度指令给云台。协议自定：这里用 ASCII: 'P{:+04d}T{:+04d}\\n' (deg/s)"""
        # 限幅并转整数
        pv = int(round(pan_vel_deg_s))
        tv = int(round(tilt_vel_deg_s))
        # clamp (optional)
        pv = max(-999, min(999, pv))
        tv = max(-999, min(999, tv))
        s = f"P{pv:+04d}T{tv:+04d}\n"
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(s.encode())
            except Exception as e:
                print("Serial write err:", e)
        # for debug
        # print("SEND:", s.strip())

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

# ---------------- Laser detection (reuse your hierarchy method) ----------------
def find_laser_centroid(mask, min_area=Config.LASER_MIN_AREA):
    found = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2] if len(found) >= 2 else []
    hierarchy = found[-1] if len(found) >= 2 else None
    best_contour = None
    largest_area = 0
    if hierarchy is not None and len(contours) > 0:
        for idx,cnt in enumerate(contours):
            child_idx = int(hierarchy[0][idx][2])
            if child_idx != -1:
                inner_cnt = contours[child_idx]
                inner_area = cv2.contourArea(inner_cnt)
                if inner_area > largest_area:
                    largest_area = inner_area
                    best_contour = inner_cnt
            else:
                area = cv2.contourArea(cnt)
                if area > largest_area:
                    largest_area = area
                    best_contour = cnt
    else:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                best_contour = cnt
    if best_contour is not None and largest_area >= min_area:
        M = cv2.moments(best_contour)
        if M.get("m00",0) != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            return cx, cy, best_contour
    return None, None, None

# ---------------- trajectory helper ----------------
def build_orbit_points(image_vertices, points_per_edge):
    """给定矩形四个顶点(顺序 TL,TR,BR,BL 或任意顺序)，返回按边 CCW 排列并把每条边分 points_per_edge 点的轨迹点列表"""
    # ensure 4x2 array
    pts = np.array(image_vertices, dtype=np.float32).reshape(4,2)
    # order to TL,TR,BR,BL
    ordered = order_points(pts)
    # compute midpoints? we want perimeter points along ordered corners CCW
    path = []
    for i in range(4):
        a = ordered[i]
        b = ordered[(i+1)%4]
        for k in range(points_per_edge):
            t = k / float(points_per_edge)
            p = (1-t)*a + t*b
            path.append((float(p[0]), float(p[1])))
    # close loop by appending first
    path.append((float(ordered[0][0]), float(ordered[0][1])))
    return path

# ---------------- main ----------------
def main():
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("camera open fail")
        return

    gimbal = GimbalSerial(Config.CONTROL_PORT, Config.CONTROL_BAUD)
    pid_pan = PID(Config.PAN_KP, Config.PAN_KI, Config.PAN_KD, -Config.MAX_PAN_VEL, Config.MAX_PAN_VEL)
    pid_tilt = PID(Config.TILT_KP, Config.TILT_KI, Config.TILT_KD, -Config.MAX_TILT_VEL, Config.MAX_TILT_VEL)

    # timing
    prev_time = time.time()
    last_laser_time = 0.0

    locked = False
    perspective_mtx = None
    inverse_mtx = None
    warped_mid_poly = None
    vertex_image_pts = None
    orbit_points = []
    current_target_idx = 0

    # precompute pixel->deg factors
    # after cropping + zoom, effective frame size will be known after first frame. we'll compute later.

    first_frame = True
    fx = fy = None
    h_pix = w_pix = None

    while True:
        t0 = time.time()
        ok, raw = cap.read()
        if not ok:
            print("frame fail")
            break

        h,w = raw.shape[:2]
        crop = raw[int(h*Config.CROP_Y_START):int(h*Config.CROP_Y_END),
                   int(w*Config.CROP_X_START):int(w*Config.CROP_X_END)]
        img = cv2.resize(crop, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)

        if first_frame:
            h_pix, w_pix = img.shape[:2]
            # compute degree per pixel
            deg_per_pix_x = Config.H_FOV_DEG / float(w_pix)
            deg_per_pix_y = Config.V_FOV_DEG / float(h_pix)
            first_frame = False
            print("frame pixels:", w_pix, h_pix, "deg/pix:", deg_per_pix_x, deg_per_pix_y)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.add(cv2.inRange(hsv, (0,40,40), (10,255,255)),
                       cv2.inRange(hsv, (156,40,40), (180,255,255)))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        lx, ly, cnt = find_laser_centroid(mask)
        laser_pt = (lx, ly) if lx is not None else None
        if laser_pt:
            last_laser_time = time.time()

        curr_time = time.time()

        # locking rectangle (first few seconds)
        if (not locked) and (curr_time - last_laser_time) < Config.LOCK_TIME_WINDOW:
            # don't use laser pixels for edge detection
            img_clean = img.copy()
            img_clean[mask>0] = (0,0,0)
            edges = cv2.Canny(cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY), Config.CANNY_TH1, Config.CANNY_TH2)
            cnts, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            rects = []
            areas = []
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx)==4:
                    area = cv2.contourArea(approx)
                    if area > Config.RECT_MIN_AREA:
                        rects.append(approx.reshape(4,2).astype(np.float32))
                        areas.append(area)
            if len(rects) >= 2:
                idxs = np.argsort(areas)[::-1]
                outer = order_points(rects[idxs[0]])
                inner = order_points(rects[idxs[-1]])
                dst_pts = np.float32([[0,0],[400,0],[400,500],[0,500]])
                try:
                    perspective_mtx = cv2.getPerspectiveTransform(outer, dst_pts)
                    inverse_mtx = cv2.getPerspectiveTransform(dst_pts, outer)
                    outer_w = cv2.perspectiveTransform(outer.reshape(1,4,2), perspective_mtx)[0]
                    inner_w = cv2.perspectiveTransform(inner.reshape(1,4,2), perspective_mtx)[0]
                    # reorder inner to match outer
                    inner_reorder = np.zeros_like(inner_w)
                    used = set()
                    for i in range(4):
                        dists = [np.linalg.norm(inner_w[j] - outer_w[i]) if j not in used else 9e9 for j in range(4)]
                        best = int(np.argmin(dists))
                        inner_reorder[i] = inner_w[best]
                        used.add(best)
                    warped_mid = (outer_w + inner_reorder) / 2.0
                    warped_mid_poly = warped_mid.astype(np.float32)
                    # compute image-space vertex points (transform back with inverse)
                    vertex_image_pts = cv2.perspectiveTransform(warped_mid_poly.reshape(1,4,2), inverse_mtx)[0]
                    # build orbit path in image coordinates
                    orbit_points = build_orbit_points(vertex_image_pts, Config.POINTS_PER_EDGE)
                    current_target_idx = 0
                    locked = True
                    print("Locked and built orbit: points =", len(orbit_points))
                except Exception as e:
                    print("perspective/lock failed:", e)

        # display rectangles/vertices
        if locked and vertex_image_pts is not None:
            # draw rectangle and vertices
            try:
                poly = vertex_image_pts.astype(np.int32).reshape(4,1,2)
                cv2.drawContours(img, [poly], -1, (255,0,0), 2)
                for i,p in enumerate(vertex_image_pts):
                    cv2.circle(img, (int(p[0]), int(p[1])), 4, (0,255,255), -1)
            except: pass

        # control logic only when locked
        pan_cmd = 0.0
        tilt_cmd = 0.0

        # find current target point in image coords
        if locked and len(orbit_points)>0:
            target = orbit_points[current_target_idx]
            tx, ty = target  # pixel coords in img
            cv2.circle(img, (int(tx), int(ty)), 6, (0,128,255), 2)

            if laser_pt is not None:
                # pixel error: positive x -> laser is to right of target (need positive pan?), sign convention later
                err_x = (tx - lx)   # target - measurement; positive means target is right of laser
                err_y = (ty - ly)   # target - measurement; positive means target is below laser

                dt = max(1e-6, curr_time - prev_time)

                # convert pixel error to degrees (image x increases to right, y increases down)
                err_deg_x = err_x * (Config.H_FOV_DEG / float(w_pix))
                err_deg_y = err_y * (Config.V_FOV_DEG / float(h_pix))

                # PID compute (we take pan controlling left-right, tilt controlling up-down)
                pan_vel = pid_pan.update(err_deg_x, dt)   # deg/s
                tilt_vel = pid_tilt.update(err_deg_y, dt)

                # clamp
                pan_vel = max(-Config.MAX_PAN_VEL, min(Config.MAX_PAN_VEL, pan_vel))
                tilt_vel = max(-Config.MAX_TILT_VEL, min(Config.MAX_TILT_VEL, tilt_vel))

                pan_cmd = pan_vel
                tilt_cmd = tilt_vel

                # if near target in pixels, advance to next point
                if math.hypot(err_x, err_y) < Config.TARGET_REACH_PIX:
                    current_target_idx = (current_target_idx + 1) % len(orbit_points)
            else:
                # no laser: stop motors (or optionally search)
                pan_cmd = 0.0
                tilt_cmd = 0.0
                # consider leaving current target index unchanged

        else:
            # not locked or no orbit: do nothing
            pan_cmd = 0.0
            tilt_cmd = 0.0

        # safety: if laser lost long, stop and reset PIDs integrators
        if time.time() - last_laser_time > Config.LOST_TIMEOUT:
            pan_cmd = 0.0; tilt_cmd = 0.0
            pid_pan.reset(); pid_tilt.reset()

        # send to gimbal every control period
        if curr_time - prev_time >= Config.CTRL_DT:
            gimbal.send_cmd(pan_cmd, tilt_cmd)
            prev_time = curr_time

        # draw laser
        if laser_pt:
            cv2.circle(img, (lx,ly), 4, (0,0,255), -1)

        # overlay some state text
        cv2.putText(img, f"Target idx: {current_target_idx}/{len(orbit_points)}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(img, f"PAN: {pan_cmd:.1f} TILT: {tilt_cmd:.1f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        cv2.imshow("orbit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    gimbal.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
