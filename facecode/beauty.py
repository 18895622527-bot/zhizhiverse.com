# -*- coding: utf-8 -*-
"""
美颜：照着美图秀秀「自然」那一档做 —— 脸型圆润一点、皮肤亮白一点、眼睛微微大一点。2026-11-07

李总：「直接就是脸型圆润一点，皮肤亮白一点，五官微调一下就行了。」

【怎么做】全在 OpenCV + MediaPipe 人脸网格（468 点）上，不接新模型、不花钱：
  ① 瘦脸：两腮下颌往脸中心推一点（Gustafsson 局部平移形变，美图/抖音瘦脸都是这一套）
  ② 大眼：以眼睛中心为圆心局部放大 6%~8%
  ③ 磨皮：只在皮肤区域（脸轮廓里，挖掉眼、眉、嘴）做双边滤波，再把一点高频细节加回来，不糊
  ④ 美白：皮肤区域 L 通道往上提；彩色照片顺便去掉一点黄
  ⑤ 嘴唇补一点血色（只彩色照片）
  每张脸都做 —— 合影两个人一起美，不会只美一个。

【强度】k 0~1，默认 0.7。0 = 不美颜。
【安全】任何一步出错就跳过那一步，原图照样返回，绝不因为美颜把修复结果弄丢。
"""
import cv2
import numpy as np

# MediaPipe face mesh 编号
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# 瘦脸推的点：左右下颌 / 腮
# ⚠️ 2026-11-07 第一版推 5 个点、每点 6%，叠加后把耳下的头发和暗背景拖进脸里，下颌出黑道。
#    现在每边只推下颌转角附近 2 个点，力度 3%，半径放大到脸宽 0.25，过渡更平。
SLIM_L = [172, 136]
SLIM_R = [397, 365]
NOSE_TIP = 4

_MESH = {}


def _mesh():
    if 'm' not in _MESH:
        import mediapipe as mp
        _MESH['m'] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=6, refine_landmarks=True, min_detection_confidence=0.4)
    return _MESH['m']


def _mesh_on(img):
    h, w = img.shape[:2]
    r = _mesh().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return [np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
            for fl in (r.multi_face_landmarks or [])]


def _det_boxes(img):
    """远距离人脸检测（BlazeFace full-range），脸占画面很小也找得到。回 [(x0,y0,x1,y1)]"""
    if 'd' not in _MESH:
        import mediapipe as mp
        _MESH['d'] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
    h, w = img.shape[:2]
    r = _MESH['d'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for d in (r.detections or []):
        b = d.location_data.relative_bounding_box
        out.append((b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h))
    return out


def faces(img):
    """每张脸 468 个点（像素坐标），numpy (N,2)。
    ⚠️ 2026-11-07 合影整张相册页（2000px、脸只有 250px）人脸网格一张都认不出。
       所以先整张试；不够就用远距离检测找到每张脸，裁出来（放大到 480）单独取点，再映射回原图。"""
    out = _mesh_on(img)
    boxes = _det_boxes(img)
    if len(out) >= len(boxes) and out:
        return out
    h, w = img.shape[:2]
    got = []
    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(bw, bh) * 1.8
        X0, Y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        X1, Y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
        roi = img[Y0:Y1, X0:X1]
        if roi.size == 0:
            continue
        sc = 480.0 / max(roi.shape[:2])
        big = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc > 1 else roi
        sc = sc if sc > 1 else 1.0
        for p in _mesh_on(big):
            q = p / sc + np.array([X0, Y0], np.float32)
            # 一个框里只要中心落在框内的那张，防止相邻的脸被取两次
            c = q[NOSE_TIP]
            if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
                got.append(q)
                break
    return got if len(got) > len(out) else out


def _is_gray(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) < 25.0


def _local_translate(img, c, m, r):
    """Gustafsson 局部平移：以 c 为圆心、半径 r 的圆里，像素沿 c→m 方向挪，越靠圆心挪得越多。"""
    h, w = img.shape[:2]
    x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
    x1, y1 = int(min(w, c[0] + r + 1)), int(min(h, c[1] + r + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - c[0], ys - c[1]
    d2 = dx * dx + dy * dy
    r2 = float(r * r)
    mc = np.array(m, np.float32) - np.array(c, np.float32)
    mc2 = float(mc[0] ** 2 + mc[1] ** 2)
    inside = d2 < r2
    ratio = np.zeros_like(d2)
    ratio[inside] = ((r2 - d2[inside]) / (r2 - d2[inside] + mc2)) ** 2
    mapx = xs - ratio * mc[0]
    mapy = ys - ratio * mc[1]
    roi = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = img.copy()
    out[y0:y1, x0:x1] = roi
    return out


def _local_scale(img, c, r, a):
    """以 c 为圆心局部放大（a>0 放大）。圆心放大最多，边缘不变。"""
    h, w = img.shape[:2]
    x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
    x1, y1 = int(min(w, c[0] + r + 1)), int(min(h, c[1] + r + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - c[0], ys - c[1]
    d = np.sqrt(dx * dx + dy * dy) / float(r)
    k = np.where(d < 1.0, 1.0 - a * (1.0 - d * d), 1.0)
    mapx = c[0] + dx * k
    mapy = c[1] + dy * k
    roi = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = img.copy()
    out[y0:y1, x0:x1] = roi
    return out


def _skin_mask(shape, pts):
    """脸轮廓里，挖掉眼睛、眉毛、嘴。羽化过的 0~1。"""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts[FACE_OVAL].astype(np.int32)), 1.0)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS):
        cv2.fillPoly(m, [cv2.convexHull(pts[idx].astype(np.int32))], 0.0)
    fw = float(np.ptp(pts[FACE_OVAL][:, 0]))
    return cv2.GaussianBlur(m, (0, 0), max(3.0, fw / 14.0))


def beautify(img, k=0.7):
    """回 (图, 说明)。说明形如 'beauty0.7:2face+slim+eye+smooth+white+lips'。"""
    k = max(0.0, min(1.0, float(k)))
    if k <= 0:
        return img, ''
    try:
        fs = faces(img)
    except Exception as e:
        print('[beauty] 找不到脸关键点，跳过：', e, flush=True)
        return img, ''
    if not fs:
        return img, ''
    # 黑白不黑白按【脸】判断，不按整张：相册页是粉的，脸是黑白的（2026-11-07 那张合影被染了粉嘴唇）
    fm = np.zeros(img.shape[:2], np.uint8)
    for p in fs:
        cv2.fillConvexPoly(fm, cv2.convexHull(p[FACE_OVAL].astype(np.int32)), 1)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    gray = float(sat[fm > 0].mean()) < 28.0 if fm.any() else _is_gray(img)
    out = img.copy()
    notes = ['%dface' % len(fs)]

    # ① 瘦脸 ② 大眼 —— 形变先做，后面的磨皮美白按形变后的位置
    try:
        for p in fs:
            fw = float(np.ptp(p[FACE_OVAL][:, 0]))
            ctr = p[NOSE_TIP]
            for i in SLIM_L + SLIM_R:
                c = p[i]
                v = ctr - c
                m = c + v * (0.035 * k)           # 往鼻尖方向推 3.5%·k 的距离
                out = _local_translate(out, c, m, fw * 0.25)
            for eye in (LEFT_EYE, RIGHT_EYE):
                ec = p[eye].mean(axis=0)
                ew = float(np.ptp(p[eye][:, 0]))
                out = _local_scale(out, ec, ew * 0.9, 0.07 * k)
        notes += ['slim', 'eye']
        fs = faces(out) or fs                     # 形变之后重新取点
    except Exception as e:
        print('[beauty] 形变跳过：', e, flush=True)

    # ③ 磨皮 ④ 美白
    try:
        mask = np.zeros(out.shape[:2], np.float32)
        for p in fs:
            mask = np.maximum(mask, _skin_mask(out.shape, p))
        m3 = mask[:, :, None]
        f = out.astype(np.float32)
        # ⚠️ 第一版混 0.75、细节只回 0.35，像蜡像。现在混 0.5、细节回 0.6，留住老照片的颗粒
        smooth = cv2.bilateralFilter(out, 0, 20 + 20 * k, 5 + 3 * k).astype(np.float32)
        detail = f - cv2.GaussianBlur(f, (0, 0), 1.2)
        sk = smooth + detail * 0.6
        f = f * (1 - m3 * (0.5 * k)) + sk * (m3 * (0.5 * k))
        lab = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        lab[:, :, 0] = L + (255.0 - L) * (0.13 * k) * mask         # 皮肤提亮（0.22 太白，像戴面具）
        if not gray:
            lab[:, :, 2] = lab[:, :, 2] - (lab[:, :, 2] - 128.0) * (0.25 * k) * mask   # 去黄
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        notes += ['smooth', 'white']
    except Exception as e:
        print('[beauty] 磨皮美白跳过：', e, flush=True)

    # ⑤ 嘴唇血色（只彩色）
    if not gray:
        try:
            lm = np.zeros(out.shape[:2], np.float32)
            for p in fs:
                cv2.fillPoly(lm, [p[LIPS].astype(np.int32)], 1.0)
            if lm.max() > 0:
                lm = cv2.GaussianBlur(lm, (0, 0), 3)
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:, :, 1] += lm * (12.0 * k)
                lab[:, :, 0] -= lm * (4.0 * k)
                out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                notes.append('lips')
        except Exception as e:
            print('[beauty] 嘴唇跳过：', e, flush=True)

    # 整体微微提亮
    out = cv2.convertScaleAbs(out, alpha=1.0 + 0.04 * k, beta=5 * k)
    return out, 'beauty%.1f:%s' % (k, '+'.join(notes))
