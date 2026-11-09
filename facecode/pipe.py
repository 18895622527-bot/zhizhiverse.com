# -*- coding: utf-8 -*-
"""
一条直线的预处理：定方向 → 定位 → 裁 → 验证。所有照片走同一条路，没有「有脸才走 / 没脸才走」的分支。

2026-11-03 李总：「一张照片一条分支，就是打地鼠的根。」
之前 crop.prepare 里有四级降级、七八道闸，每道闸都是某一张照片的教训，下一张照片就撞上。
这个文件只保留一条路，crop.py 里的函数当零件用。

【四步】
  ① 方向  orient()   先试 0/90/180/270 四个粗方向，哪个方向 MediaPipe 认出的脸最多、最正就选哪个；
                     再按所有脸倾斜角的【中位数】细调（合影里头靠在一起不会被当成歪）。
                     细调完再读一次，没变正就退回粗方向。脸太小认不出就粗扫一圈（_sweep_face）。
  ② 定位  locate()   转正之后：先找纸（deskew / auto_crop，只在裁完脸没少的时候采用），
                     找不到就按【所有脸】的并集裁（face_crop，合影不会裁成一个人）。
  ③ 去边  strip_border
  ④ 验证  裁完脸的数量不能比裁前少，少了就退回上一步的结果。
                     ⚠️ 两条铁律：裁错了不许悄悄送去修；裁不到就原样放行让用户自己框。

【返回】(图, 说明)。说明是给人看的，也原样回给前端：
    'orient180 tilt-3 paper+border'  /  'orient0 faces+border'  /  ''（没动）
"""
import cv2
import numpy as np

import crop


def _faces_lm(img):
    """用 MediaPipe 关键点推出脸框，比 Haar 稳，而且带倾斜角。回 [(x,y,w,h,tilt)]。"""
    out = []
    for f in crop.landmarks(img):
        le, re, chin = f['eye_l'], f['eye_r'], f['chin']
        mid = (le + re) / 2.0
        d = float(np.linalg.norm(re - le))
        if d < 8:
            continue
        s = max(d * 2.4, float(np.linalg.norm(chin - mid)) * 1.6)
        # ⚠️ 倾角用「眼睛中点 → 下巴」这根线算，不用两眼连线：
        #    两眼连线在倒着的脸上也是水平的，分不出 0° 和 180°（2026-11-03 一张倒着的照片被当成正的）。
        #    下巴在正下方 = 0°，在正上方 = ±180°。正值 = 需要逆时针转这么多度才正（和 cv2 的角度方向一致）。
        v = chin - mid
        tilt = -float(np.degrees(np.arctan2(float(v[0]), float(v[1]))))
        out.append((int(mid[0] - s / 2), int(mid[1] - s * 0.4), int(s), int(s), tilt))
    return out


_DET = {}


def det_score(img):
    """MediaPipe 人脸检测器（BlazeFace）在这张图上所有脸的置信度之和。只认正着的脸，所以可以当方向的裁判。"""
    try:
        if 'd' not in _DET:
            import mediapipe as mp
            _DET['d'] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        h, w = img.shape[:2]
        sc = 640.0 / max(h, w)
        sm = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
        r = _DET['d'].process(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))
        return float(sum(d.score[0] for d in (r.detections or [])))
    except Exception:
        return 0.0


def _blaze_boxes(img, conf=0.6):
    """BlazeFace（远距离模型）找脸，回 [(x,y,w,h)] 原图像素。
    ⚠️ 2026-11-07：一张两人合影，face mesh 只认出男方，BlazeFace 和 Haar 都认出了两个人。
       按 mesh 的结果裁，女方被裁掉了。所以按脸裁之前用它补一遍漏掉的脸。"""
    try:
        if 'd1' not in _DET:
            import mediapipe as mp
            _DET['d1'] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=conf)
        h, w = img.shape[:2]
        sc = min(1.0, 1000.0 / max(h, w))
        sm = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
        r = _DET['d1'].process(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))
        out = []
        for d in (r.detections or []):
            if d.score[0] < conf:
                continue
            b = d.location_data.relative_bounding_box
            x, y, bw, bh = b.xmin * w, b.ymin * h, b.width * w, b.height * h
            if bw < 0.03 * min(h, w):   # 太小的不算（衣服花纹、纽扣）
                continue
            out.append((int(x), int(y), int(bw), int(bh)))
        return out
    except Exception:
        return []


def _merge_faces(img, fs):
    """mesh 的脸 + BlazeFace 补上的、mesh 漏掉的脸。"""
    out = [tuple(int(v) for v in f[:4]) for f in fs]
    for b in _blaze_boxes(img):
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        if any(f[0] - 0.25 * f[2] <= cx <= f[0] + 1.25 * f[2] and f[1] - 0.25 * f[3] <= cy <= f[1] + 1.25 * f[3] for f in out):
            continue
        out.append(b)
    return out


def _count_faces(img):
    return max(len(_faces_lm(img)), len(_blaze_boxes(img)))


def find_faces_small(img):
    """Haar 在缩到 600px 的图上找正脸。只用来判「这个方向有没有正着的脸」。"""
    h, w = img.shape[:2]
    sc = 600.0 / max(h, w)
    sm = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
    return crop.find_faces(sm, min_side=24)


def _rot90(img, k):
    if k == 0:
        return img
    return cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])


def orient(img):
    """回 (转正后的图, 说明, 脸列表)。认不出任何脸就 (原图, '', [])。"""
    best = None
    for k in (0, 90, 180, 270):
        r = _rot90(img, k)
        fs = _faces_lm(r)
        if not fs:
            continue
        med = float(np.median([abs(f[4]) for f in fs]))
        # ⚠️ 倒着的脸 MediaPipe 会硬套成一张正的（眼睛和嘴对调），tilt 报 0.6°——靠它分不出 0° 和 180°。
        #    Haar 只认正着的脸，拿它当裁判：Haar 也认出脸的方向才算数。
        # 四个方向都算完再比。裁判是 MediaPipe 的【人脸检测器】（BlazeFace）的置信度 ——
        # 实测七张图：正着 0.82~0.93，转 90°/180°/270° 几乎全是空。
        # ⚠️ 不用 Haar 当裁判：它在倒着的图上会报假脸，而且比正着还多（2026-11-03 四张正的图被倒过来）。
        # ⚠️ 不用 face_mesh 的 tilt 当裁判：倒着的脸它会硬套成正的，tilt 报 0.6°。
        # 0° 只在打平时优先（k=0 先算，后面的要【严格】更好才换）。
        score = (round(det_score(r), 1), len(fs), -med)
        if best is None or score > best[0]:
            best = (score, k, r, fs, med)
    if best is None:
        # 脸太小（照片只占画面一角）：粗扫一圈找到脸的那一小块，先裁到那块再转、再认
        found = crop._sweep_face(img, step=20)
        if found is None:
            return img, '', []
        a, (x0, y0, x1, y1) = found
        h, w = img.shape[:2]
        mx, my = int(0.5 * (x1 - x0)), int(0.5 * (y1 - y0))
        roi = img[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
        # ⚠️ 先在没转的小块上读角度（这时脸够大，MediaPipe 读得出 51.6°），
        #    转完画布会被补边撑大一圈，脸相对又变小，读不出来了 —— 所以读在前、转在后。
        fs0 = _faces_lm(roi)
        a = float(np.median([f[4] for f in fs0])) if fs0 else float(a)
        # ⚠️ 2026-11-07：合影里 mesh 只认出一个人，按他歪着的头把整张照片转了 -9°，后面去白边全乱了。
        #    BlazeFace 数出不止一张脸 = 合影 —— 合影只认 90° 的粗方向，不按头的角度细转（和下面的规矩一致）。
        if len(_blaze_boxes(roi)) >= 2:
            a = float(round(a / 90.0) * 90.0)
        r = crop.rotate_keep(roi, a) if abs(a) > 1 else roi
        fs = _faces_lm(r)
        if not fs:
            # 转完认不出就把补出来的边裁掉再认一次
            d = r.shape[0]; c = int(d * 0.18)
            r = r[c:d - c, c:d - c]
            fs = _faces_lm(r)
        if not fs:
            return img, '', []
        return r, 'sweep%+.0f' % a, fs
    else:
        note = 'orient%d' % best[1]
    r, fs = best[2], best[3]
    tilts = [f[4] for f in fs]
    t = float(np.median(tilts))
    # 细调的规矩：
    #   单人：超过 12° 才转 —— 12° 以内多半是人本来歪着头，不是纸歪了
    #   多人：所有人歪的方向一致（极差 ≤ 10°）且中位数超过 5° 才转 —— 合影里头靠在一起是姿势，不是纸歪
    #   转完再读一次，没变正就不认
    spread = max(tilts) - min(tilts)
    # mesh 漏脸时（合影里常见）用 BlazeFace 再数一遍，不止一张脸就按合影处理
    if len(fs) >= 2 or len(_blaze_boxes(r)) >= 2:
        # 合影【不做细调】。试过两种证据（眼睛倾角中位数、脸连线倾角）—— 两个人一高一矮、
        # 头互相靠着，两种证据都会说「歪了」，把一张正的合影转歪（2026-11-03 相册页那张，连着两版）。
        # 合影的纸歪了几度，交给「是这张吗？」那一步让用户拖一下，比转错安全。90°/180° 粗方向照常处理。
        want = False
    else:
        want = 12.0 < abs(t) <= 45.0
    if want:
        r2 = crop.rotate_keep(r, t)
        fs2 = _faces_lm(r2)
        if fs2 and len(fs2) >= len(fs):
            if len(fs2) >= 2:
                pts2 = np.array([[f[0] + f[2] / 2.0, f[1] + f[3] / 2.0] for f in fs2], np.float32)
                vx, vy, _, _ = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)
                t2 = abs(float(np.degrees(np.arctan2(float(vy), float(vx)))))
                t2 = min(t2, 180 - t2)
            else:
                t2 = abs(float(fs2[0][4]))
            if t2 <= 6.0:
                return r2, note + ' tilt%+.0f' % t, fs2
    return r, note, fs


def _union(fs):
    x0 = min(f[0] for f in fs); y0 = min(f[1] for f in fs)
    x1 = max(f[0] + f[2] for f in fs); y1 = max(f[1] + f[3] for f in fs)
    return x0, y0, x1, y1


def _keeps_faces(sub, n):
    return len(_faces_lm(sub)) >= n


def locate(img, fs, allow_paper=True, strict_keep=True):
    """转正之后找照片本身。回 (图, 说明)。"""
    n = len(fs)
    if allow_paper:
        out, why = crop.deskew(img)
        if out is not None and (n == 0 or _keeps_faces(out, n)):
            return out, 'paper'
        box = crop.auto_crop(img)
        if box:
            sub = img[box[1]:box[3], box[0]:box[2]]
            if n == 0 or _keeps_faces(sub, n):
                return sub, 'paper'
    if n:
        allf = _merge_faces(img, fs)
        if len(allf) > n:
            print('[pipe] mesh 认出 %d 张脸，BlazeFace 补到 %d 张' % (n, len(allf)), flush=True)
        box = crop.face_crop(img, allf)
        if box:
            sub = img[box[1]:box[3], box[0]:box[2]]
            keep = float(sub.shape[0] * sub.shape[1]) / float(img.shape[0] * img.shape[1])
            # 已经是一张构图好的人像时，按脸裁只会再切掉一圈 —— 裁不掉四成以上就不值得动
            # ⚠️ 裁完还得装得下【所有】脸（两种检测器里多的那个数），少一张就不裁
            if (not strict_keep or keep <= 0.6) and _count_faces(sub) >= len(allf):
                return sub, 'faces'
    return img, ''


FULL_BODY = 4.5


def body_below_face(img):
    """最大那张脸（BlazeFace）下沿往下，连着这张脸的人（人像分割）还有几个脸高。认不出回 0。"""
    m = crop._mp()
    if not m or 'det' not in m:
        return 0.0
    try:
        h, w = img.shape[:2]
        sc = 640.0 / max(h, w)
        sm = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
        rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
        r = m['det'].process(rgb)
        if not r.detections:
            return 0.0
        sh, sw = sm.shape[:2]
        bs = [(d.location_data.relative_bounding_box.xmin * sw, d.location_data.relative_bounding_box.ymin * sh,
               d.location_data.relative_bounding_box.width * sw, d.location_data.relative_bounding_box.height * sh)
              for d in r.detections]
        x, y, bw, bh = max(bs, key=lambda b: b[2] * b[3])
        seg = (m['seg'].process(rgb).segmentation_mask > 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(seg, 8)
        cx, cy = int(min(sw - 1, max(0, x + bw / 2))), int(min(sh - 1, max(0, y + bh / 2)))
        i = lab[cy, cx]
        if i == 0:
            return 0.0
        return float(st[i][1] + st[i][3] - (y + bh)) / max(1.0, bh)
    except Exception as e:
        print('[pipe] 量身体出错，当不知道：%s' % str(e)[:120], flush=True)
        return 0.0


def run(img, quad=None, do_border=True):
    """整条链。quad 是千问指的四个角（可能只是包围盒），没有就在整张图上找。"""
    note = []
    src = img
    if quad is not None:
        w, why = crop.warp_quad(img, quad)
        if w is not None:
            src = w
            note.append('vl')
    up, o, fs = orient(src)
    # ⚠️ 2026-11-09 李总传了一张全身照（老人站在石碑前，脸很小）：脸小到 face mesh 认不出，走了 sweep 那条路 ——
    #    那条路是给「桌上一小张照片」用的：按脸挖一块、按头的角度转。全身照被挖成半截，后面又换了背景。
    #    身体从下巴往下还有 4.5 个脸那么长 = 这就是一整张照片里的人，不是桌上的小照片，一个像素都不动。
    #    （12 张回归照最多 3.1，走 sweep 的几张都 ≤ 2.2；那张全身照 7.6）
    if o.startswith('sweep') and quad is None:
        b = body_below_face(src)
        if b >= FULL_BODY:
            print('[pipe] 身体有 %.1f 个脸长，是整张照片里的全身像，不裁不转' % b, flush=True)
            return img, ''
    if o:
        note.append(o)
    # ⚠️ 转过角度的图四周是补出来的边（拉丝），找纸会在拉丝上拟出假四边形（2026-11-03 线上那张）。
    #    转过的图只按脸裁，不找纸。
    rotated = ('tilt' in o) or o.startswith('sweep')
    loc, l = locate(up, fs, allow_paper=not rotated, strict_keep=(quad is None))
    if l:
        note.append(l)
    # 定位之后脸变大了，再定一次方向 —— 45° 放的那张：先前认不出脸，找到纸拉正后还差 90°，这一步补上
    if l:
        up2, o2, fs2 = orient(loc)
        if o2 and (not o2.startswith('orient0') or 'tilt' in o2):
            loc, fs = up2, fs2
            note.append(o2)
            if 'tilt' in o2 or o2.startswith('sweep'):
                # 第二次转过之后同样不能找纸：按脸再裁一次
                loc2, l2 = locate(loc, fs, allow_paper=False, strict_keep=False)
                if l2:
                    loc = loc2
    out = loc
    if do_border and (l or quad is not None):
        bb = crop.strip_border(out)
        if bb:
            sub = out[bb[1]:bb[3], bb[0]:bb[2]]
            if not fs or _keeps_faces(sub, len(fs)):
                out = sub
                note.append('border')
        # 底下还剩一条相纸白边（相册里的照片常见）就再切一刀；切完脸一张都不能少
        yb = crop.trim_bottom_paper(out)
        if yb:
            sub = out[:yb]
            if not fs or _count_faces(sub) >= _count_faces(out):
                out = sub
                note.append('bottom')
    if not note or (len(note) == 1 and note[0].startswith('orient0')):
        return img, ''
    return out, '+'.join(note)
