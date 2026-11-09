# -*- coding: utf-8 -*-
"""
吱吱屋留影 —— 人脸修复服务（CodeFormer）

跑在阿里云函数计算 FC 的 GPU 函数里。小程序不直接连它，链路是：
    小程序 → 腾讯云函数 zzFaceCode → 这个服务 → 返回修好的图

【接口】
    GET  /            健康检查。FC 靠它判断容器起没起来，必须有。
    POST /enhance     {"image": "<base64>", "w": 0.7}  →  {"ok": true, "image": "<base64>"}

【为什么收发都用 base64，不用图片 URL】
    我们的照片在腾讯云，这个服务在阿里云。让阿里云去拉腾讯云的临时链接，
    多一次跨云下载：慢、可能被拒、临时链接还会过期。
    直接把字节传过来最省事，一张老照片也就 1~3 MB。

【w 是什么】
    fidelity_weight，0~1。越大越忠于原图、越少自由发挥。
    ⚠️ 默认 0.7，这是 2026-10-29 在真实老照片上比过四档之后定的。
       老照片这个场景宁可少编一点 —— 模型补出来的睫毛、牙齿、发丝
       原图里一个像素都没有，调低 w 只会让它编得更起劲。

【一个必须记住的事实】
    这个服务做的是【重建】，不是【修清楚】。
    它把 150 像素的糊脸"想"成 512 像素的清晰脸。想得很合理，
    但那是模型的想象。所以产品侧必须保留 AI 标识，
    而且永远不能把结果说成"照片原本的样子"。
"""
import base64
import io
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, '/app/CodeFormer')

from basicsr.utils import img2tensor, tensor2img          # noqa: E402
from basicsr.utils.registry import ARCH_REGISTRY          # noqa: E402
from facelib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: E402
from torchvision.transforms.functional import normalize   # noqa: E402

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = '/app/CodeFormer/weights/CodeFormer/codeformer.pth'

# 进来的图最长边超过这个就先缩小。
# ⚠️ 不是为了省时间 —— 人脸检测在超大图上反而容易漏检，
#    而且显存会炸。1600 对手机翻拍的老照片够用了。
MAX_IN_SIDE = 1600

_net = None
_helper = None


def _load():
    """模型只加载一次。
    ⚠️ 放在模块级而不是请求级：FC 的实例会被复用，热启动时
       这个函数早就跑完了，请求直接拿现成的网络用。
       每个请求都加载一次的话，每张图要多花十几秒。"""
    global _net, _helper
    if _net is not None:
        return
    t0 = time.time()
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']).to(DEVICE)
    ck = torch.load(CKPT, map_location='cpu')['params_ema']
    net.load_state_dict(ck)
    net.eval()
    _net = net

    _helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=DEVICE)
    print('[facecode] 模型加载完成 %.1fs  device=%s' % (time.time() - t0, DEVICE), flush=True)


def _shrink(img):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= MAX_IN_SIDE:
        return img
    s = MAX_IN_SIDE / float(m)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------
# 美颜（2026-11-07）
# 李总：「大家修老照片，也是想自己的过去变好看一点，不要只是还原。」
# 比过 CodeFormer 保真度 w=0.7/0.5/0.3 三档，三张几乎一样 —— 问题不在五官，
# 在【颜色】：整张偏冷偏灰，皮肤发青黄，嘴唇发灰紫。所以美颜做的是修完之后调一次色：
#   ① 灰世界白平衡（只拉回一半，别矫枉过正）
#   ② 往暖一点：R 加一点、B 减一点 —— 皮肤和嘴唇回到正常血色
#   ③ 脸上轻微磨皮：双边滤波按 0~1 混回去，只在 CodeFormer 找到的脸框里做
#   ④ L 通道轻微 CLAHE + 整体提亮一点
# 黑白照片（饱和度很低）跳过 ①② —— 黑白照不许被染上颜色，只做 ③④。
# strength 0~1，默认 0.6。调用方传 beauty:0 关掉。
# ---------------------------------------------------------------
def _is_gray(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) < 18.0


def _beauty(img, face_boxes, strength=0.6):
    k = max(0.0, min(1.0, float(strength)))
    if k <= 0:
        return img, ''
    out = img.astype(np.float32)
    gray = _is_gray(img)
    notes = []
    if not gray:
        # ① 白平衡：每个通道往三通道均值拉，拉一半
        means = out.reshape(-1, 3).mean(axis=0)
        avg = float(means.mean())
        gain = 1.0 + (avg / np.maximum(means, 1.0) - 1.0) * (0.5 * k)
        out = out * gain.reshape(1, 1, 3)
        # ② 暖一点（BGR）
        out[:, :, 2] *= (1.0 + 0.05 * k)
        out[:, :, 0] *= (1.0 - 0.04 * k)
        notes.append('wb+warm')
    out = np.clip(out, 0, 255).astype(np.uint8)
    # ③ 脸上磨皮
    h, w = out.shape[:2]
    for b in face_boxes or []:
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        roi = out[y0:y1, x0:x1]
        d = max(5, int(0.03 * (x1 - x0)) | 1)
        smooth = cv2.bilateralFilter(roi, d, 40, 40)
        # 椭圆羽化遮罩，边缘渐变，不出方框
        m = np.zeros(roi.shape[:2], np.float32)
        cv2.ellipse(m, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(3, (x1 - x0) / 12.0)) * (0.55 * k)
        m = m[:, :, None]
        out[y0:y1, x0:x1] = (roi * (1 - m) + smooth * m).astype(np.uint8)
    if face_boxes:
        notes.append('smooth')
    # ③' 嘴唇补一点血色：只在嘴唇那一圈、只对彩色照片。Lab 的 a 通道（偏红）加一点，饱和一点
    #     2026-11-07 李总看样片：「嘴唇的颜色上面都染了什么，嘴唇的颜色都不好」
    if not gray:
        try:
            import crop as _crop
            fs = _crop.landmarks(out)
            if fs:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                mask = np.zeros(out.shape[:2], np.float32)
                for f in fs:
                    lp = f.get('lips')
                    if lp is not None and len(lp) >= 6:
                        cv2.fillPoly(mask, [lp.astype(np.int32)], 1.0)
                if mask.max() > 0:
                    span = max(3.0, float(np.sqrt(mask.sum())) / 10.0)
                    mask = cv2.GaussianBlur(mask, (0, 0), span)
                    lab[:, :, 1] += mask * (10.0 * k)      # 往红
                    lab[:, :, 2] += mask * (3.0 * k)       # 一点点暖
                    lab[:, :, 0] -= mask * (3.0 * k)       # 稍深一点，不发白
                    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                    notes.append('lips')
        except Exception as le:
            print('[facecode] 嘴唇那步跳过：', le, flush=True)
    # ④ 提亮 + 轻微局部对比
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.0 + 0.8 * k, tileGridSize=(8, 8))
    l2 = clahe.apply(l).astype(np.float32)
    l2 = l * (1 - 0.5 * k) + l2 * (0.5 * k)
    l2 = np.clip(l2 * (1.0 + 0.04 * k) + 4 * k, 0, 255)
    lab[:, :, 0] = l2.astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    notes.append('tone')
    return out, ('beauty' + ('-gray' if gray else '') + ':' + '+'.join(notes))


# ---------------------------------------------------------------
# 调明暗（2026-11-07 李总：「整体色调能不能明亮一点，精神气足一点」，拿 MyHeritage 的对比）
#   放在修脸【之后】：试过放在修脸之前，CodeFormer 会把黑白照的眼睛修出一点蓝色，之后去偏色正好把它洗掉。
#   只动明暗和偏色，不动形状，不上色。彩色照片的颜色一点不动（只有黑白照才去偏色）。
#   调用方传 tone:0 关掉。
# ---------------------------------------------------------------
def _looks_bw(img, line=3.0):
    """黑白照（含泛黄、泛蓝的黑白照）还是彩色照：画面中间那块 a 通道（红绿）的起伏。
    ⚠️ 2026-11-09 用【修之前】的照片判断，线从 3.5 降到 3.0：石碑前那张褪色的彩色照是 3.4，被当成黑白照把暖色洗掉了，
       脸修出来发白、像外国人。手上的黑白照 1.3~1.9，彩色照 4.9 以上。
       判错的代价不一样：彩色当黑白 = 颜色被洗掉；黑白当彩色 = 留一点泛黄。所以往彩色那边偏。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    ca = lab[int(h * .2):int(h * .8), int(w * .2):int(w * .8), 1]
    return float(ca.std()) < line


def _tone(img, k=1.0, bw=None):
    """修完之后调一次明暗：拉开黑白场（高光留余量）+ 中间调提亮 + 局部对比 + 去偏色。不动形状，不加颜色。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.0)
    if hi - lo > 20:
        Ln = np.clip((L - lo) * 245.0 / (hi - lo), 0, 255)
        L = L * (1 - 0.8 * k) + Ln * 0.8 * k
    # 中间调提亮
    L = 255.0 * np.power(np.clip(L / 255.0, 0, 1), 1.0 - 0.10 * k)
    # 高光肩部：205 以上慢慢收，脸和背景不冲成一片白
    sh = 205.0
    over = L > sh
    L[over] = sh + (L[over] - sh) * 0.55
    # 一点局部对比（五官、头发更立体）
    L8 = np.clip(L, 0, 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(L8).astype(np.float32)
    L = L * (1 - 0.35 * k) + cl * 0.35 * k
    # 去偏色【只对黑白照】：看画面中间那块 a 通道（红绿）的起伏 —— 黑白照里人脸、衣服、背景都是同一层色，起伏很小；
    # 彩色照片里皮肤和衣服颜色不一样，起伏大。实测黑白 1.6~2.4，彩色 5.9~6.9，界线取 3.5。
    # 彩色照片的颜色一点不动（2026-11-07 试过整体拉中性，证件照老人的肤色被洗成了灰）。
    h, w = L.shape
    ca = A[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    if (float(ca.std()) < 3.5) if bw is None else bw:
        A = 128 + (A - 128) * (1 - 0.5 * k); B = 128 + (B - 128) * (1 - 0.5 * k)
        A = A - (A.mean() - 128) * 0.6 * k
        B = B - (B.mean() - 128) * 0.6 * k
    lab = np.stack([L, A, B], -1)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------
# 脸上折痕两边的明暗拉平（2026-11-09 李总：「脸上的这个划痕不应该有」）
#   裂纹补掉之后，原照片折痕两边一边亮一边暗，CodeFormer 照着修成一道斜的明暗分界。
#   在脸上把跨过折痕的亮度跳变去掉（梯度清零），泊松重建，纹理留着；再把脸的平均亮度拉回原样。
#   只在「修脸前去裂纹」找到了长折痕时才做。
# ---------------------------------------------------------------
from scipy.fft import dstn as _dstn, idstn as _idstn  # noqa: E402

_FOLD_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
              152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _poisson(f, div):
    """解 ∇²u = div，边界 = f 的边界（Dirichlet）。DST 求解。"""
    h, w = f.shape
    u = f.copy()
    b = div[1:-1, 1:-1].copy()
    b[0, :] -= f[0, 1:-1]; b[-1, :] -= f[-1, 1:-1]; b[:, 0] -= f[1:-1, 0]; b[:, -1] -= f[1:-1, -1]
    bt = _dstn(b, type=1)
    yy = np.arange(1, h - 1)[:, None]; xx = np.arange(1, w - 1)[None, :]
    den = (2 * np.cos(np.pi * yy / (h - 1)) - 2) + (2 * np.cos(np.pi * xx / (w - 1)) - 2)
    u[1:-1, 1:-1] = _idstn(bt / den, type=1)
    return u

def _fold_lines(mask, min_span=0.15):
    h, w = mask.shape[:2]
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lab == i) for i in range(1, n) if max(st[i][2], st[i][3]) >= min_span * max(h, w)]

def _even_fold(img, lines, mesh):
    """lines：修脸前那张图上的折痕（布尔图）。在脸上把跨过折痕的亮度跳变去掉，用泊松重建，纹理留着。"""
    if not lines: return img, 0
    h, w = img.shape[:2]
    r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks: return img, 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    fold = np.zeros((h, w), np.uint8)
    for m in lines:
        fold |= cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    n = 0
    for fl in r.multi_face_landmarks:
        P = np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
        oval = P[_FOLD_OVAL]; fw = float(oval[:, 0].max() - oval[:, 0].min())
        x0, y0 = np.maximum(0, oval.min(0) - fw * 0.15).astype(int); x1, y1 = np.minimum([w, h], oval.max(0) + fw * 0.15).astype(int)
        band = cv2.dilate(fold[y0:y1, x0:x1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(fw / 45)) | 1,) * 2))
        if band.sum() < 50: continue
        f = L[y0:y1, x0:x1].astype(np.float64)
        gx = np.zeros_like(f); gy = np.zeros_like(f)
        gx[:, :-1] = f[:, 1:] - f[:, :-1]; gy[:-1, :] = f[1:, :] - f[:-1, :]
        kill = band > 0
        gx[kill] = 0; gy[kill] = 0
        div = np.zeros_like(f)
        div[:, 1:] += gx[:, 1:] - gx[:, :-1]; div[:, 0] += gx[:, 0]
        div[1:, :] += gy[1:, :] - gy[:-1, :]; div[0, :] += gy[0, :]
        u = _poisson(f, div)
        # 只换到脸上（羽化），框的边上保持原样
        fm = np.zeros((h, w), np.float32); cv2.fillPoly(fm, [oval.astype(np.int32)], 1.0)
        fm = cv2.GaussianBlur(fm, (0, 0), max(3, fw / 25))[y0:y1, x0:x1]
        # 折痕上那几个像素没有纹理（梯度被清零了），把原来的细节加回一点
        hi = f - cv2.GaussianBlur(f, (0, 0), 2.0)
        u = u + hi * (band > 0) * 0.8
        # ⚠️ 从边框积分回来，整张脸会整体偏暗/偏亮（实测暗了 20 级）。按脸上的平均亮度拉回原来那样
        inside = fm > 0.5
        u = u + (float(f[inside].mean()) - float(u[inside].mean()))
        L[y0:y1, x0:x1] = np.clip(f * (1 - fm) + u * fm, 0, 255)
        n += 1
    if not n: return img, 0
    lab[..., 0] = L
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), n


# ---------------------------------------------------------------
# 修脸之前先去折痕、裂纹（2026-11-09 李总拿一张撕坏的照片试：折痕斜穿过脸，修完还在）
#   CodeFormer 为了像本人，会照着输入把裂纹也「修」进脸里。所以破损严重的照片先整张清一遍亮线和霉点，再交给它。
#   五官（眼、眉、嘴）挖掉不碰。只在破损严重时触发。调用方传 precrack:0 关掉。
# ---------------------------------------------------------------
_PC_EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_PC_EYE_R = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_PC_BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_PC_BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_PC_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

def _crack_mask(img, mesh, thr=12):
    """修脸之前：整张照片上细长的亮线（折痕、裂纹）和亮的小霉点。五官（眼、眉、嘴）挖掉不碰。
    返回 (mask, 占比)"""
    h, w = img.shape[:2]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(5, int(round(min(h, w) / 90)) | 1)
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    raw = (th > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros_like(raw)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        span = max(bw, bh)
        if area < 3: continue
        fill = area / float(bw * bh)
        if (span >= k * 2.0 and fill < 0.4) or area <= k * k * 0.6:
            keep[lab == i] = 1
    r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fl in (r.multi_face_landmarks or []):
        P = np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
        fw = float(P[:, 0].max() - P[:, 0].min())
        ex = np.zeros((h, w), np.uint8)
        for poly in (_PC_EYE_L, _PC_EYE_R, _PC_BROW_L, _PC_BROW_R, _PC_LIPS):
            cv2.fillPoly(ex, [cv2.convexHull(P[poly].astype(np.int32))], 1)
        ex = cv2.dilate(ex, np.ones((max(3, int(fw * 0.04)),) * 2, np.uint8))
        keep[ex > 0] = 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return keep, float(keep.mean())

def _preclean_raw(img, mesh):
    out = img; tot = np.zeros(img.shape[:2], np.uint8)
    for thr in (14, 9):
        m, f = _crack_mask(out, mesh, thr)
        if m.sum():
            out = cv2.inpaint(out, m * 255, 4, cv2.INPAINT_TELEA); tot |= m
    return out, tot

def _preclean(img, min_frac=0.05, min_line=0.25):
    """只给破损严重的照片用（折痕穿过脸、满身裂纹）。
    判断：亮线占比 > 5%，或者最长的一条线 > 照片长边的 1/4。2026-11-09 测 14 张：撕坏的那张 18.2% / 0.44，其余都 ≤ 1.1% / 0.08。"""
    import crop as _cropmod
    m0 = _cropmod._mp()
    if not m0:
        return img, '', []
    mesh = m0['mesh']
    m, frac = _crack_mask(img, mesh, 14)
    h, w = img.shape[:2]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    longest = max([max(st[i][2], st[i][3]) for i in range(1, n)] or [0]) / float(max(h, w))
    if frac < min_frac and longest < min_line:
        return img, '', []
    out, tot = _preclean_raw(img, mesh)
    return out, 'precrack%.0f' % (100 * float(tot.mean())), _fold_lines(m)


# ---------------------------------------------------------------
# 衣服和背景收拾干净（2026-11-09 李总：「整张照片还显得脏的地方，把衣服和背景搞干净一点」）
#   ① 背景：人以外（连着脸的那个人），用二次曲面拟合出干净的渐变整块换掉 —— 撕口、霉斑、灰团一起没了
#   ② 衣服：下巴以下、比周围亮一截的白点补掉   ③ 衣服上大块发白的霉斑压回周围的暗度
#   ⚠️ 脸、脖子（跟脸一样亮、连着下巴的那片 + 下颌往下 0.45 脸宽）不动。只在修脸前找到严重裂纹（precrack）时做。调用方传 tidy:0 关掉。
# ---------------------------------------------------------------
def _tidy(img, mesh, seg):
    """破损严重的照片修完之后：① 背景整块重画成干净的渐变 ② 衣服上的浅色霉斑、白点补掉。"""
    h, w = img.shape[:2]
    out = img.copy()
    r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks: return img, ''
    face = np.zeros((h, w), np.uint8); chin = 0; fw = 0; necks = []
    for fl in r.multi_face_landmarks:
        P = np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
        cv2.fillPoly(face, [P[_SC_OVAL].astype(np.int32)], 1)
        chin = max(chin, int(P[152][1])); fw = max(fw, float(P[_SC_OVAL][:, 0].max() - P[_SC_OVAL][:, 0].min()))
        # 脖子：下颌那一圈往下延伸 0.45 个脸宽
        jaw = P[[136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365]]
        d = 0.45 * float(P[_SC_OVAL][:, 0].max() - P[_SC_OVAL][:, 0].min())
        necks.append(np.vstack([jaw, (jaw + [0, d])[::-1]]).astype(np.int32))
    pm = seg.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).segmentation_mask
    pm = cv2.GaussianBlur(pm, (0, 0), 3)
    # ---- ① 背景：人以外的地方，用二次曲面拟合干净背景的亮度和颜色，整块替换 ----
    person = (pm > 0.5).astype(np.uint8)
    # 人要连着脸：撕口那团跟人不连通的碎块，不算人
    n, lab, st, _ = cv2.connectedComponentsWithStats(person, 8)
    keep = np.zeros_like(person)
    for i in range(1, n):
        if (face[lab == i] > 0).any(): keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fw / 25) | 1,) * 2))
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep, 8)
    keep2 = np.zeros_like(keep)
    for i in range(1, n):
        if (face[lab == i] > 0).any(): keep2[lab == i] = 1
    bgm = (keep2 == 0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sample = bgm & (cv2.erode(bgm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    A = np.stack([np.ones_like(xx), xx / w, yy / h, (xx / w) ** 2, (yy / h) ** 2, xx * yy / (w * h)], -1)
    bg = np.zeros_like(out, np.float32)
    for c in range(3):
        v = out[..., c].astype(np.float32)
        # 用中位数抗住残留的脏点：先粗拟，去掉偏差大的，再拟一次
        As, vs = A[sample][::7], v[sample][::7]
        coef = np.linalg.lstsq(As, vs, rcond=None)[0]
        res = np.abs(As @ coef - vs); ok = res < np.percentile(res, 80)
        coef = np.linalg.lstsq(As[ok], vs[ok], rcond=None)[0]
        bg[..., c] = A @ coef
    alpha = cv2.GaussianBlur(keep2.astype(np.float32), (0, 0), max(2, fw / 120))[..., None]
    out = (out.astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
    # ---- ② 衣服：下巴以下、人身上，比周围亮一截的斑块和白点 ----
    cloth = (keep2 > 0) & (face == 0)
    cloth[:chin, :] = False
    # 脖子是皮肤，不是衣服：跟脸差不多亮、又跟下巴连着的那一片，不动
    L0 = cv2.GaussianBlur(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), max(3.0, fw / 40))
    fmed = float(np.median(L0[face > 0]))
    bright = ((L0 > fmed - 35) & (keep2 > 0)).astype(np.uint8)
    seed = cv2.dilate(face, np.ones((int(fw / 15) | 1,) * 2, np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    skin = np.zeros_like(bright)
    for i in range(1, n):
        if (seed[lab == i] > 0).any(): skin[lab == i] = 1
    neck = np.zeros((h, w), np.uint8)
    for q in necks: cv2.fillPoly(neck, [q], 1)
    skin |= (neck > 0) & (L0 > fmed - 70)
    skin = cv2.dilate(skin, np.ones((int(fw / 20) | 1,) * 2, np.uint8))
    cloth &= (skin == 0)
    L = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(15, int(fw / 6) | 1)
    med = cv2.medianBlur(L, min(k, 99) | 1)
    spot = ((L.astype(np.int16) - med.astype(np.int16)) > 12) & cloth
    spot = cv2.morphologyEx(spot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    spot = cv2.dilate(spot, np.ones((5, 5), np.uint8))
    if spot.sum():
        out = cv2.inpaint(out, spot * 255, 6, cv2.INPAINT_TELEA)
    # ③ 衣服上大块的发白霉斑：低频亮度比「周围偏暗的那层」高出来的部分压下去；衣服本身的明暗、纹理留着
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    sig = max(4.0, fw / 22)
    low = cv2.GaussianBlur(L, (0, 0), sig)
    win = int(fw / 3) | 1
    base = cv2.GaussianBlur(cv2.erode(low, np.ones((win, win), np.uint8)), (0, 0), fw / 6)
    excess = np.clip(low - base - 6, 0, None)
    cm = cv2.GaussianBlur(cloth.astype(np.float32), (0, 0), fw / 30)
    lab[..., 0] = np.clip(L - excess * 0.8 * cm, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, 'tidy(bg+cloth%.1f%%)' % (100 * spot.mean())


# ---------------------------------------------------------------
# 去划痕（2026-11-08 李总：「照片上的划痕感能去掉吗？特别头发那一块」）
#   CodeFormer 只重画脸，头发、背景上的白色划痕原样留着。这里找细长的亮线（形态学 top-hat），用周围补上（inpaint）。
#   ⚠️ 只管下巴以上、脸的轮廓以外（脸已经修干净了；衣服的条纹、格子会被误认）。只有划痕很密才动。调用方传 descratch:0 关掉。
# ---------------------------------------------------------------
_SC_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def _scratch_mask(img, mesh=None, thr=14):
    """细长的亮线（划痕、折痕、霉点连成的线）→ 掩膜。脸的轮廓里面不算（脸已经由 CodeFormer 重画过）。"""
    h, w = img.shape[:2]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(5, int(round(min(h, w) / 180)) | 1)              # 1400px 的图大约 7~9px
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (th > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        span = max(bw, bh)
        if area < 6: continue
        # 细长：外接框的长边够长，而且填充率低（线），或者面积很小的亮点（霉点）
        if (span >= k * 1.6 and area / float(bw * bh) < 0.5) or area <= k * 1.5:
            keep[lab == i] = 1
    if mesh is None:
        return np.zeros_like(keep)
    r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks:
        return np.zeros_like(keep)      # 认不出脸就不动：不知道哪儿是头发、哪儿是衣服
    chin = 0
    if True:
        for fl in (r.multi_face_landmarks or []):
            pts = np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
            oval = pts[_SC_OVAL].astype(np.int32)
            fm = np.zeros((h, w), np.uint8); cv2.fillPoly(fm, [oval], 1)
            fm = cv2.dilate(fm, np.ones((k, k), np.uint8))
            keep[fm > 0] = 0
            chin = max(chin, int(pts[152][1]))
    # ⚠️ 只管下巴以上（头发、背景）。下面是衣服：毛衣条纹、格子衬衫也是「深底上的细亮线」，会被当成划痕抹掉
    keep[chin:, :] = 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return keep

def _descratch_raw(img, mesh=None):
    """两遍：先去明显的，再去剩下的淡印子"""
    total = np.zeros(img.shape[:2], np.uint8)
    out = img
    for thr in (14, 9):
        m = _scratch_mask(out, mesh, thr)
        if m.sum() == 0: continue
        out = cv2.inpaint(out, m * 255, 5, cv2.INPAINT_TELEA)
        total |= m
    return out, total

def _descratch(img, min_frac=0.025):
    """只有划痕很密的时候才动（2026-11-08：证件照里老人的白头发会被当成划痕抹掉 —— 等于改了人的样子）。
    7 张测试照只有一张（头发上满是白道的那张）超过 2.5%，其余都在 1.4% 以下。"""
    import crop as _cropmod
    m0 = _cropmod._mp()
    if not m0:
        return img, ''
    out, m = _descratch_raw(img, m0['mesh'])
    frac = float(m.sum()) / float(m.size)
    if frac < min_frac:
        return img, ''
    return out, 'descratch%.1f' % (frac * 100)


# ---------------------------------------------------------------
# 脸部清理（2026-11-08 李总：「左边的脸弄干净一些，整体的脸部弄干净一些」）
#   CodeFormer 修完，破损严重的照片脸上还会留原来划痕的印子和霉点。这里只补细长的划痕和孤立的小点，
#   不磨皮（试过混 0.35 的双边滤波，像蜡像 —— 李总之前就否过），眼、眉、鼻孔、嘴、眉毛以上都不碰。
#   ⚠️ 只跟着去划痕一起触发（照片破损严重才做）。完好的照片上，皱纹、痣是这个人本来的样子。
# ---------------------------------------------------------------
_FC_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
_FC_EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_FC_EYE_R = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_FC_BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_FC_BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_FC_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_FC_NOSE_LOW = [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64]

def _face_clean_raw(img, mesh, smooth=0.0):
    """破损严重的照片：脸上皮肤的划痕、斑点补掉，再轻轻匀一下。眼、眉、鼻孔、嘴不碰。"""
    h, w = img.shape[:2]
    r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks:
        return img, 0
    out = img.copy()
    n = 0
    for fl in r.multi_face_landmarks:
        P = np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32)
        fw = float(P[_FC_OVAL][:, 0].max() - P[_FC_OVAL][:, 0].min())
        if fw < 60: continue
        skin = np.zeros((h, w), np.uint8)
        cv2.fillPoly(skin, [P[_FC_OVAL].astype(np.int32)], 255)
        skin = cv2.erode(skin, np.ones((max(3, int(fw * 0.03)),) * 2, np.uint8))
        ex = np.zeros((h, w), np.uint8)
        for poly in (_FC_EYE_L, _FC_EYE_R, _FC_BROW_L, _FC_BROW_R, _FC_LIPS):
            cv2.fillPoly(ex, [cv2.convexHull(P[poly].astype(np.int32))], 255)
        cv2.fillPoly(ex, [cv2.convexHull(P[_FC_NOSE_LOW].astype(np.int32))], 255)
        ex = cv2.dilate(ex, np.ones((max(3, int(fw * 0.045)),) * 2, np.uint8))
        skin[ex > 0] = 0
        # 眉毛以上：额头常有刘海（深色细线），会被当成划痕涂成一块灰 —— 所以额头只补【孤立的小白点】（霉点、相纸掉的点），
        # 深色的线、成片的都不碰（2026-11-09 李总圈出撕坏那张额头、发际线上剩的两三个白点）
        brow_y = int(max(P[_FC_BROW_L][:, 1].max(), P[_FC_BROW_R][:, 1].max()) + fw * 0.02)
        fore = skin.copy(); fore[brow_y:, :] = 0
        skin[:brow_y, :] = 0
        L = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0]
        k = max(5, int(fw / 28) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dark = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, ker)
        bright = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, ker)
        raw = (((dark > 16) | (bright > 16)) & (skin > 0)).astype(np.uint8)
        fb = ((bright > 16) & (fore > 0)).astype(np.uint8)
        cnt2, lab2, st2, _ = cv2.connectedComponentsWithStats(fb, 8)
        for i in range(1, cnt2):
            x, y, bw, bh, area = st2[i]
            if 3 <= area <= int(k * k * 0.5) and max(bw, bh) <= k:
                raw[lab2 == i] = 1
        # 只留两种：细长的（划痕）和孤立的小点（霉点）。皮肤本来的纹理是一片一片的，不留
        cnt, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8)
        m = np.zeros_like(raw)
        for i in range(1, cnt):
            x, y, bw, bh, area = st[i]
            span = max(bw, bh)
            if area < 4: continue
            if (span >= k * 1.5 and area / float(bw * bh) < 0.45) or area <= int(k * k * 0.5):
                m[lab == i] = 1
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        if m.sum():
            out = cv2.inpaint(out, m * 255, 4, cv2.INPAINT_TELEA)
        if smooth <= 0:
            n += 1
            continue
        bl = cv2.bilateralFilter(out, max(5, int(fw / 45) | 1), 18, 9)
        a = cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (0, 0), max(2, fw / 60)) * smooth
        a = a[..., None]
        out = (out.astype(np.float32) * (1 - a) + bl.astype(np.float32) * a).astype(np.uint8)
        n += 1
    return out, n


# ---------------------------------------------------------------
# 眼珠修正（2026-11-08）
#   褪色严重的老照片，眼睛在原图里发白，CodeFormer 会把眼珠修成浅灰绿 / 灰蓝 —— 看着像外国人的眼睛
#   （李总之前说过上色「会把人的眼睛也变色」，这是同一类问题）。
#   只动虹膜那个圆，只修发绿、发蓝的；棕色眼珠不碰。12 张测试照里只改了两张（都是该改的）。
#   调用方传 iris:0 关掉。
# ---------------------------------------------------------------
def _mesh_all(img, m0):
    """每张脸的 478 个点（原图坐标）。
    ⚠️ 2026-11-09 石碑前那张全身照：脸只占画面一小块，face mesh 在整张图上认不出 → 眼珠修正整个跳过，蓝眼睛留在了脸上。
       整张图上认不出的脸，用 BlazeFace（小脸也认得）找到位置，把那一块放大到 512 再认。"""
    h, w = img.shape[:2]
    out = []
    r = m0['mesh'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fl in (r.multi_face_landmarks or []):
        out.append(np.array([(q.x * w, q.y * h) for q in fl.landmark], np.float32))
    if 'det' not in m0:
        return out
    d = m0['det'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for det in (d.detections or []):
        b = det.location_data.relative_bounding_box
        bx, by, bw, bh = b.xmin * w, b.ymin * h, b.width * w, b.height * h
        cx, cy = bx + bw / 2, by + bh / 2
        if any(P[:, 0].min() <= cx <= P[:, 0].max() and P[:, 1].min() <= cy <= P[:, 1].max() for P in out):
            continue   # 这张脸整张图上已经认出来了
        x0, y0 = int(max(0, bx - bw)), int(max(0, by - bh))
        x1, y1 = int(min(w, bx + 2 * bw)), int(min(h, by + 2 * bh))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        roi = img[y0:y1, x0:x1]
        sc = 512.0 / max(roi.shape[:2])
        big = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        rr = m0['mesh'].process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        for fl in (rr.multi_face_landmarks or [])[:1]:
            bh2, bw2 = big.shape[:2]
            out.append(np.array([(q.x * bw2 / sc + x0, q.y * bh2 / sc + y0) for q in fl.landmark], np.float32))
    return out


def _iris_from_orig(result, orig, m0):
    """彩色照片：眼珠的颜色用【原照片】那个位置的颜色，修脸模型只管把眼睛画清楚，不许它编颜色。
    ⚠️ 2026-11-09 石碑前那张全身照：脸太糊，CodeFormer 把眼珠画成了蓝色。原照片那里没有蓝色，修完也不该有。
       本来就是棕色眼睛的，颜色照旧是棕色。只动颜色（LAB 的 a、b），明暗和细节不动。黑白照片不走这一步。"""
    h, w = result.shape[:2]
    pts = _mesh_all(result, m0)
    if not pts:
        return result, ''
    o = cv2.resize(orig, (w, h), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    lo = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    n = 0
    for P in pts:
        for c, ring in ((468, [469, 470, 471, 472]), (473, [474, 475, 476, 477])):
            rad = float(np.mean(np.linalg.norm(P[ring] - P[c], axis=1)))
            if rad < 2:
                continue
            cv2.circle(mask, (int(round(P[c][0])), int(round(P[c][1]))), int(round(rad * 1.15)), 1.0, -1)
            n += 1
    if not n:
        return result, ''
    rad_mean = max(2.0, float(np.sqrt(mask.sum() / (n * np.pi))))
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, rad_mean * 0.25))[..., None]
    # 原照片的颜色先抹匀一点（原照片本来就糊，抹匀只是去噪）
    ab = cv2.GaussianBlur(lo[..., 1:], (0, 0), max(1.0, rad_mean * 0.3))
    lab[..., 1:] = lab[..., 1:] * (1 - mask) + ab * mask
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, 'irisc%d' % n


def _iris_fix(img, max_ratio=0.36):
    """眼珠太浅（修复模型把褪色老照片的眼睛修成浅灰/灰绿）时，把眼珠压回深色、去掉颜色。
    只动虹膜那个圆（MediaPipe 468~477 号点），眼白、眼皮、高光都不动。
    判断标准：虹膜亮度 / 鼻梁皮肤亮度 > max_ratio 才动（中国人的眼珠在照片里一般比皮肤暗得多）。"""
    h, w = img.shape[:2]
    import crop as _cropmod   # ⚠️ 不能直接写 crop：下面有个 Flask 路由函数也叫 crop，会把模块名盖掉
    m0 = _cropmod._mp()
    if not m0:
        return img, ''
    pts = _mesh_all(img, m0)
    if not pts:
        return img, ''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    n = 0
    for P in pts:
        nose = P[6]   # 两眼之间的鼻梁
        sk = L[max(0, int(nose[1]) - 6):int(nose[1]) + 6, max(0, int(nose[0]) - 6):int(nose[0]) + 6]
        if sk.size == 0: continue
        skin = float(np.median(sk))
        for c, ring in ((468, [469, 470, 471, 472]), (473, [474, 475, 476, 477])):
            cx, cy = P[c]
            rad = float(np.mean(np.linalg.norm(P[ring] - P[c], axis=1)))
            if rad < 2: continue
            m = np.zeros((h, w), np.float32)
            cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(rad))), 1.0, -1)
            # 眼皮会盖住虹膜上下沿：只留比皮肤暗的像素（眼皮、眼白都比虹膜亮），再羽化
            inside = m > 0
            if inside.sum() < 6: continue
            iris_med = float(np.median(L[inside]))
            ratio = iris_med / max(skin, 1.0)
            if ratio <= max_ratio: continue
            # ⚠️ 只修发绿、发蓝的眼珠。棕色眼珠（a 偏红）本来就对，不碰 —— 第一版连彩色照片的棕眼睛一起压黑了
            ia = float(np.median(A[inside])) - 128; ib = float(np.median(B[inside])) - 128
            sb = float(np.median(B[max(0, int(nose[1]) - 6):int(nose[1]) + 6, max(0, int(nose[0]) - 6):int(nose[0]) + 6])) - 128
            if not (ia < -0.8 or (ib < sb - 6 and ia <= 1.0)): continue
            target = skin * 0.32
            k = min(1.0, target / max(iris_med, 1.0))
            keep = (L < skin * 0.92).astype(np.float32)       # 高光、眼白不压
            mm = cv2.GaussianBlur(m * keep, (0, 0), max(1.0, rad * 0.18))
            L[:] = L * (1 - mm) + (L * k) * mm
            A[:] = A * (1 - mm) + (128 + (A - 128) * 0.25) * mm
            B[:] = B * (1 - mm) + (128 + (B - 128) * 0.25) * mm
            n += 1
    if not n:
        return img, ''
    out = cv2.cvtColor(np.clip(np.stack([L, A, B], -1), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, 'iris%d' % n


# ⚠️ 摆正/裁剪/去花边/人脸定位全在 crop.py 里。
#    它只依赖 OpenCV，不需要 GPU，也不需要 PyTorch ——
#    所以同一个文件可以单独跑在腾讯云 SCF 的轻量 Python 函数上，
#    不用为了裁张照片养一个 7GB 的容器镜像。
#    这里 import 它，GPU 这条路一行都不用改。
from crop import beautify as crop_beautify  # noqa: E402
import beauty  # noqa: E402  美颜：瘦脸 / 大眼 / 磨皮美白（2026-11-07）
import pipe  # noqa: E402  一条直线的预处理（定方向 → 定位 → 裁 → 验证），和腾讯云 SCF 上跑的是同一份


@app.get('/')
def health():
    """FC 靠这个判断容器活没活。
    ⚠️ 这里【不要】加载模型 —— 探活要快，加载放到第一次真请求里。"""
    return jsonify(ok=True, device=DEVICE, cuda=torch.cuda.is_available())


@app.post('/crop')
def crop():
    """只裁不修。给调用方一个单独试裁剪的口子。

    ⚠️ 这个接口【不加载模型】，所以又快又不烧显存。
       上线前拿一批真实用户照片批量跑这个，就能量出命中率，
       不用每张都走一遍 GPU。
    """
    if not _auth_ok():
        return jsonify(ok=False, msg='没有权限'), 403
    try:
        body = request.get_json(force=True, silent=True) or {}
        b64 = body.get('image') or ''
        if not b64:
            return jsonify(ok=False, msg='没给图片'), 400
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(ok=False, msg='图片解不开'), 400
        h, w = img.shape[:2]

        out, why = pipe.run(img, do_border=body.get('border', True))
        # 没裁到就一个像素不碰（2026-11-02 的教训：没裁到还换背景，整张图糊成灰蓝）
        if why and body.get('beautify', True):
            out, note = crop_beautify(out)
            if note:
                why = why + '+' + note
        if not why:
            # ⚠️ 不是故障。调用方据此决定：让用户自己框。
            return jsonify(ok=True, fix='', msg='没找到照片的边，让用户自己框',
                           w=w, h=h), 200

        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            return jsonify(ok=False, msg='结果图编码失败'), 500
        return jsonify(ok=True, fix=why,
                       image=base64.b64encode(buf.tobytes()).decode('ascii'),
                       w=out.shape[1], h=out.shape[0],
                       ratio=round(float(out.shape[0] * out.shape[1]) / float(w * h), 3))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300]), 500


def _auth_ok():
    """⚠️ 这个接口挂在公网上，而且每次调用都烧 GPU 的钱。
       不加校验等于把你的信用卡放在路边。
       FC 的 HTTP 触发器可以配「无认证」图省事，所以这一道闸必须在应用层自己把。
       部署时在函数的环境变量里设 ZZ_SHARED_TOKEN（FC_ 开头是阿里保留前缀，不让用），
       腾讯云那边的 zzFaceCode 带同一个值。"""
    # ⚠️ 2026-11-03：阿里云 FC 不让用 FC_ 开头的环境变量名（保留前缀），改叫 ZZ_SHARED_TOKEN。
    #    本地调试还认旧名字。
    want = os.environ.get('ZZ_SHARED_TOKEN', '') or os.environ.get('FC_SHARED_TOKEN', '')
    if not want:
        return True   # 没设就不校验，方便本地调试；线上一定要设
    return request.headers.get('X-ZZ-Token', '') == want


@app.post('/enhance')
def enhance():
    t0 = time.time()
    if not _auth_ok():
        return jsonify(ok=False, msg='没有权限'), 403
    try:
        body = request.get_json(force=True, silent=True) or {}
        b64 = body.get('image') or ''
        if not b64:
            return jsonify(ok=False, msg='没给图片'), 400
        try:
            w = float(body.get('w', 0.7))
        except Exception:
            w = 0.7
        w = min(1.0, max(0.0, w))

        raw = base64.b64decode(b64)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(ok=False, msg='图片解不开'), 400

        # ⚠️ 摆正和裁剪要在修脸【之前】。
        #    顺序反了的话，CodeFormer 会在整张手机照片上找脸 ——
        #    照片只占画面一半时，脸相对更小，检测更容易漏，
        #    而且修完再裁等于白修了外面那一半的算力。
        # ⚠️ 默认开。调用方传 crop:false 可以关掉（比如用户已经自己框过了）。
        # ⚠️ 变量不能叫 cropped —— 下面 `for cropped in _helper.cropped_faces`
        #    会把它覆盖成一张人脸图，返回时 cropped 就不是布尔值了。
        #    2026-11-01 差点发出去。
        did_fix = ''
        if body.get('crop', True):
            img, did_fix = pipe.run(img, do_border=body.get('border', True))
            if not did_fix:
                print('[facecode] 没自动处理，原样往下走', flush=True)

        # ⚠️ 构图 + 换背景放在修脸【之前】、裁剪【之后】。
        #    之前：CodeFormer 贴回脸时要的是最终画面，先换背景再修脸，
        #          脸边缘的柔化只做一次，不会出现"修过的脸贴在旧背景上再被抠"的双重接缝。
        #    之后：第一层把相框、墙、网纹裁掉了，分割才不会把相框当成人。
        #    2026-11-01 李总：「人像很明显是没有摆正的」。量过头是正的，
        #    不正的是画面 —— 所以这一层解决的是构图，不是旋转。
        if did_fix and body.get('beautify', True):
            img, note = crop_beautify(img)
            if note:
                did_fix = did_fix + '+' + note

        img = _shrink(img)

        pre_note = ''
        fold_lines = []
        if body.get('precrack', True):
            try:
                img, pre_note, fold_lines = _preclean(img)
            except Exception as pe:
                print('[facecode] 修脸前去裂纹出错，跳过：', pe, flush=True)

        # 黑白还是彩色，看【修之前】的照片（修完的脸颜色会变，判断就不稳）
        bw_in = _looks_bw(img)

        _load()
        _helper.clean_all()
        _helper.read_image(img)
        n = _helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        if not n:
            # ⚠️ 检测不到脸就原样退回，并且明说。
            #    绝不能默默返回原图让调用方以为修过了 ——
            #    那样用户会对着一张没修的照片说"这就是修完的效果"。
            return jsonify(ok=False, msg='这张照片里没找到人脸', faces=0), 200
        _helper.align_warp_face()

        for cropped in _helper.cropped_faces:
            t = img2tensor(cropped / 255., bgr2rgb=True, float32=True)
            normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            t = t.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = _net(t, w=w, adain=True)[0]
                restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
            del out
            _helper.add_restored_face(restored.astype('uint8'), cropped)

        _helper.get_inverse_affine(None)
        result = _helper.paste_faces_to_input_image(upsample_img=None)

        # 彩色照片：眼珠颜色回到原照片（修脸模型在糊脸上会编出蓝眼睛）
        irisc_note = ''
        if not bw_in and body.get('iris', True):
            try:
                import crop as _cropmod
                _m3 = _cropmod._mp()
                if _m3:
                    result, irisc_note = _iris_from_orig(result, img, _m3)
            except Exception as ie2:
                print('[facecode] 眼珠颜色还原出错，跳过：', ie2, flush=True)

        # 美颜（2026-11-07 李总：照美图秀秀那样，脸型圆润一点、皮肤亮白一点、五官微调）
        #   瘦脸 + 大眼 + 皮肤磨皮美白 + 彩色照片嘴唇补血色，每张脸都做。见 beauty.py。
        #   默认 0.7，调用方传 beauty:0 关掉。出错就跳过，修复结果照样返回。
        beauty_note = ''
        try:
            bk = float(body.get('beauty', 0))
        except Exception:
            bk = 0   # 2026-11-07 美颜不做了，默认关（调用方不传就是关）
        if bk > 0:
            try:
                result, beauty_note = beauty.beautify(result, bk)
            except Exception as be:
                print('[facecode] 美颜出错，跳过：', be, flush=True)

        tone_note = ''
        try:
            tk = float(body.get('tone', 1.0))
        except Exception:
            tk = 1.0
        if tk > 0:
            try:
                result = _tone(result, min(1.0, tk), bw=bw_in)
                tone_note = 'tone'
            except Exception as te:
                print('[facecode] 调明暗出错，跳过：', te, flush=True)

        scratch_note = ''
        if body.get('descratch', True):
            try:
                result, scratch_note = _descratch(result)
            except Exception as se:
                print('[facecode] 去划痕出错，跳过：', se, flush=True)
        if scratch_note:
            try:
                import crop as _cropmod
                _m0 = _cropmod._mp()
                if _m0:
                    result, _nf = _face_clean_raw(result, _m0['mesh'])
                    if _nf: scratch_note += '+face%d' % _nf
            except Exception as fe:
                print('[facecode] 脸部清理出错，跳过：', fe, flush=True)

        if fold_lines:
            try:
                import crop as _cropmod
                _m1 = _cropmod._mp()
                if _m1:
                    result, _nf2 = _even_fold(result, fold_lines, _m1['mesh'])
                    if _nf2: pre_note += '+fold%d' % _nf2
            except Exception as fe2:
                print('[facecode] 折痕明暗拉平出错，跳过：', fe2, flush=True)

        tidy_note = ''
        if pre_note and body.get('tidy', True):
            try:
                import crop as _cropmod
                _m2 = _cropmod._mp()
                if _m2:
                    result, tidy_note = _tidy(result, _m2['mesh'], _m2['seg'])
            except Exception as te2:
                print('[facecode] 衣服背景清理出错，跳过：', te2, flush=True)

        iris_note = ''
        if body.get('iris', True):
            try:
                result, iris_note = _iris_fix(result)
            except Exception as ie:
                print('[facecode] 眼珠修正出错，跳过：', ie, flush=True)

        ok, buf = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            return jsonify(ok=False, msg='结果图编码失败'), 500

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

        # ⚠️ 顺手给一张【修好之后】的真灰度图。
        #    百度上色只吃黑白：褪色发黄的老照片在它眼里「已经是彩色的」，它不会动。
        #    之前裁剪那一步出过一张灰度图，但那是修脸【之前】的 —— 拿它去上色等于把修好的脸扔了。
        #    所以修完这里再出一张，调用方上色就喂这张。2026-11-04。
        gray_b64 = ''
        if body.get('gray', True):
            g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            okg, bufg = cv2.imencode('.jpg', g, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if okg:
                gray_b64 = base64.b64encode(bufg.tobytes()).decode('ascii')

        return jsonify(
            ok=True,
            image=base64.b64encode(buf.tobytes()).decode('ascii'),
            gray=gray_b64,
            faces=n,
            w=w,
            # ⚠️ 动没动、怎么动的，原样带回去。
            #    调用方要能告诉用户「我顺手把相框和墙裁掉了、摆正了」——
            #    悄悄改了用户的照片而不说，比不改更糟。
            #    fix 的取值：'' 没动 / 'crop' 只裁了 / 'deskew -2.6°+border' 摆正并去了花边
            #              / 后面还可能接 '+recomp'（重新构图）'+bg'（换了背景）
            fix=did_fix,
            beauty=beauty_note,
            tone=tone_note,
            iris='+'.join([x for x in (irisc_note, iris_note) if x]),
            descratch=scratch_note,
            precrack=pre_note,
            tidy=tidy_note,
            size=[img.shape[1], img.shape[0]],
            ms=int((time.time() - t0) * 1000))
    except Exception as e:
        # ⚠️ 把真实堆栈带回去。云函数里看日志要押金，
        #    调用方拿不到原因就只能靠猜 —— 这个项目在这上面已经浪费过六天。
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300], trace=traceback.format_exc()[-800:]), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('FC_SERVER_PORT', 9000)))
