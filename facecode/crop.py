# -*- coding: utf-8 -*-
"""
老照片自动预处理：摆正 / 裁剪 / 去花边 / 人脸定位。

【为什么单独一个文件】
    2026-11-01 发现这几件事【不需要 GPU，也不需要 PyTorch】，只要 OpenCV。
    原来都塞在 facecode/app.py 里，等于跟 CodeFormer 绑在一起，
    而那个要 7GB 的容器镜像和一个每月 117 元的镜像仓库。
    拆出来之后，这个文件可以单独跑在腾讯云 SCF 的 Python 函数上，
    按次计费、几乎免费，而且照片本来就存在腾讯云 —— 少一次跨云传图。

    app.py 仍然 import 它，所以 GPU 那条路不受影响。

【为什么必须有这一套】
    2026-11-01 李总发来真实用户的第一张结果：一张糊脸 + 三分之二的灰墙。
    用户拍的是贴在墙上的照片，我们把整张手机照片拿去修了。
    「用户一看，就是简单上个色，根本不可能转发给朋友，下次也不会再打开。」
    老人不会框，也看不懂为什么要框 —— 最好的框选界面是不出现。

【两条腿 + 一条退路，顺序不能换】
    ① deskew      找四个角做透视变换，摆正 + 裁剪一步到位
                  要求：照片四边完整、背景有对比
    ② face_crop   ①失败时用。只要检测到脸，就按人像构图裁
                  要求：有人脸
    ③ 都失败      原样返回，让用户自己框（现有流程）

    ① 更准（它处理的是整张照片的真实边界），② 更稳（只要有脸就行）。

【实测（2026-11-01，CPU，三个真实样本 + 一个人造样本）】
    墙上的花边照片 810x1400  → ① deskew -2.6° + 去花边 → 558x655
    人为扭歪 26°   1051x1296 → ① deskew 26.4° + 去花边 → 550x682
    身份证上的证件照 1500x2000 → ①失败(area 0.03) → ② 人脸定位 → 587x747
    已处理好的图   756x499   → 全部闸门拦住，不动

【⚠️ 顺手解决的一个合规问题】
    真实用户会拍身份证 —— 家里老人最清楚的一张脸常常就在证件上。
    身份证号是《个人信息保护法》里的敏感个人信息，而我们会把图转发给
    百度/阿里云去上色。
    人脸定位裁出来的是人像构图，号码和姓名天然落在框外 ——
    裁剪发生在上传之后、送给第三方之前，号码压根到不了外面。
    这不是刻意设计的，但既然成立，就不能在以后调参时把它弄丢：
    改 FACE_SCALE 之前先拿身份证那张样本验一遍。
"""
import cv2
import numpy as np
import traceback

# 文档检测的两道面积闸（占全图比例）
# ⚠️ 下限 0.30 不是拟合出来的，有实际道理：
#    一张真正嵌在桌面/墙面里的照片，在画面里占比不会低于三成 ——
#    低于三成说明用户拍得太远，照片本身的像素也不够修了。
#    而抓到局部（一块衣服、一只眼睛）时，占比往往就是一两成。
#    2026-11-01 设 0.12 时，一张已经裁好的证件照被当成"周围有废边"，
#    切掉了 78%。
# ⚠️ 上限 0.85：裁不掉多少不值得冒险。一个 84% 的轮廓加完边距变成 88%
#    还被裁出去过 —— 所以面积要在【加完边距之后】判。
MIN_AREA, MAX_AREA = 0.30, 0.85

# 人脸定位的构图参数。
# ⚠️ 2.2 不是随手定的：证件照里人脸框 356px、照片本体高 730px，比值约 2.05。
#    取 2.2 略松一点，再交给 strip_border 收边。
#    改大会裁到身份证号，改小会切掉下巴。
FACE_SCALE = 2.2
FACE_AR = 0.75      # 3:4 竖版
FACE_EYE = 0.42     # 人脸中心落在裁剪框内这个高度比例上（偏上才像人像）

_CASCADE = None


def _cascade():
    """Haar 人脸检测器。opencv 自带，不用额外下载权重 —— 这对
    「能不能塞进一个轻量云函数」是决定性的。"""
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _CASCADE


def _biggest_contour(img):
    """找画面里最大的那一圈边缘，连它外接矩形占全图的比例一起返回。

    ⚠️ dilate 的第三个【位置】参数是 anchor 不是 iterations。
       写成 cv2.dilate(ed, k, 2) 的话膨胀根本没发生，轮廓是碎的，
       面积算出来只有 1%，整条路直接判失败。2026-11-01 踩过。
    """
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ed = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 90)
    ed = cv2.dilate(ed, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c)
    return c, float(ww * hh) / float(w * h)


def _order_pts(p):
    """四个点排成 左上 / 右上 / 右下 / 左下"""
    r = np.zeros((4, 2), dtype='float32')
    s = p.sum(axis=1)
    d = np.diff(p, axis=1)
    r[0] = p[np.argmin(s)]
    r[2] = p[np.argmax(s)]
    r[1] = p[np.argmin(d)]
    r[3] = p[np.argmax(d)]
    return r


def deskew(img):
    """找照片的四个角，透视拉正 —— 顺带把裁剪也做了。

    【⚠️ 必须先取凸包再拟合四边形】
        老照片常有锯齿花边。直接对轮廓 approxPolyDP 拟合不出四个角
        （波浪线上找不到拐点），2026-11-01 实测三张全部 not-a-quad。
        先 convexHull 把锯齿磨平，四个角立刻就出来了。

    返回 (正过来的图, 说明) 或 (None, 原因)。
    """
    try:
        c, area = _biggest_contour(img)
        if c is None:
            return None, 'no-contour'
        if area < MIN_AREA or area > MAX_AREA:
            return None, 'area %.2f' % area

        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        # ⚠️ eps 只试到 0.05。
        #    2026-11-01：一张【已经处理好的正照片】要 eps=0.08 这么粗的近似
        #    才凑得出四个角，然后被判成歪 18.4° —— 转下去这张就毁了。
        #    「需要很粗的近似才能凑出四边形」本身就是「它不是四边形」的信号。
        for eps in (0.02, 0.03, 0.04, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) != 4:
                continue
            q = _order_pts(ap.reshape(4, 2).astype('float32'))
            tl, tr, br, bl = q
            top = np.linalg.norm(tr - tl)
            bot = np.linalg.norm(br - bl)
            lft = np.linalg.norm(bl - tl)
            rgt = np.linalg.norm(br - tr)
            if min(top, bot, lft, rgt) < 1:
                return None, 'degenerate'
            # ⚠️ 对边长度比 —— 区分「真·斜着拍的矩形」和「乱凑的四边形」
            #    最有效的一个数。实测：真歪 26° 的 0.65，真实原图 0.92，
            #    误判那张只有 0.42。
            #    ⚠️ 0.55 是 4 个样本拟合出来的，不是验出来的，上线后看数据再调。
            #    ⚠️ 风险不对称，所以宁可卡紧：
            #       漏判 → 退回只裁，照片还是好的，只是歪着；
            #       误判 → 照片被转歪，毁了。
            par = (min(top, bot) / max(top, bot)) * (min(lft, rgt) / max(lft, rgt))
            if par < 0.55:
                return None, 'not-rect %.2f' % par

            W = int(max(bot, top))
            H = int(max(rgt, lft))
            if W < 80 or H < 80:
                return None, 'too-small'
            ar = W / float(H)
            if ar < 0.3 or ar > 3.3:
                return None, 'ratio %.2f' % ar
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], 'float32')
            out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (W, H))
            ang = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
            return out, 'deskew %.1f°' % ang
        return None, 'not-a-quad'
    except Exception:
        traceback.print_exc()
        return None, 'error'


def auto_crop(img, pad=0.01):
    """只裁不摆正。deskew 拟合不出四边形时的降级方案。"""
    h, w = img.shape[:2]
    try:
        c, _ = _biggest_contour(img)
        if c is None:
            return None
        x, y, ww, hh = cv2.boundingRect(c)
        # ⚠️ 先加边距，再量面积，最后才判闸。
        #    反过来会漏网：2026-11-01 一个 84% 的轮廓通过了 0.85 的闸，
        #    加完 1% 边距变成 88% 才裁出去 —— 判的和返回的不是同一个东西。
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + ww + px), min(h, y + hh + py)
        area = float((x1 - x0) * (y1 - y0)) / float(w * h)
        if area < MIN_AREA or area > MAX_AREA:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        traceback.print_exc()
        return None


def find_faces(img, min_side=80):
    """返回 [(x, y, w, h)]，没有就空列表。"""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fs = _cascade().detectMultiScale(g, 1.08, 5, minSize=(min_side, min_side))
        return [tuple(int(v) for v in f) for f in fs]
    except Exception:
        traceback.print_exc()
        return []


def face_crop(img, faces=None, scale=FACE_SCALE, ar=FACE_AR, eye=FACE_EYE):
    """以人脸为中心裁一张人像构图。deskew 和 auto_crop 都失败时用。

    【什么时候轮到它】
        照片的边被画面切掉了、背景太平（米黄桌面）、照片嵌在别的东西里
        （身份证、奖状、相册内页）—— 这些情况文档检测全都抓不到边，
        但只要有脸就能定位。

    【⚠️ 合影要特殊处理】
        宽度取 max(高×比例, 人脸包围盒宽×1.6)，否则两个人会被竖版框切掉一个。

    返回 (x0, y0, x1, y1) 或 None。
    """
    try:
        if faces is None:
            faces = find_faces(img)
        if not faces:
            return None
        h, w = img.shape[:2]
        bx0 = min(f[0] for f in faces)
        by0 = min(f[1] for f in faces)
        bx1 = max(f[0] + f[2] for f in faces)
        by1 = max(f[1] + f[3] for f in faces)
        fw, fh = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0

        H = fh * scale
        W = max(H * ar, fw * 1.6)
        H = max(H, W / ar)

        x0, y0 = cx - W / 2.0, cy - H * eye
        x1, y1 = x0 + W, y0 + H
        # 超出画面就整体推回去，不要直接截断 —— 截断会让人脸跑偏
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > w:
            x0 -= (x1 - w)
            x1 = w
        if y1 > h:
            y0 -= (y1 - h)
            y1 = h
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 - x0 < 80 or y1 - y0 < 80:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        traceback.print_exc()
        return None


def strip_border(img, min_keep=0.35, max_keep=0.97):
    """去掉四周那圈花边（锯齿边、相纸留白、证件的防伪网纹）。

    【怎么分辨边和照片】
        花边是一张均匀的浅色纸，照片本体有内容。
        用【局部方差图】一刀就切开，实测同一张图上：
          上边（纯花边）亮度 224、方差 4
          中心（照片）  亮度 133、方差 31
        亮度不可靠（照片本身可能也很亮），方差可靠。

    【⚠️ 必须在摆正之后】
        歪着的时候花边是斜的，方差图切出来的矩形会把斜边的纸也圈进去。
    """
    try:
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = 15
        mean = cv2.blur(g, (k, k))
        var = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))
        thr = float(var.mean()) * 0.8
        if thr <= 0:
            return None
        m = (var > thr).astype(np.uint8) * 255
        # ⚠️ 闭运算把照片内部的平坦区域（天空、墙、浅色衣服）补回来，
        #    不然一张背景干净的证件照会被切得七零八落。
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, ww, hh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        keep = float(ww * hh) / float(w * h)
        if keep < min_keep or keep > max_keep:
            return None
        return (x, y, x + ww, y + hh)
    except Exception:
        traceback.print_exc()
        return None


def _ring_ratio(img, band=0.12):
    """画面四周一圈的方差 ÷ 中心区域的方差。

    比值明显小于 1 = 周围比中间平坦 = 照片嵌在桌面/墙面里，周围都是废面积。
    这是「该不该走人脸定位」唯一靠得住的判据 —— 人脸占画面的比例完全
    区分不了（实测身份证那张和一张已经处理好的，都是 24%）。

    实测（2026-11-01，四个样本）：
        身份证上的证件照   0.50   ← 该裁
        一张已处理好的图   0.74   ← 不该裁
        一张已裁好的图     0.95   ← 不该裁
        刚裁出来的结果     2.10   ← 不该再裁
    """
    try:
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = max(8, int(min(h, w) * band))
        ring = np.concatenate([g[:m, :].ravel(), g[-m:, :].ravel(),
                               g[:, :m].ravel(), g[:, -m:].ravel()])
        core = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
        return float(ring.std()) / max(float(core.std()), 1.0)
    except Exception:
        traceback.print_exc()
        return 1.0


# 人脸定位这条腿的两道闸。
# ⚠️ 这两个数是 4 个样本拟合出来的，而且 0.50 和 0.74 之间只隔 0.24。
#    这是整套预处理里【最不可靠】的一处。上线后第一件事就是拿一批
#    真实照片跑 /crop，看这条腿的误判率，再决定调参还是干脆关掉。
# ⚠️ 风险不对称：不裁只是构图差，裁错了是把用户的照片切掉一块。
#    所以两道闸是【并且】的关系，不是或。
RING_MAX = 0.60     # 周围得明显比中间平坦
FACE_KEEP_MAX = 0.40  # 而且确实能裁掉大部分，否则不值得冒险


def prepare(img, do_border=True):
    """整条预处理链。给调用方一个口子，不用自己拼。

    返回 (处理后的图, 说明字符串)。说明是给人看的，也原样回给前端 ——
    ⚠️ 悄悄改了用户的照片而不说，比不改更糟。

    说明的取值举例：
        'deskew -2.6°+border'   摆正了，还去了花边
        'crop+border'           只裁了，还去了花边
        'face'                  文档检测失败，按人脸裁的
        ''                      什么都没做，让用户自己框
    """
    fix = ''
    out, why = deskew(img)
    if out is not None:
        img = out
        fix = why
    else:
        box = auto_crop(img)
        if box:
            img = img[box[1]:box[3], box[0]:box[2]]
            fix = 'crop'
        else:
            # ⚠️ 这一步是身份证那类照片唯一能走通的路。别删。
            #    但它也是最容易误伤的一步 —— 它对任何有脸的图都会动手，
            #    包括已经构图好的照片。所以前面要过两道闸。
            #    2026-11-01：没加闸之前，一张已经处理好的图被它裁掉了 69%。
            h0, w0 = img.shape[:2]
            ring = _ring_ratio(img)
            box = face_crop(img) if ring < RING_MAX else None
            if box:
                keep = float((box[2] - box[0]) * (box[3] - box[1])) / float(w0 * h0)
                if keep <= FACE_KEEP_MAX:
                    img = img[box[1]:box[3], box[0]:box[2]]
                    fix = 'face'
                else:
                    print('[crop] 人脸定位能裁掉的不够多（留 %.0f%%），不动' % (100 * keep),
                          flush=True)
            elif ring >= RING_MAX:
                print('[crop] 周围不比中间平坦（%.2f），不是嵌在桌面里的照片，不动'
                      % ring, flush=True)

    # ⚠️ 只在前面真的动过才去边。原图都没处理成功的时候，
    #    画面里还有墙和桌子，方差图切出来的不是花边是整张照片。
    if fix and do_border:
        bb = strip_border(img)
        if bb:
            img = img[bb[1]:bb[3], bb[0]:bb[2]]
            fix += '+border'
    return img, fix
