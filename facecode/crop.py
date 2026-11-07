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

# 透视拉正的两道闸（见 deskew 的说明）。
# ⚠️ 两个数都只在 3 个样本上验过。
QUAD_RECT_TOL = 8.0    # 四边形的边和 minAreaRect 的角度差，超过就不信四边形
RECT_FILL_MIN = 0.80   # 凸包占 minAreaRect 的比例，够满才允许只转不透视

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
    """四个点排成 左上 / 右上 / 右下 / 左下。

    ⚠️ 2026-11-01 改成按「绕中心的角度」排。
       原来用 x+y 最小当左上、x-y 最小当右上 —— 照片歪到 45° 时这两个会挑中同一个点，
       四边形退化成三角形，整条透视路直接废掉。李总发来的第一张老照片就是 45° 放的。
       按角度排永远不会重复；谁当「左上」仍取 x+y 最小的那个，
       歪 45° 时左上/右上分不清也没关系，后面 _upright 会按脸把方向转对。"""
    p = np.asarray(p, dtype='float32')
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                 # 图像坐标 y 朝下，角度递增 = 顺时针
    start = int(np.argmin(p.sum(axis=1)))
    return np.roll(p, -start, axis=0).astype('float32')


def _norm_ang(a):
    """把角度折到 (-45, 45]：矩形转 90° 还是矩形，比的是「歪了多少」不是「朝哪边」。"""
    a = float(a) % 90.0
    return a - 90.0 if a > 45.0 else a


def _rotate_crop(img, rect):
    """按 minAreaRect 转平再裁。只转不透视 —— 给「四个角找不全」的照片用。"""
    (cx, cy), (rw, rh), ang = rect
    a = _norm_ang(ang)
    # ⚠️ 角度折过之后宽高可能要对调（OpenCV 的 angle 是跟 width 那条边算的）
    if abs(a - float(ang)) > 1.0:
        rw, rh = rh, rw
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = int(max(0, cx - rw / 2.0)), int(max(0, cy - rh / 2.0))
    x1, y1 = int(min(w, cx + rw / 2.0)), int(min(h, cy + rh / 2.0))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None, 'too-small'
    return rot[y0:y1, x0:x1], 'deskew %.1f°' % a


def deskew(img):
    """找照片的四个角，透视拉正 —— 顺带把裁剪也做了。

    【⚠️ 必须先取凸包再拟合四边形】
        老照片常有锯齿花边。直接对轮廓 approxPolyDP 拟合不出四个角
        （波浪线上找不到拐点），2026-11-01 实测三张全部 not-a-quad。
        先 convexHull 把锯齿磨平，四个角立刻就出来了。

    【⚠️ 四边形必须和 minAreaRect 对得上，否则不许透视】
        2026-11-01 下午踩的坑：一张纸只歪 5° 的照片，轮廓在右上角断了一截
        （浅色花边融进了浅色背景），凸包少了一个角，拟出来的四边形上边是条
        斜线 —— 判成「歪 26°」，透视一拉，人脸被剪切了 18°。对边比 0.65 居然过了闸。
        minAreaRect 不吃这一套：它包住整个凸包，缺个角也还是那个矩形，角度 5°。
        所以：四边形的边角度和 minAreaRect 的角度差超过 QUAD_RECT_TOL 就不信四边形。
        不信四边形时，如果凸包把矩形填得够满（≥ RECT_FILL_MIN），退一步只转不透视。

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
        rect = cv2.minAreaRect(hull)
        rect_ang = _norm_ang(rect[2])
        rect_area = float(rect[1][0] * rect[1][1])
        fill = float(cv2.contourArea(hull)) / max(rect_area, 1.0)
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
                break   # 退化了就别透视，下面还有只转不透视那条路
            # ⚠️ 对边长度比 —— 区分「真·斜着拍的矩形」和「乱凑的四边形」
            #    最有效的一个数。实测：真歪 26° 的 0.65，真实原图 0.92，
            #    误判那张只有 0.42。
            #    ⚠️ 0.55 是 4 个样本拟合出来的，不是验出来的，上线后看数据再调。
            #    ⚠️ 风险不对称，所以宁可卡紧：
            #       漏判 → 退回只裁，照片还是好的，只是歪着；
            #       误判 → 照片被转歪，毁了。
            par = (min(top, bot) / max(top, bot)) * (min(lft, rgt) / max(lft, rgt))
            if par < 0.55:
                break

            # 四条边各自的角度都得和 minAreaRect 一致
            angs = [np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
                    for a, b in ((tl, tr), (bl, br), (tl, bl), (tr, br))]
            off = max(abs(_norm_ang(x) - rect_ang) for x in angs)
            if off > QUAD_RECT_TOL:
                print('[crop] 四边形和外接矩形对不上（差 %.1f°），不透视' % off, flush=True)
                break

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

        # 四边形不可信。凸包把矩形填得够满，就只转不透视。
        if fill >= RECT_FILL_MIN:
            return _rotate_crop(img, rect)
        return None, 'not-a-quad fill %.2f' % fill
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
        # ⚠️ 合影不能硬套 3:4 竖版。
        #    2026-11-02 相册页里一张两人合影：两张脸并排，W 被撑到 629，
        #    再按 0.75 反推 H 就成了 838 —— 把上下两条花边一起框进来，等于没裁。
        #    多张脸时让框横过来：高度只保证装下脸和肩膀（W 的 0.75 倍），不再反推。
        if len(faces) >= 2:
            H = max(H, W * 0.8)
            eye = 0.47   # 合影里头顶上方多留一点：两个人的头不在一条线上，矮的那个会被切
        else:
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


def trim_bottom_paper(img, zone=0.30, row_frac=0.55, min_band=0.015):
    """去掉照片【底下】那条相纸白边（以及白边下面露出来的相册、桌面）。回新的下沿 y，不用切就回 None。

    ⚠️ 2026-11-07 一张相册里的合影：底下锯齿白边 + 粉色相册纸有花纹，strip_border 当成照片内容留下了。
    【怎么认】一行里一半以上的像素「又亮又平」（灰度 > 170，局部标准差 < 10）= 这一行是相纸。
             在最下面 30% 里，从上往下找第一段连续的相纸行（至少 1.5% 高），而且它上面紧挨着的是有内容的行，
             就从这段的上沿切。
    【只切下边】上边和左右常常是照片里本来的浅色背景（影楼背景布、墙），也是又亮又平，切了就伤内容。
             肖像照的下边是衣服，几乎不会整行又亮又平。"""
    try:
        h, w = img.shape[:2]
        if h < 120:
            return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = 9
        mean = cv2.blur(g, (k, k))
        sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))
        paper = ((g > 170) & (sd < 10)).mean(axis=1) > row_frac
        # 左中右三段分开算「偏亮的像素占多少」：照片略斜时白边一头高一头低，整行平均会把高的那头平均掉
        third = max(1, w // 3)
        bright = np.max(np.stack([(g[:, i * third:(i + 1) * third] > 135).mean(axis=1) for i in range(3)]), axis=0)
        y_start = int(h * (1.0 - zone))
        need = max(3, int(h * min_band))
        y = y_start
        while y < h:
            if paper[y]:
                e = y
                while e < h and paper[e]:
                    e += 1
                if e - y >= need:
                    above = paper[max(0, y - need):y]
                    if len(above) and not above.any():
                        # 白边上沿是一段渐变（照片略斜、锯齿），往上退到渐变开始的地方，别留一条亮线
                        lim = max(y_start, y - int(0.04 * h))
                        while y > lim and bright[y - 1] > 0.25:
                            y -= 1
                        return y
                y = e
            else:
                y += 1
        return None
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
# 第二条进门的路：脸在画面里太小。
# ⚠️ 2026-11-02 相册页那张合影：周围全是花纹，一点都不平坦（ring 1.21），
#    上面那道闸永远不会开。但两张脸只占画面高度的 18% —— 一张构图好的人像，
#    脸不会这么小。身份证那张也是 18%，已经裁好的人像是 35% 以上。
#    ⚠️ 全身照会被误伤（脸也小）：会被裁成半身。现阶段接受 —— 老照片里
#       证件照和合影远多于全身照，而且用户在框选页还能自己改。
FACE_SMALL_MAX = 0.22  # 最大那张脸的高 ÷ 画面高，小于它就允许按脸裁


# ============================================================
# 「摆正」的第二层意思：人在画面里摆正，不只是纸摆正
#
# 2026-11-01 李总指出：裁完的照片「人像很明显是没有摆正的」。
# 量了关键点才发现：两张单人照头都是正的（-0.1° / +1.0°），人脸也在中线上。
# 他看着不正，是因为【框住人的东西不对称】—— 网纹只在一边、背景是脏灰墙、
# 下面衣服留得比头顶多。人在正中但画面不正，看着就是没摆正。
#
# 所以这一层要做三件事，缺一样感觉就不成立：
#   ① 按眼睛和下巴重新构图：两眼中点水平居中、眼睛落在 0.36、下巴落在 0.72
#   ② 头歪超过 3° 就转平（⚠️ 只对单人照；合影里靠在肩上的歪头是真实姿态）
#   ③ 人像分割，把背景换成干净的影楼渐变
#
# 这一层依赖 mediapipe（人脸关键点 + 人像分割）。
# ⚠️ 用 0.10.x 的旧 API（mp.solutions）。1.x 的 tasks API 要 libEGL，
#    在没有显示环境的云函数里起不来。2026-11-01 踩过。
# ⚠️ 没装 mediapipe 时这一层整个跳过，前面的裁剪不受影响 —— 所以是懒加载。
# ============================================================

# ⚠️ 2026-11-01 从 0.36 / 0.72 改成 0.43 / 0.68：原来头顶被切掉了。
#    眼睛到下巴算 1 个单位，眼睛到头发顶大约 1.2~1.4 个单位，
#    0.36 只给眼睛上面留了 1 个单位，刘海都放不下。
#    现在眼睛上面留 1.7、下巴下面留 1.3 —— 头顶有余、肩膀露一截，像影楼的半身照。
# ⚠️ 2026-11-03 再放宽到 0.40 / 0.58：0.43/0.68 出来的图头顶贴边、肩膀没了（李总发的第四张）。
#    眼到下巴算 1，现在头顶上面留 2.2、下巴下面留 2.3 —— 肩膀和上衣能进来，像证件照不像大头贴。
EYE_Y, CHIN_Y, PORTRAIT_AR = 0.40, 0.58, 0.75
# 重新构图后画面不能比裁剪结果小太多 —— 小于这个比例说明切过头了，不用它
RECOMP_MIN_KEEP = 0.70
TILT_FIX_DEG = 5.0   # 2026-11-03：3° 时一张正的证件照被转了 8.7°，关键点在糊脸上会抖，放宽
# 超过这个角度不信关键点 —— 影楼照里没人歪 25° 以上的头，那多半是网格贴错了
TILT_MAX_DEG = 12.0  # 超过这个不信关键点。单人证件照/半身照没人歪 12° 以上
_MP = {}


def _mp():
    """懒加载 mediapipe。导入失败就返回 None，调用方跳过美化。"""
    if 'ok' in _MP:
        return _MP if _MP['ok'] else None
    try:
        import mediapipe as mp
        _MP['mesh'] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=6, refine_landmarks=True,
            min_detection_confidence=0.4)
        _MP['seg'] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        # 2026-11-07 给构图数人头用的第二个检测器。face mesh 在合影里会漏脸（侧着、靠在一起的那张），
        # 漏了就会被当成单人照裁成一个人。两个检测器有一个说「不止一个人」就不构图。
        _MP['det'] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        _MP['ok'] = True
        return _MP
    except Exception as e:
        print('[crop] mediapipe 不可用，跳过构图和换背景：%s' % str(e)[:120], flush=True)
        _MP['ok'] = False
        return None


LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def landmarks(img):
    """返回每张脸的 {eye_l, eye_r, chin, tilt}（像素坐标），没检出就空列表。
    关键点编号是 mediapipe face mesh 的：左眼 33/133，右眼 362/263，下巴 152。"""
    m = _mp()
    if not m:
        return []
    h, w = img.shape[:2]
    r = m['mesh'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for fl in (r.multi_face_landmarks or []):
        p = [(q.x * w, q.y * h) for q in fl.landmark]
        le = np.mean([p[33], p[133]], axis=0)
        re = np.mean([p[362], p[263]], axis=0)
        out.append({
            'eye_l': le, 'eye_r': re, 'chin': np.array(p[152]),
            'tilt': float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))),
            # 外嘴唇轮廓一圈（mediapipe face mesh 编号）。2026-11-07 美颜给嘴唇补血色用，别的地方不用它
            'lips': np.array([p[i] for i in LIP_OUTER], dtype=np.float32)
        })
    return out


def recompose(img):
    """按关键点重新构图。只处理单人照 —— 多张脸直接原样返回。

    实测（2026-11-01）：
        身份证裁出来的 587x747 → 528x704，头 -0.1° 不用转
        墙上那张     558x655 → 394x525，头 +1.0° 不用转
    """
    fs = landmarks(img)
    if len(fs) != 1:
        return img, ''
    # ⚠️ 2026-11-07：合影被裁成一个人（女方被裁掉）。face mesh 只认出了男方，
    #    这里再用 BlazeFace 数一遍，不止一张脸就整张保留，不构图。
    if count_faces(img) > 1:
        print('[crop] 检测到不止一张脸，不构图', flush=True)
        return img, ''
    f = fs[0]
    h, w = img.shape[:2]
    mid = (f['eye_l'] + f['eye_r']) / 2.0
    chin = f['chin']
    note = ''
    if abs(f['tilt']) > TILT_MAX_DEG:
        print('[crop] 头歪 %.1f°，不像真的，不动' % f['tilt'], flush=True)
        return img, ''
    if abs(f['tilt']) > TILT_FIX_DEG:
        M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), f['tilt'], 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        chin = M[:, :2] @ chin + M[:, 2]
        note = 'tilt%+.1f ' % (-f['tilt'])
    d = float(chin[1] - mid[1])
    if d < 20:
        return img, ''
    h, w = img.shape[:2]
    H = d / (CHIN_Y - EYE_Y)
    # ⚠️ 框超出画面就【缩框】，不补边。
    #    2026-11-01 试过 BORDER_REPLICATE 补边：下边把衣服拉成一道道竖条，
    #    上边把头发和证件网纹一起抹上去，分割还把那团当成人 —— 比不构图难看得多。
    #    缩框的代价是比例不再精确是 0.43 / 0.68，但画面里每个像素都是真的。
    H = min(H, mid[1] / EYE_Y, (h - mid[1]) / (1.0 - EYE_Y), w / PORTRAIT_AR)
    W = H * PORTRAIT_AR
    if H < 80 or W < 60:
        return img, ''
    y0 = mid[1] - EYE_Y * H
    x0 = min(max(0.0, mid[0] - W / 2.0), w - W)
    out = img[int(y0):int(y0 + H), int(x0):int(x0 + W)]
    if out.shape[0] * out.shape[1] < RECOMP_MIN_KEEP * h * w:
        print('[crop] 重新构图只剩 %.0f%%，切过头了，不用' % (100.0 * out.shape[0] * out.shape[1] / (h * w)), flush=True)
        return img, ''
    return out, note + 'recomp'


def count_faces(img):
    """BlazeFace 数脸（缩到 640 以内）。出错回 0，调用方当「不知道」处理。"""
    m = _mp()
    if not m or 'det' not in m:
        return 0
    try:
        h, w = img.shape[:2]
        sc = 640.0 / max(h, w)
        sm = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
        r = m['det'].process(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))
        return len(r.detections or [])
    except Exception:
        return 0


# face mesh 脸外圈 36 个点（额头 → 右脸 → 下巴 → 左脸）
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _face_keep_mask(img, k):
    """每张脸的外轮廓填满，往里收 2% 再羽化 —— 羽化只发生在脸里面，不把外面的纸色带进来。
    回 0~1 的 float32 图；没检出脸回 None。"""
    m = _mp()
    if not m:
        return None
    h, w = img.shape[:2]
    r = m['mesh'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    keep = None
    for fl in (r.multi_face_landmarks or []):
        pts = np.array([(fl.landmark[i].x * w, fl.landmark[i].y * h) for i in FACE_OVAL], np.int32)
        face = np.zeros((h, w), np.uint8)
        cv2.fillPoly(face, [pts], 255)
        shrink = max(2, int(0.02 * (pts[:, 0].max() - pts[:, 0].min())))
        face = cv2.erode(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1)))
        face = cv2.GaussianBlur(face.astype(np.float32) / 255.0, (k, k), 0)
        keep = face if keep is None else np.maximum(keep, face)
    return keep


def clean_bg(img, color=(188, 176, 160), feather=9):
    """人像分割 + 柔边 + 换成影楼式浅灰渐变背景（上浅下深）。
    ⚠️ 颜色是 BGR。默认这个暖灰跟首页的纸色是一家的，印出来不突兀。
    ⚠️ 这一步是「换背景」里最保守的做法 —— 纯渐变，不生成任何场景。
       李总要的「自主生成更好看的背景」如果指的是场景，那是另一个模型的事。"""
    m = _mp()
    if not m:
        return img, ''
    h, w = img.shape[:2]
    mask = m['seg'].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).segmentation_mask
    if mask is None:
        return img, ''
    k = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip((mask - 0.25) / 0.5, 0, 1)
    # ⚠️ 2026-11-07 李总：「右脸像是少了一块」。脸很白、背景纸也很白的时候，分割把脸颊边当成背景抹成了灰蓝，
    #    后面 CodeFormer 按缺了一块的脸去修，轮廓就往里缩。所以脸的外轮廓里面一律保留原样。
    try:
        keep = _face_keep_mask(img, k)
        if keep is not None:
            mask = np.maximum(mask, keep)
    except Exception as fe:
        print('[crop] 保护脸轮廓那步跳过：%s' % str(fe)[:120], flush=True)
    mask = mask[..., None]
    # 人太少说明分割失败，别把整张图糊成一块灰
    if float(mask.mean()) < 0.08:
        return img, ''
    g = np.linspace(0, 1, h)[:, None, None]
    top = np.array(color, np.float32) * 1.08
    bot = np.array(color, np.float32) * 0.88
    bg = np.repeat(top * (1 - g) + bot * g, w, axis=1)
    out = (img.astype(np.float32) * mask + bg * (1 - mask)).astype(np.uint8)
    return out, 'bg'


def face_angle(img):
    """靠人脸找「转多少度才是正的」。回 (角度, 用来量的那块图的框) 或 (None, None)。

    2026-11-03 在水磨石桌面那张上逐 10° 扫了一遍才看清：MediaPipe 的 tilt 在 -40°~60° 之间
    是【线性可靠】的（0° 时报 51.7，转 50° 后报 0.9）。之前不信它是因为给它设了 25° 的上限 ——
    上限是为了防「正着的脸被误转」，但那该用「小角度不转」来防，不是「大角度不信」。
    所以：读一次 tilt → 大于 12° 就按它转 → 转完再读一次验证（|tilt| ≤ 6 且 Haar 认得出脸）。
    一次读 + 一次验，不扫圈，几十毫秒。
    ⚠️ 12° 以内不转：那是人本来歪着头，不是纸歪了。"""
    h, w = img.shape[:2]
    lm = landmarks(img)
    box = None
    if not lm:
        # 脸在画面里太小（照片只占一角）时 MediaPipe 找不到。
        # 退一步：每 30° 转一次让 Haar 粗找，找到就在那一小块上让 MediaPipe 验 —— 脸在小块里够大。
        found = _sweep_face(img)
        if found is None:
            return None, None
        a, box = found
        roi = img[box[1]:box[3], box[0]:box[2]]
        lm = landmarks(rotate_keep(roi, a)) if abs(a) > 1.0 else landmarks(roi)
        if not lm:
            return None, None
        t = a + max(lm, key=lambda x: np.linalg.norm(x['eye_r'] - x['eye_l']))['tilt']
    else:
        t = max(lm, key=lambda x: np.linalg.norm(x['eye_r'] - x['eye_l']))['tilt']
    if abs(t) <= 12.0:
        return 0, box
    src = img if box is None else img[box[1]:box[3], box[0]:box[2]]
    rot = rotate_keep(src, t)
    lm2 = landmarks(rot)
    if not lm2:
        return None, None
    t2 = min(abs(x['tilt']) for x in lm2)
    if t2 > 6.0 or not find_faces(rot, min_side=max(30, int(0.06 * min(rot.shape[:2])))):
        print('[crop] 按脸转 %.1f° 之后还是不正（%.1f°），不信' % (t, t2), flush=True)
        return None, None
    return float(t), box


def _sweep_face(img, step=30):
    """每 step 度转一次，Haar 找脸；每个命中在原图坐标里框一块（脸的 3 倍）让 MediaPipe 验。
    回 (转角, 原图里的框) 或 None。⚠️ Haar 在错角度上会报假脸，所以必须验。"""
    h, w = img.shape[:2]
    s = 500.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    diag = int(np.ceil(np.hypot(sw, sh)))
    for a in range(0, 360, step):
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), a, 1.0)
        M[0, 2] += (diag - sw) / 2.0
        M[1, 2] += (diag - sh) / 2.0
        r = cv2.warpAffine(small, M, (diag, diag), borderMode=cv2.BORDER_REPLICATE)
        fs = sorted(find_faces(r, min_side=24), key=lambda f: -f[2] * f[3])[:2]
        if not fs:
            continue
        Minv = cv2.invertAffineTransform(M)
        for (fx, fy, fw, fh) in fs:
            # 脸中心映回原图，框 3 倍脸宽的一块
            c = Minv @ np.array([fx + fw / 2.0, fy + fh / 2.0, 1.0])
            cx, cy = c[0] / s, c[1] / s
            half = 1.6 * fw / s
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
            if x1 - x0 < 60 or y1 - y0 < 60:
                continue
            roi = img[y0:y1, x0:x1]
            lm = landmarks(rotate_keep(roi, a)) if a else landmarks(roi)
            if lm and min(abs(x['tilt']) for x in lm) <= 15.0:
                return (a - 360 if a > 180 else a), (x0, y0, x1, y1)
    return None


def rotate_keep(img, deg, with_mask=False):
    """绕中心转 deg 度，画布放大到装得下整张图，补边用边缘复制。
    with_mask=True 时连一张「哪些像素是真图」的掩膜一起回 ——
    ⚠️ 复制出来的边是大片平坦区域，找纸（低方差）的时候会把它当成纸，必须用掩膜排掉。"""
    h, w = img.shape[:2]
    diag = int(np.ceil(np.hypot(w, h)))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0
    out = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if not with_mask:
        return out
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0)
    return out, mask


def paper_box(img, face=None, valid=None):
    """转正之后，找那张「纸」：低方差的一大块（相纸、白边、花边都比桌面/墙面平）。
    有脸就只认包着脸的那块。回 (x0, y0, x1, y1) 或 None。"""
    try:
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = 21
        mean = cv2.blur(g, (k, k))
        var = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))
        if valid is not None:
            # 转图补出来的边不算纸：方差强行拉高
            var = var.copy()
            var[cv2.erode(valid, np.ones((k, k), np.uint8)) == 0] = var.max() + 1.0
        for f in (0.5, 0.7):
            m = (var < var.mean() * f).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cands = []
            for c in cnts:
                x, y, ww, hh = cv2.boundingRect(c)
                if ww * hh < 0.02 * w * h:
                    continue
                if face is not None:
                    fx, fy = face[0] + face[2] / 2.0, face[1] + face[3] / 2.0
                    # 纸得包住脸，允许脸的下沿略出框（花边照常见）
                    if not (x - 0.3 * face[2] <= fx <= x + ww + 0.3 * face[2] and y - 0.3 * face[3] <= fy <= y + hh + 0.6 * face[3]):
                        continue
                cands.append((ww * hh, x, y, ww, hh))
            if not cands:
                continue
            _, x, y, ww, hh = max(cands)
            if face is not None:
                # 包住整张脸再加一点
                x0 = min(x, int(face[0] - 0.2 * face[2])); y0 = min(y, int(face[1] - 0.6 * face[3]))
                x1 = max(x + ww, int(face[0] + 1.2 * face[2])); y1 = max(y + hh, int(face[1] + 1.8 * face[3]))
            else:
                x0, y0, x1, y1 = x, y, x + ww, y + hh
            return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        return None
    except Exception:
        traceback.print_exc()
        return None


def prepare_by_face(img, do_border=True):
    """靠脸的那条路：读角度 → 转正 → 找纸 → 裁 → 去花边。
    给「纸找不到边」的照片用（花纹桌面、花纹相册页、照片占画面很小）。
    回 (图, 说明) 或 (原图, '')。"""
    ang, box = face_angle(img)
    if ang is None:
        return img, ''
    src = img if box is None else img[box[1]:box[3], box[0]:box[2]]
    if abs(ang) > 1.0:
        rot, valid = rotate_keep(src, ang, with_mask=True)
    else:
        rot, valid = src, None
    faces = find_faces(rot)
    face = max(faces, key=lambda q: q[2] * q[3]) if faces else None
    # ⚠️ 转正之后按脸裁，不找纸。找纸（低方差）在花纹桌面上不可靠 —— 2026-11-03 试过，
    #    tight / loose 两种包围盒按脸裁出来一模一样，找纸的两次都框成了整张。
    pb = face_crop(rot, [face]) if face is not None else None
    if pb is None:
        return img, ''
    out = rot[pb[1]:pb[3], pb[0]:pb[2]]
    if out.shape[0] < 80 or out.shape[1] < 80:
        return img, ''
    why = 'byface%+.0f' % ang
    if do_border:
        bb = strip_border(out)
        if bb:
            out = out[bb[1]:bb[3], bb[0]:bb[2]]
            why += '+border'
    return out, why


def _upright(img):
    """转 0 / 90 / 180 / 270 四个方向，留下脸最正的那个。

    为什么需要它：照片歪到 45° 附近时，「转 -45°」和「转 +45°」都能把纸摆平，
    但一个是正的、一个是躺着的；矩形本身分不出来，脸分得出来。
    ⚠️ 只在前面真的转过（deskew）之后才调。没转过的图方向就是用户拍的方向，不要碰。
    返回 (图, 转了几个 90°)。全部方向都找不到脸就原样返回。"""
    cands = [(0, img),
             (1, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
             (2, cv2.rotate(img, cv2.ROTATE_180)),
             (3, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))]
    best = None
    for k, cand in cands:
        fs = landmarks(cand)
        if fs:
            t = min(abs(f['tilt']) for f in fs)
        elif find_faces(cand):
            t = 15.0            # Haar 只认大致正着的脸，认出来就算 15°
        else:
            continue
        if best is None or t < best[0]:
            best = (t, k, cand)
    if best is None or best[1] == 0:
        return img, 0
    print('[crop] 按脸把方向转了 %d 个 90°' % best[1], flush=True)
    return best[2], best[1]


def beautify(img):
    """裁剪之后的第二层：构图 + 换背景。两步各自可失败，失败就跳过。"""
    notes = []
    img, n = recompose(img)
    if n:
        notes.append(n)
    img, n = clean_bg(img)
    if n:
        notes.append(n)
    return img, '+'.join(notes)


def warp_quad(img, q):
    """按给定的四个角（原图像素，任意顺序）透视拉正。给千问 VL 指的角用。
    回 (图, 说明) 或 (None, 原因)。"""
    try:
        q = _order_pts(np.asarray(q, dtype='float32'))
        tl, tr, br, bl = q
        top, bot = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
        lft, rgt = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
        W, H = int(max(top, bot)), int(max(lft, rgt))
        if W < 80 or H < 80:
            return None, 'too-small'
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], 'float32')
        out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (W, H))
        return out, 'vl'
    except Exception:
        traceback.print_exc()
        return None, 'error'


def prepare_with_quad(img, q, do_border=True):
    """VL 指了角之后的整条链：拉正 → 按脸转正方向 → 去花边。
    回 (图, 说明)；拉不出来回 (原图, '')，调用方再走规则。"""
    out, why = warp_quad(img, q)
    if out is None:
        return img, ''
    # ⚠️ 千问多半回的是【轴对齐的包围盒】，不是斜着的四个角 ——
    #    拉正等于只是裁了一块，照片在里面可能还斜着 25°。
    #    所以拉正之后再靠脸找一次角度；找不到脸才退回只转 90° 那套。
    lm = landmarks(out)
    upright_ok = bool(lm) and min(abs(x['tilt']) for x in lm) <= 8.0
    if not upright_ok:
        out2, why2 = prepare_by_face(out, do_border=do_border)
        if why2:
            return out2, why + ' ' + why2
    out, turned = _upright(out)
    if turned:
        why += ' turn%d' % turned
    if do_border:
        bb = strip_border(out)
        if bb:
            out = out[bb[1]:bb[3], bb[0]:bb[2]]
            why += '+border'
    return out, why


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
    # ⚠️ 「裁完脸还在不在」是最后一道保险。
    #    2026-11-01：一张已经摆正的照片，最大的轮廓是毛衣不是照片 ——
    #    auto_crop 把人裁成了一条只剩毛衣的横条。面积闸（刚好 0.30）没拦住。
    #    原图里有脸、裁完没脸了，这一刀就是错的，退回去走下一级。
    n_face = len(find_faces(img))

    def _lost_face(cand):
        return n_face > 0 and not find_faces(cand)

    out, why = deskew(img)
    if out is not None:
        # 歪得厉害的照片转平之后可能是躺着的 —— 按脸把方向转对
        out, turned = _upright(out)
        if turned:
            why += ' turn%d' % turned
    if out is not None and _lost_face(out):
        print('[crop] 摆正之后脸没了，不信这次摆正（%s）' % why, flush=True)
        out = None
    if out is not None:
        img = out
        fix = why
    else:
        box = auto_crop(img)
        if box and _lost_face(img[box[1]:box[3], box[0]:box[2]]):
            print('[crop] 裁完脸没了，不裁', flush=True)
            box = None
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
            faces = find_faces(img)
            small = bool(faces) and max(f[3] for f in faces) < FACE_SMALL_MAX * h0
            box = face_crop(img, faces) if (ring < RING_MAX or small) else None
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

    # 第四级：纸找不到、正着也认不出脸 —— 可能是斜着的。靠脸找角度再来一遍。
    # ⚠️ 正着就有脸的不走这里：那种图不动比动安全。
    if not fix and n_face == 0:
        out, why = prepare_by_face(img, do_border=do_border)
        if why:
            return out, why

    # ⚠️ 只在前面真的动过才去边。原图都没处理成功的时候，
    #    画面里还有墙和桌子，方差图切出来的不是花边是整张照片。
    if fix and do_border:
        bb = strip_border(img)
        if bb:
            img = img[bb[1]:bb[3], bb[0]:bb[2]]
            fix += '+border'
    return img, fix
