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


MIN_AREA, MAX_AREA = 0.12, 0.85


def _biggest_contour(img):
    """找画面里最大的那一圈边缘，连它外接矩形占全图的比例一起返回。

    ⚠️ dilate 的第三个【位置】参数是 anchor 不是 iterations。
       写成 cv2.dilate(ed, k, 2) 的话膨胀根本没发生，轮廓是碎的，
       面积算出来只有 1%，整条路直接判失败。2026-11-01 在测试脚本里踩过。
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

    【为什么要这一步】
        用户举着手机对着墙上/桌上的照片拍，很少有正对着的。
        拍歪了再修再上色，出来还是歪的，用户一看就觉得"没处理过"。
        李总的话：「首先就是减少用户的操作，其次就是让照片质量尽可能变高。」

    【⚠️ 必须先取凸包再拟合四边形】
        老照片常有锯齿花边。直接对轮廓 approxPolyDP 拟合不出四个角
        （波浪线上找不到拐点），2026-11-01 实测三张全部 not-a-quad。
        先 convexHull 把锯齿磨平，四个角立刻就出来了。

    【验证】
        2026-11-01：
          人为扭歪 26° 的样本  → 拉正，631×717
          真实用户那张花边照片 → 检测到歪 2.6°，拉正并裁掉灰墙，668×826
          已经裁好的图          → 面积闸拦住，不动

    【⚠️ 失败就返回 None，让调用方退回 auto_crop，再失败就让用户自己框】
        三级降级。自动化失败绝不能变成用户卡住。

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
            # ⚠️ 对边长度比 —— 这是区分「真·斜着拍的矩形」和「乱凑的四边形」
            #    最有效的一个数。实测：真歪 26° 的 0.65，真实原图 0.92，
            #    误判那张只有 0.42。
            #    ⚠️ 0.55 这个阈值是 4 个样本拟合出来的，不是验出来的。
            #       上线后要看真实数据再调。
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
            # ⚠️ 长宽比离谱说明抓错了东西（抓到一条边、一道反光），别硬拉
            ar = W / float(H)
            if ar < 0.3 or ar > 3.3:
                return None, 'ratio %.2f' % ar
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], 'float32')
            out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (W, H))
            ang = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
            return out, 'deskew %.1f°' % ang
        return None, 'not-a-quad'
    except Exception:
        # ⚠️ 摆正只是锦上添花，出任何岔子都不该让整条链路失败
        traceback.print_exc()
        return None, 'error'


def strip_border(img, min_keep=0.35, max_keep=0.97):
    """去掉老照片四周那圈花边（锯齿边、白边、相纸留白）。

    【为什么要这一步】
        2026-11-01 李总：「照片外面的齿轮是一定要去掉的」。
        那圈锯齿花边是相纸的物理边缘，不是照片内容。留着它：
          - 画面里三分之一是米白色的纸，观感廉价
          - 上色模型会把那圈纸也当成要上色的区域
          - 人脸在整幅里占比更小，检测更容易漏

    【怎么分辨花边和照片】
        花边是一张均匀的浅色纸，照片本体有内容。
        用【局部方差图】一刀就切开了，实测同一张图上：
          上边（纯花边）亮度 224、方差 4
          中心（照片）  亮度 133、方差 31
        亮度不可靠（照片本身可能也很亮），方差可靠。

    【⚠️ 两道闸】
        留下来不足 min_keep：多半把照片内容也切掉了，不如不切。
        留下来超过 max_keep：本来就没花边，白忙。

    【⚠️ 顺序：必须在 deskew 之后】
        歪着的时候花边是斜的，方差图切出来的矩形会把斜边的纸也圈进去。
        先摆正，花边才是横平竖直的一圈。

    返回 (x0, y0, x1, y1) 或 None。
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


def auto_crop(img, pad=0.01):
    """把老照片本身从手机拍的那张里抠出来（不摆正，只裁）。

    ⚠️ 这是 deskew 的降级方案 —— 四个角拟合不出来的时候用它，
       至少能把周围的墙和桌面去掉。

    【为什么必须有这一步】
        2026-11-01 李总发来一张真实用户的结果：一张糊脸 + 三分之二的灰墙。
        用户拍的是贴在墙上的照片，我们却把整张手机照片拿去修了。
        他的话：「用户一看，就是简单上个色，根本不可能转发给朋友，
        下次也不会再打开这个小程序。」
        老人不会框，也看不懂为什么要框 —— 最好的框选界面是不出现。

    【怎么做的】
        高斯模糊 → Canny 边缘 → 膨胀连边 → 取最大轮廓的外接矩形。
        就这三步，不需要模型，CPU 上一毫秒级。

    【⚠️ 两道闸，宁可不裁也不要裁错】
        小于 MIN_AREA：多半抓到的是照片里的某个局部（一张脸、一块衣服），
                       裁下去会把人头切了。
        大于 MAX_AREA：裁不掉多少，不值得冒险。
        任何一道没过就返回 None，调用方退回让用户手动框 ——
        自动化失败不能变成用户卡住。

    【⚠️ MAX_AREA 为什么是 0.85 而不是 0.95】
        2026-11-01 拿一张【已经裁好的】照片试，检测框占到 88%，
        于是又切掉了 12% —— 那 12% 切的是真内容。
        改成 0.85 的意思是：至少能裁掉 15% 才值得动手，
        否则收益抵不上切错东西的风险。
        墙上那张要裁的占 53%，离这道闸远得很，不受影响。

    【⚠️ 已知会失败的情况，别指望它万能】
        照片放在花纹相册里、强反光、照片颜色和背景接近。
        2026-11-01 只在一张「花边照片贴素色墙」上验过，那是最好的情况。
        真实命中率要上线之后看数据。

    返回 (x0, y0, x1, y1) 或 None。
    """
    h, w = img.shape[:2]
    try:
        c, _ = _biggest_contour(img)
        if c is None:
            return None
        x, y, ww, hh = cv2.boundingRect(c)
        # ⚠️ 先加边距，再量面积，最后才判闸。
        #    反过来的话（量轮廓 → 判闸 → 加边距）会漏网：
        #    2026-11-01 一个 84% 的轮廓通过了 0.85 的闸，加完 1% 边距
        #    变成 88% 才裁出去 —— 判的和返回的不是同一个东西。
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + ww + px), min(h, y + hh + py)
        area = float((x1 - x0) * (y1 - y0)) / float(w * h)
        if area < MIN_AREA or area > MAX_AREA:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        # ⚠️ 裁剪只是锦上添花，出任何岔子都不该让整条链路失败
        traceback.print_exc()
        return None


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

        # 三级降级，跟 /enhance 里一模一样：摆正 → 只裁 → 让用户自己框
        out, why = deskew(img)
        box = None
        if out is None:
            box = auto_crop(img)
            if box:
                out = img[box[1]:box[3], box[0]:box[2]]
                why = 'crop'
            else:
                # ⚠️ 不是故障。调用方据此决定：让用户自己框。
                return jsonify(ok=True, fix='', msg='没找到照片的边，让用户自己框',
                               why=why, w=w, h=h), 200

        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            return jsonify(ok=False, msg='结果图编码失败'), 500
        return jsonify(ok=True, fix=why,
                       image=base64.b64encode(buf.tobytes()).decode('ascii'),
                       box=list(box) if box else None,
                       w=out.shape[1], h=out.shape[0],
                       ratio=round(float(out.shape[0] * out.shape[1]) / float(w * h), 3))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300]), 500


def _auth_ok():
    """⚠️ 这个接口挂在公网上，而且每次调用都烧 GPU 的钱。
       不加校验等于把你的信用卡放在路边。
       FC 的 HTTP 触发器可以配「无认证」图省事，所以这一道闸必须在应用层自己把。
       部署时在函数的环境变量里设 FC_SHARED_TOKEN，
       腾讯云那边的 zzFaceCode 带同一个值。"""
    want = os.environ.get('FC_SHARED_TOKEN', '')
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
        fix_box = None
        if body.get('crop', True):
            # 三级降级：摆正 → 只裁 → 原样交给用户自己框
            out, why = deskew(img)
            if out is not None:
                img = out
                did_fix = why
            else:
                fix_box = auto_crop(img)
                if fix_box:
                    img = img[fix_box[1]:fix_box[3], fix_box[0]:fix_box[2]]
                    did_fix = 'crop'
                else:
                    did_fix = ''
                    print('[facecode] 没自动处理：%s' % why, flush=True)

            # ⚠️ 去花边必须在摆正【之后】：歪着的时候花边是斜的，
            #    方差图切出来的矩形会把斜边的纸也圈进去。
            # ⚠️ 只在前面真的动过才做。原图都没裁成功的时候，
            #    画面里还有墙和桌子，方差图切出来的不是花边是整张照片。
            if did_fix and body.get('border', True):
                bb = strip_border(img)
                if bb:
                    img = img[bb[1]:bb[3], bb[0]:bb[2]]
                    did_fix += '+border'

        img = _shrink(img)

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

        ok, buf = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            return jsonify(ok=False, msg='结果图编码失败'), 500

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

        return jsonify(
            ok=True,
            image=base64.b64encode(buf.tobytes()).decode('ascii'),
            faces=n,
            w=w,
            # ⚠️ 动没动、怎么动的，原样带回去。
            #    调用方要能告诉用户「我顺手把相框和墙裁掉了、摆正了」——
            #    悄悄改了用户的照片而不说，比不改更糟。
            #    fix 的取值：'' 没动 / 'crop' 只裁了 / 'deskew -2.6°' 摆正了
            fix=did_fix,
            box=list(fix_box) if fix_box else None,
            ms=int((time.time() - t0) * 1000))
    except Exception as e:
        # ⚠️ 把真实堆栈带回去。云函数里看日志要押金，
        #    调用方拿不到原因就只能靠猜 —— 这个项目在这上面已经浪费过六天。
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300], trace=traceback.format_exc()[-800:]), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('FC_SERVER_PORT', 9000)))
