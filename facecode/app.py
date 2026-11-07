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

        # 美颜（2026-11-07 李总：照美图秀秀那样，脸型圆润一点、皮肤亮白一点、五官微调）
        #   瘦脸 + 大眼 + 皮肤磨皮美白 + 彩色照片嘴唇补血色，每张脸都做。见 beauty.py。
        #   默认 0.7，调用方传 beauty:0 关掉。出错就跳过，修复结果照样返回。
        beauty_note = ''
        try:
            bk = float(body.get('beauty', 0.7))
        except Exception:
            bk = 0.7
        if bk > 0:
            try:
                result, beauty_note = beauty.beautify(result, bk)
            except Exception as be:
                print('[facecode] 美颜出错，跳过：', be, flush=True)

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
            size=[img.shape[1], img.shape[0]],
            ms=int((time.time() - t0) * 1000))
    except Exception as e:
        # ⚠️ 把真实堆栈带回去。云函数里看日志要押金，
        #    调用方拿不到原因就只能靠猜 —— 这个项目在这上面已经浪费过六天。
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300], trace=traceback.format_exc()[-800:]), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('FC_SERVER_PORT', 9000)))
