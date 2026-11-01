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


# ⚠️ 摆正/裁剪/去花边/人脸定位全在 crop.py 里。
#    它只依赖 OpenCV，不需要 GPU，也不需要 PyTorch ——
#    所以同一个文件可以单独跑在腾讯云 SCF 的轻量 Python 函数上，
#    不用为了裁张照片养一个 7GB 的容器镜像。
#    这里 import 它，GPU 这条路一行都不用改。
from crop import prepare as crop_prepare   # noqa: E402


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

        out, why = crop_prepare(img)
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
        if body.get('crop', True):
            img, did_fix = crop_prepare(img, do_border=body.get('border', True))
            if not did_fix:
                print('[facecode] 没自动处理，原样往下走', flush=True)

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
            size=[img.shape[1], img.shape[0]],
            ms=int((time.time() - t0) * 1000))
    except Exception as e:
        # ⚠️ 把真实堆栈带回去。云函数里看日志要押金，
        #    调用方拿不到原因就只能靠猜 —— 这个项目在这上面已经浪费过六天。
        traceback.print_exc()
        return jsonify(ok=False, msg=str(e)[:300], trace=traceback.format_exc()[-800:]), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('FC_SERVER_PORT', 9000)))
