# -*- coding: utf-8 -*-
"""
吱吱留影 —— 照片预处理云函数（腾讯云 SCF，Python 3.9，事件函数 + API 网关触发器）

【它做什么】
    收一张手机翻拍的老照片，回一张「只剩照片本身」的图：
      摆正（纸歪了拉正）→ 裁掉桌面/墙 → 去花边 → 人脸定位（证件照那类）
      → 按眼睛下巴重新构图 → 把脏背景换成干净渐变
    全部逻辑在 crop.py，这个文件只管收发。

【为什么是 SCF 不是阿里云 FC】
    这几步只要 OpenCV + MediaPipe，不要 GPU。
    照片本来就在腾讯云（uniCloud-tcb），同一朵云里传图不跨网；
    按次计费，一张约 0.0001 元；不用养镜像仓库（阿里云 ACR 每月 117 元）。

【接口】
    POST  body = {"image": "<base64>", "beautify": true, "border": true}
       或 body = {"url": "https://…", …}
    ⚠️ 优先用 url。SCF 同步调用的请求体上限 6MB，手机原图 base64 之后很容易超；
       给个临时地址让这边自己去下，照片在腾讯云存储上，同一朵云里几十毫秒就下完。
    回    {"ok": true, "fix": "deskew -2.6°+border+recomp+bg", "image": "<base64>", "w":…, "h":…}
          fix 为空字符串 = 什么都没动（不是故障），调用方让用户自己框。

【鉴权】
    环境变量 ZZ_TOKEN。调用方在 header 里带 X-ZZ-Token。
    ⚠️ 这个地址挂在公网上。不设 ZZ_TOKEN 就不校验 —— 只能在本地调试时这么干。

【依赖怎么装】
    层（layer）里放 opencv-python-headless / numpy / mediapipe / protobuf / flatbuffers / attrs / absl-py，
    ⚠️ mediapipe 必须 --no-deps 装：它声明的依赖里有 jax / matplotlib / scipy，
       一共 300 多 MB，全装进去超过 SCF 500MB 的上限，而且运行根本用不到。
    层解压在 /opt，SCF 会自动把 /opt 加进 sys.path。
"""
import base64
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2          # noqa: E402
import numpy as np  # noqa: E402

from crop import prepare, beautify  # noqa: E402


def _resp(code, obj):
    return {
        'isBase64Encoded': False,
        'statusCode': code,
        'headers': {'Content-Type': 'application/json; charset=utf-8'},
        'body': json.dumps(obj, ensure_ascii=False),
    }


def _header(event, name):
    h = event.get('headers') or {}
    for k, v in h.items():
        if str(k).lower() == name.lower():
            return v
    return ''


def _body(event):
    b = event.get('body') or ''
    if event.get('isBase64Encoded'):
        b = base64.b64decode(b).decode('utf-8', 'ignore')
    if isinstance(b, dict):
        return b
    try:
        return json.loads(b) if b else {}
    except Exception:
        return {}


def _download(url, limit=20 * 1024 * 1024):
    """把临时地址下成字节。⚠️ 只认 http(s)，并且封顶 20MB —— 这是公网接口，别让人拿它当下载器。"""
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'zz-crop/1'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError('图片超过 20MB')
    return data


def main_handler(event, context):
    t0 = time.time()
    want = os.environ.get('ZZ_TOKEN', '')
    if want and _header(event, 'X-ZZ-Token') != want:
        return _resp(403, {'ok': False, 'msg': '没有权限'})
    try:
        body = _body(event)
        b64 = body.get('image') or ''
        url = str(body.get('url') or '').strip()
        if url.startswith('http'):
            raw = _download(url)
        elif b64:
            raw = base64.b64decode(b64)
        else:
            return _resp(400, {'ok': False, 'msg': '没给图片'})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return _resp(400, {'ok': False, 'msg': '图片解不开'})
        h, w = img.shape[:2]

        out, fix = prepare(img, do_border=body.get('border', True))
        if body.get('beautify', True):
            out, note = beautify(out)
            if note:
                fix = (fix + '+' if fix else '') + note

        if not fix:
            return _resp(200, {'ok': True, 'fix': '', 'w': w, 'h': h,
                               'msg': '没找到照片的边，让用户自己框',
                               'ms': int((time.time() - t0) * 1000)})

        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            return _resp(500, {'ok': False, 'msg': '结果图编码失败'})
        # ⚠️ 顺手给一张真灰度图。百度上色只吃黑白：褪色发黄的老照片在它眼里
        #    「已经是彩色的」，它不会动。原来这张灰度图是手机端框选时生成的，
        #    裁剪搬到云端之后，灰度图也得跟着搬，不然上色那一步没得喂。
        gray_b64 = ''
        if body.get('gray', True):
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            okg, bufg = cv2.imencode('.jpg', g, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if okg:
                gray_b64 = base64.b64encode(bufg.tobytes()).decode('ascii')
        return _resp(200, {
            'ok': True, 'fix': fix,
            'image': base64.b64encode(buf.tobytes()).decode('ascii'),
            'gray': gray_b64,
            'w': int(out.shape[1]), 'h': int(out.shape[0]),
            'ratio': round(float(out.shape[0] * out.shape[1]) / float(w * h), 3),
            'ms': int((time.time() - t0) * 1000),
        })
    except Exception as e:
        # ⚠️ 把堆栈带回去。云函数日志翻起来慢，调用方拿不到原因就只能靠猜。
        traceback.print_exc()
        return _resp(500, {'ok': False, 'msg': str(e)[:300],
                           'trace': traceback.format_exc()[-800:]})
