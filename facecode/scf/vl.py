# -*- coding: utf-8 -*-
"""
让千问 VL 指出「照片在哪、四个角在哪」。

【为什么要它】
    2026-11-02 李总：「用户使用的时候，照片都是五花八门的。」
    手写的边缘规则（crop.py 里 deskew / auto_crop / face_crop）每条都是从三五张样本
    拟出来的，换个相册、换个光线就绕过去 —— 两天里修了五个坑，第六个还在后面。
    千问 VL 见过几亿张图，知道「那是一张照片」。让它指角，OpenCV 只负责按角拉正裁切。
    规则那套退到备胎：VL 没指出来、或指得不合理，再走规则。

【坐标怎么对得上】
    各代千问 VL 回坐标的约定不一样（有的是像素，有的是 0~1000 千分比）。
    所以先把图【等比缩放并补成 1000×1000 的正方形】再发给它 —— 这样像素和千分比
    是同一个数，不用猜它用的哪种。拿回来再减掉补的边、除以缩放比，换回原图像素。

【合理性检查】缺一不可，不然模型偶尔胡说一次就毁一张照片：
    ① 正好 4 个点，都落在画面内（允许 3% 出界）
    ② 凸四边形，面积占原图 3%~97%
    ③ 原图里 Haar 能认出的脸，中心都得在四边形里 —— 照片不可能不包含照片里的人

【调用】DashScope 的 OpenAI 兼容接口，图片用 base64 data URL 直接带过去，不用上传。
    key 从环境变量 DASHSCOPE_KEY 读；没设就整层跳过，不报错。
    ⚠️ 这个文件我这边连不上阿里云，接口细节按公开文档写，第一次跑要看返回。
"""
import base64
import json
import os
import re
import time
import urllib.request

import cv2
import numpy as np

VL_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
VL_MODEL = os.environ.get('VL_MODEL', 'qwen-vl-max')
CANVAS = 1000

PROMPT = (
    '这是一张用手机拍的画面，里面有一张实体老照片（可能贴在墙上、放在桌上、夹在相册页里、'
    '印在证件上，或者被手拿着）。请找出【照片本身】的四个角。'
    '照片自带的白边、花边算在照片内；相册页的花纹、相框、桌面、墙、手指不算。'
    '图片尺寸 1000×1000 像素。'
    '只输出 JSON，不要任何别的文字：'
    '{"found": true, "quad": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}，'
    '顺序为左上、右上、右下、左下，坐标为像素整数。'
    '如果整张图本身就是那张照片、四周没有多余的东西，输出 {"found": false}。'
)


def _letterbox(img):
    """等比缩到长边 1000，补成 1000×1000。回 (方图, 缩放比, 左补, 上补)。"""
    h, w = img.shape[:2]
    s = CANVAS / float(max(h, w))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((CANVAS, CANVAS, 3), 128, np.uint8)
    ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, s, ox, oy


def _ask(canvas, key, timeout=25):
    ok, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError('编码失败')
    data_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
    payload = {
        'model': VL_MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': data_url}},
                {'type': 'text', 'text': PROMPT},
            ],
        }],
        'temperature': 0,
        'max_tokens': 200,
    }
    req = urllib.request.Request(
        VL_URL, data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode('utf-8', 'ignore'))
    text = ''
    try:
        text = body['choices'][0]['message']['content']
    except Exception:
        raise ValueError('返回里没有 content：' + json.dumps(body, ensure_ascii=False)[:300])
    if isinstance(text, list):   # 有的版本 content 是分段数组
        text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
    return str(text)


def _parse(text):
    """从模型回的文字里抠 JSON。它偶尔会包一层 ```json ```，偶尔前后带话。"""
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('回的不是 JSON：' + text[:200])
    d = json.loads(m.group(0))
    if not d.get('found'):
        return None
    q = d.get('quad')
    if not isinstance(q, list) or len(q) != 4:
        raise ValueError('quad 不是 4 个点：' + text[:200])
    return np.array([[float(p[0]), float(p[1])] for p in q], dtype='float32')


def _plausible(q, w, h, faces):
    """合理性检查。回 (True, '') 或 (False, 原因)。"""
    slack_x, slack_y = 0.03 * w, 0.03 * h
    if (q[:, 0] < -slack_x).any() or (q[:, 0] > w + slack_x).any() or \
       (q[:, 1] < -slack_y).any() or (q[:, 1] > h + slack_y).any():
        return False, 'out-of-frame'
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False, 'not-convex'
    area = float(cv2.contourArea(q)) / float(w * h)
    if area < 0.03 or area > 0.97:
        return False, 'area %.2f' % area
    for (fx, fy, fw, fh) in faces or []:
        c = (float(fx + fw / 2.0), float(fy + fh / 2.0))
        if cv2.pointPolygonTest(q.reshape(-1, 1, 2).astype(np.float32), c, False) < 0:
            return False, 'face-outside'
    return True, ''


def locate(img, faces=None, key=None):
    """回 (quad 4×2 原图像素 | None, 说明)。
    quad 为 None 时说明里写原因：no-key / found=false / 模型说的不合理 / 出错。
    ⚠️ 永远不抛异常 —— 这一层失败，调用方退回规则就行。"""
    key = key or os.environ.get('DASHSCOPE_KEY', '')
    if not key:
        return None, 'no-key'
    t0 = time.time()
    try:
        h, w = img.shape[:2]
        canvas, s, ox, oy = _letterbox(img)
        text = _ask(canvas, key)
        q = _parse(text)
        if q is None:
            return None, 'found=false %dms' % int((time.time() - t0) * 1000)
        q[:, 0] = (q[:, 0] - ox) / s
        q[:, 1] = (q[:, 1] - oy) / s
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        ok, why = _plausible(q, w, h, faces)
        if not ok:
            return None, 'implausible:%s %s' % (why, json.dumps(q.round().astype(int).tolist()))
        return q, 'vl %dms' % int((time.time() - t0) * 1000)
    except Exception as e:
        return None, 'error:' + str(e)[:160]
