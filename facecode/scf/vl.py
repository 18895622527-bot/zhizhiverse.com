# -*- coding: utf-8 -*-
"""
让千问 VL 指出「照片在哪、四个角在哪」。

【为什么要它】
    2026-11-02 李总：「用户使用的时候，照片都是五花八门的。」
    手写的边缘规则（crop.py 里 deskew / auto_crop / face_crop）每条都是从三五张样本
    拟出来的，换个相册、换个光线就绕过去 —— 两天里修了五个坑，第六个还在后面。
    千问 VL 见过几亿张图，知道「那是一张照片」。让它指角，OpenCV 只负责按角拉正裁切。
    规则那套退到备胎：VL 没指出来、或指得不合理，再走规则。

【接口照 zzPhotoTell 的写法】
    DashScope 原生多模态接口 + qwen-vl-max → qwen-vl-plus 降级 +
    output.choices[0].message.content 里拼 text。那条路在线上跑了一个月，照抄不改。

【坐标怎么对得上】
    各代千问 VL 回坐标的约定不一样（有的是像素，有的是 0~1000 千分比）。
    所以先把图【等比缩放并补成 1000×1000 的正方形】再发给它（base64 直接带） ——
    这样像素和千分比是同一个数，不用猜它用的哪种。拿回来再减掉补的边、除以缩放比。
    ⚠️ 万一这个接口不收 base64，退一步把原图的临时地址给它，坐标按
       「数值都 ≤1000 而图比 1000 大 → 千分比」来猜。两条路都走不通才放弃。

【合理性检查】缺一不可，不然模型偶尔胡说一次就毁一张照片：
    ① 正好 4 个点，都落在画面内（允许 3% 出界）
    ② 凸四边形，面积占原图 3%~97%
    ③ 原图里 Haar 能认出的脸，中心都得在四边形里 —— 照片不可能不包含照片里的人

【key】环境变量 DASHSCOPE_KEY。整段 116 位的那串粘进去也行，这里只抠 sk- 开头那一截。
    没设就整层跳过，不报错。
"""
import base64
import json
import os
import re
import time
import urllib.request

import cv2
import numpy as np

# 2026-11-03：qwen3-vl 系列只在 OpenAI 兼容接口上有，原生 multimodal-generation 接口回「Model not exist」
MULTI = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
# ⚠️ 2026-10-11 阿里下线了 qwen-vl-max / qwen-vl-plus。按顺序试，「模型不存在」就试下一个。
#    环境变量 VL_MODELS 可以覆盖（逗号分隔），不用重新打包。
VL_MODELS = [m.strip() for m in os.environ.get('VL_MODELS', 'qwen3-vl-plus,qwen3-vl-flash').split(',') if m.strip()]
CANVAS = 1000

PROMPT = (
    '这是一张用手机拍的画面，里面有一张实体老照片（可能贴在墙上、放在桌上、夹在相册页里、'
    '印在证件上，或者被手拿着）。请找出【照片本身】的四个角。'
    '照片自带的白边、花边算在照片内；相册页的花纹、相框、桌面、墙、手指不算。'
    '图片尺寸 {W}×{H} 像素。'
    '只输出 JSON，不要任何别的文字：'
    '{{"found": true, "quad": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}}，'
    '顺序为左上、右上、右下、左下，坐标为像素整数。'
    '如果整张图本身就是那张照片、四周没有多余的东西，输出 {{"found": false}}。'
)


def pick_key(raw):
    """照 zzPhotoTell 的 pickKey：从任何一段文字里抠出 sk- 开头的那一截。"""
    s = str(raw or '')
    m = re.search(r'sk-[A-Za-z0-9]{16,}', s)
    if m:
        return m.group(0)
    t = s.strip()
    if t.startswith('sk-') and len(t) >= 10:
        return t.split()[0]
    return ''


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


def _call(key, image_ref, w, h, timeout=30):
    """照 zzPhotoTell.look：原生接口，模型降级。回文字；两个模型都不行就抛。"""
    messages = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_ref}},
        {'type': 'text', 'text': PROMPT.format(W=w, H=h)},
    ]}]
    last = ''
    for model in VL_MODELS:
        payload = {'model': model, 'messages': messages, 'temperature': 0.0, 'max_tokens': 200}
        req = urllib.request.Request(
            MULTI, data=json.dumps(payload).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode('utf-8', 'ignore'))
        except urllib.error.HTTPError as e:
            try:
                d = json.loads(e.read().decode('utf-8', 'ignore'))
            except Exception:
                d = {'code': 'HTTP%d' % e.code}
        except Exception as e:
            last = str(e)[:120]
            continue
        if d.get('error'):
            er = d['error'] if isinstance(d['error'], dict) else {'message': str(d['error'])}
            last = '%s:%s' % (er.get('code', ''), str(er.get('message', ''))[:100])
            continue
        text = ''
        try:
            c = d['choices'][0]['message']['content']
            if isinstance(c, list):
                text = ''.join(x.get('text', '') for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                text = c
        except Exception:
            pass
        if text.strip():
            return text
        last = '这一次它没说话'
    raise ValueError(last or '没有可用模型')


def _parse(text):
    """从模型回的文字里抠 JSON。它偶尔会包一层 ```json ```，偶尔前后带话。"""
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('回的不是 JSON：' + text[:160])
    d = json.loads(m.group(0))
    if not d.get('found'):
        return None
    q = d.get('quad')
    if not isinstance(q, list) or len(q) != 4:
        raise ValueError('quad 不是 4 个点：' + text[:160])
    return np.array([[float(p[0]), float(p[1])] for p in q], dtype='float32')


def _plausible(q, w, h, faces):
    """合理性检查。回 (True, '') 或 (False, 原因)。"""
    slack_x, slack_y = 0.03 * w, 0.03 * h
    if (q[:, 0] < -slack_x).any() or (q[:, 0] > w + slack_x).any() or \
       (q[:, 1] < -slack_y).any() or (q[:, 1] > h + slack_y).any():
        return False, 'out-of-frame'
    poly = q.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(poly):
        return False, 'not-convex'
    area = float(cv2.contourArea(q)) / float(w * h)
    if area < 0.03 or area > 0.97:
        return False, 'area %.2f' % area
    for (fx, fy, fw, fh) in faces or []:
        c = (float(fx + fw / 2.0), float(fy + fh / 2.0))
        if cv2.pointPolygonTest(poly, c, False) < 0:
            return False, 'face-outside'
    return True, ''


def locate(img, faces=None, key=None, url=''):
    """回 (quad 4×2 原图像素 | None, 说明)。
    quad 为 None 时说明里写原因：no-key / found=false / implausible:… / error:…
    ⚠️ 永远不抛异常 —— 这一层失败，调用方退回规则就行。"""
    key = pick_key(key or os.environ.get('DASHSCOPE_KEY', ''))
    if not key:
        return None, 'no-key'
    t0 = time.time()
    h, w = img.shape[:2]
    faces = faces or []
    notes = []

    # 路 A：补成 1000×1000 的方图，base64 直接带过去。坐标无歧义。
    try:
        canvas, s, ox, oy = _letterbox(img)
        ok, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        ref = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
        text = _call(key, ref, CANVAS, CANVAS)
        q = _parse(text)
        if q is None:
            return None, 'found=false %dms' % int((time.time() - t0) * 1000)
        q[:, 0] = np.clip((q[:, 0] - ox) / s, 0, w - 1)
        q[:, 1] = np.clip((q[:, 1] - oy) / s, 0, h - 1)
        okp, why = _plausible(q, w, h, faces)
        if okp:
            return q, 'vl %dms' % int((time.time() - t0) * 1000)
        notes.append('A:implausible:%s %s' % (why, json.dumps(q.round().astype(int).tolist())))
    except Exception as e:
        notes.append('A:error:' + str(e)[:120])

    # 路 B：接口不收 base64 的话，给原图的临时地址。坐标按像素或千分比猜。
    if url:
        try:
            text = _call(key, url, w, h)
            q = _parse(text)
            if q is None:
                return None, 'found=false(B) %dms' % int((time.time() - t0) * 1000)
            if q.max() <= 1000.0 and max(w, h) > 1000:
                q[:, 0] *= w / 1000.0
                q[:, 1] *= h / 1000.0
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            okp, why = _plausible(q, w, h, faces)
            if okp:
                return q, 'vl(B) %dms' % int((time.time() - t0) * 1000)
            notes.append('B:implausible:%s' % why)
        except Exception as e:
            notes.append('B:error:' + str(e)[:120])

    return None, ' | '.join(notes)[:300]
