# ============================================================
# 吱吱留影 —— 人脸修复服务（CodeFormer）
# 跑在阿里云函数计算 FC 的 GPU 函数上（自定义容器），镜像由 ACR 从 GitHub 自动构建
#
# 【2026-11-03 基础镜像为什么是 python:3.10-slim，不是 nvidia/cuda】
#   ACR 杭州的构建机：拉不到 Docker Hub 的 nvidia/cuda（DNS 被劫持到境外 IP，i/o timeout），
#   拉阿里自家 ac2 的 PyTorch 镜像又被拒（tls: bad certificate），开不开「海外源智能加速」都一样。
#   ACR 文档说构建机「托管了常用基础镜像」—— 指的就是 python / ubuntu 这类官方库镜像。
#   而 GPU 函数里 NVIDIA 驱动是 FC 挂进容器的，PyTorch 的 cu118 轮子自带 CUDA 运行库和 cuDNN，
#   根本不需要 nvidia/cuda 这个基础镜像。
#
# 【所有下载源都在国内】
#   apt   mirrors.aliyun.com（Debian）
#   pip   mirrors.aliyun.com/pypi
#   torch mirrors.aliyun.com/pytorch-wheels/cu118（阿里镜像的 PyTorch 官方轮子）
#   CodeFormer 源码  仓库里自带（facecode/CodeFormer，1MB）
#   权重  hf-mirror.com（HuggingFace 国内镜像）
#
# 【版本钉死的理由】
#   torch 2.1.2+cu118  2026-10-29 在 CPU 机器上跑通过的组合；T4 卡 + FC 的驱动兼容 CUDA 11.8
#   numpy<2            torch 2.1 按 numpy 1.x 编译
#   opencv 4.10 headless  5.x 强依赖 numpy>=2
#   Pillow<11          mediapipe 0.10.14 会拉到 11，和 torchvision 的 _imaging 冲突
#   mediapipe 0.10.14  1.x 要 libEGL，无显示的容器起不来
# ============================================================
# ⚠️ 2026-11-03 20:00 实测：ACR 构建机连 docker.io 被劫持到境外 IP（108.160.170.43 / 157.240.15.8），
#    连官方库镜像 python:3.10-slim 都拉不到。走 DaoCloud 的 Docker Hub 国内镜像站。
FROM docker.m.daocloud.io/library/python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN (sed -i 's@deb.debian.org@mirrors.aliyun.com@g; s@security.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true) \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch（CUDA 11.8 轮子，阿里镜像）----
# ⚠️ 必须 --no-index + -f：否则 pip 会去 PyPI 拿默认的 cu12 版，2.5GB 还可能和 T4 的驱动不配。
RUN pip install --upgrade pip && \
    pip install --no-index -f https://mirrors.aliyun.com/pytorch-wheels/cu118/ \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    || pip install --no-index -f https://mirrors.aliyun.com/pytorch-wheels/cu118/ \
        torch==2.1.2 torchvision==0.16.2

RUN pip install \
        "numpy<2" \
        "opencv-python-headless==4.10.0.84" \
        "Pillow<11" \
        scipy scikit-image tqdm pyyaml lmdb addict future yapf requests \
        lpips \
        "mediapipe==0.10.14" \
        flask==3.0.3 gunicorn==22.0.0 \
    && pip install "numpy<2" "Pillow<11"

# ---- CodeFormer 源码（仓库里自带，不 clone）----
COPY CodeFormer /app/CodeFormer

# ---- 权重：hf-mirror.com，失败退到 GitHub ----
WORKDIR /app/CodeFormer
RUN set -e; \
    H=https://hf-mirror.com/sczhou/CodeFormer/resolve/main; \
    G=https://github.com/sczhou/CodeFormer/releases/download/v0.1.0; \
    dl() { curl -fsSL --retry 3 -o "$1" "$H/$2" || curl -fsSL --retry 3 -o "$1" "$G/$3"; }; \
    dl weights/CodeFormer/codeformer.pth            codeformer.pth                        codeformer.pth; \
    dl weights/facelib/detection_Resnet50_Final.pth facelib/detection_Resnet50_Final.pth  detection_Resnet50_Final.pth; \
    dl weights/facelib/parsing_parsenet.pth         facelib/parsing_parsenet.pth          parsing_parsenet.pth; \
    dl weights/realesrgan/RealESRGAN_x2plus.pth     realesrgan/RealESRGAN_x2plus.pth      RealESRGAN_x2plus.pth; \
    ls -la weights/CodeFormer weights/facelib weights/realesrgan; \
    test "$(stat -c %s weights/CodeFormer/codeformer.pth)" -gt 300000000

# ⚠️ crop.py / pipe.py 必须一起拷。app.py 开头就 import 它们。
COPY app.py crop.py pipe.py /app/
WORKDIR /app

# 构建时自检：所有 import 都能过。别等到 FC 上才 500。
RUN python -c "import sys; sys.path.insert(0,'/app/CodeFormer'); import torch, cv2, numpy, mediapipe, flask; from basicsr.utils.registry import ARCH_REGISTRY; from facelib.utils.face_restoration_helper import FaceRestoreHelper; import crop, pipe; print('selfcheck ok, torch', torch.__version__, 'cuda build', torch.version.cuda)"

ENV FC_SERVER_PORT=9000
EXPOSE 9000
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:${FC_SERVER_PORT:-9000} --timeout 600 app:app"]
