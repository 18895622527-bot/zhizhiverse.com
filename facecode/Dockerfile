# ============================================================
# 吱吱留影 —— 人脸修复服务（CodeFormer）
# 跑在阿里云函数计算 FC 的 GPU 函数上（自定义容器），镜像由 ACR 从 GitHub 自动构建
#
# 【2026-11-03 为什么全换成国内源】
#   ACR 杭州的构建机连不上 Docker Hub（registry-1.docker.io i/o timeout），
#   开了「海外源智能加速」也不行。所以这个 Dockerfile 不碰任何国外地址：
#     基础镜像  阿里云 ac2 官方 PyTorch 镜像（自带 CUDA 11.8 + cuDNN8 + torch 2.3 + Python）
#     pip       mirrors.aliyun.com
#     CodeFormer 源码  直接放在仓库里（facecode/CodeFormer，1MB），不再 git clone
#     模型权重  hf-mirror.com（HuggingFace 国内镜像，sczhou/CodeFormer 那个仓库）
#
# 【为什么权重烤进镜像，不在运行时下载】
#   冷启动时再下 640MB 权重，每次冷启动多等几分钟。烤进去镜像大（约 6GB），
#   但 FC 有镜像加速，拉取只慢在第一次。
#
# 【版本钉死的理由】
#   numpy<2            torch 2.x 按 numpy 1.x 编译的，装 2.x 直接崩
#   opencv 4.10 headless  5.x 强依赖 numpy>=2
#   Pillow<11          mediapipe 0.10.14 会拉到 11，和 torchvision 的 _imaging 冲突
#   mediapipe 0.10.14  1.x 要 libEGL，无显示的容器起不来
#   lpips              CodeFormer 的 basicsr 启动时就 import 它
# ============================================================
FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/pytorch-ubuntu:2.3.0-cuda11.8.0-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

# 系统库：OpenCV 要 libgl / libglib；curl 下权重。apt 换阿里源。
RUN sed -i 's@http://archive.ubuntu.com@https://mirrors.aliyun.com@g; s@http://security.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 基础镜像自带；torchvision 没带就补一个和 torch 2.3 配套的（从阿里的 pytorch 轮子镜像拿，不碰 PyPI 的 cu12 版）
RUN python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda)" && \
    (python -c "import torchvision; print('torchvision', torchvision.__version__)" || \
     pip install --no-deps "torchvision==0.18.0" -f https://mirrors.aliyun.com/pytorch-wheels/cu118/)

RUN pip install \
        "numpy<2" \
        "opencv-python-headless==4.10.0.84" \
        "Pillow<11" \
        scipy scikit-image tqdm pyyaml lmdb addict future yapf requests \
        lpips \
        "mediapipe==0.10.14" \
        flask==3.0.3 gunicorn==22.0.0 \
    && pip install "numpy<2" "Pillow<11"

# ---- CodeFormer 源码（仓库里自带）----
COPY CodeFormer /app/CodeFormer

# ---- 权重：hf-mirror.com，失败退到 GitHub（在国外构建机上才通）----
WORKDIR /app/CodeFormer
RUN set -e; \
    H=https://hf-mirror.com/sczhou/CodeFormer/resolve/main; \
    G=https://github.com/sczhou/CodeFormer/releases/download/v0.1.0; \
    dl() { curl -fsSL --retry 3 -o "$1" "$H/$2" || curl -fsSL --retry 3 -o "$1" "$G/$3"; }; \
    dl weights/CodeFormer/codeformer.pth            codeformer.pth                           codeformer.pth; \
    dl weights/facelib/detection_Resnet50_Final.pth facelib/detection_Resnet50_Final.pth     detection_Resnet50_Final.pth; \
    dl weights/facelib/parsing_parsenet.pth         facelib/parsing_parsenet.pth             parsing_parsenet.pth; \
    dl weights/realesrgan/RealESRGAN_x2plus.pth     realesrgan/RealESRGAN_x2plus.pth         RealESRGAN_x2plus.pth; \
    ls -la weights/CodeFormer weights/facelib weights/realesrgan; \
    test "$(stat -c %s weights/CodeFormer/codeformer.pth)" -gt 300000000

# ⚠️ crop.py / pipe.py 必须一起拷。app.py 第一行就 import 它们，
#    漏了的话容器起来探活是好的，第一个 /enhance 请求直接 500。
COPY app.py crop.py pipe.py /app/
WORKDIR /app

# 起容器前自检：所有 import 都能过。构建时就暴露问题，别等到 FC 上 500。
RUN python -c "import sys; sys.path.insert(0,'/app/CodeFormer'); import cv2, numpy, mediapipe, flask; from basicsr.utils.registry import ARCH_REGISTRY; from facelib.utils.face_restoration_helper import FaceRestoreHelper; import crop, pipe; print('selfcheck ok')"

# ⚠️ FC 自定义容器必须监听 0.0.0.0，端口从 FC_SERVER_PORT 读（默认 9000）。
ENV FC_SERVER_PORT=9000
EXPOSE 9000

# 单 worker：模型只加载一次，热启动直接复用。多 worker 会把显存占满好几份。
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:${FC_SERVER_PORT:-9000} --timeout 600 app:app"]
