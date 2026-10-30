# ============================================================
# 吱吱屋留影 —— 人脸修复服务（CodeFormer）
# 跑在阿里云函数计算 FC 的 GPU 函数上（自定义容器）
#
# 【为什么是这个基础镜像】
#   FC 的 Tesla 卡型驱动兼容 CUDA 11.x，torch 2.1.2+cu118 是最稳的组合。
#   用 runtime 不用 devel：runtime 小 2~3 GB，我们不需要编译 CUDA 代码。
#
# 【为什么权重烤进镜像，不在运行时下载】
#   冷启动时再去 GitHub 下 880 MB 权重，等于每次冷启动多等几分钟，
#   而且 GitHub 在国内不稳。烤进去镜像会大（约 7~8 GB），
#   但 FC 有镜像加速，拉取只慢在第一次。
#   ⚠️ 这是个明确的取舍：镜像体积换冷启动时间。
#
# 【这套依赖是验证过的】
#   2026-10-29 我在一台 4 核 CPU 的机器上用完全相同的版本组合
#   跑通了同一张老照片（锐度 2.4 → 113）。下面每一个 pin 都有理由：
#     numpy<2        —— torch 2.1.2 是按 numpy 1.x 编译的，装 2.x 直接崩
#     opencv 4.10    —— opencv 5.x 强依赖 numpy>=2，跟上一条冲突
#     lpips          —— CodeFormer 自带的 basicsr 启动时就 import 它，缺了起不来
#     basicsr/version.py —— 仓库里没有这个文件，是 setup 生成的。
#                       不自己补一个，import basicsr 直接报 ModuleNotFoundError。
#                       （这个坑我踩过，写在这儿省得下次再踩）
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# ---- PyTorch（CUDA 11.8）----
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu118

# ---- 其余依赖。版本钉死的理由见文件头 ----
RUN pip install \
        "numpy<2" \
        "opencv-python-headless==4.10.0.84" \
        scipy scikit-image tqdm pyyaml lmdb addict future yapf requests \
        lpips \
        flask==3.0.3 gunicorn==22.0.0

# ---- CodeFormer 源码 ----
# 用它自带的 basicsr / facelib，不要装 pip 版 basicsr ——
# pip 版会 import torchvision.transforms.functional_tensor，新版 torchvision 已经删了。
RUN git clone --depth 1 https://github.com/sczhou/CodeFormer.git /app/CodeFormer && \
    printf "__version__ = '1.3.2'\n__gitsha__ = 'unknown'\nversion_info = (1, 3, 2)\n" \
        > /app/CodeFormer/basicsr/version.py

# ---- 权重 ----
WORKDIR /app/CodeFormer
RUN mkdir -p weights/CodeFormer weights/facelib weights/realesrgan && \
    B=https://github.com/sczhou/CodeFormer/releases/download/v0.1.0 && \
    curl -sSL -o weights/CodeFormer/codeformer.pth            $B/codeformer.pth && \
    curl -sSL -o weights/facelib/detection_Resnet50_Final.pth $B/detection_Resnet50_Final.pth && \
    curl -sSL -o weights/facelib/parsing_parsenet.pth         $B/parsing_parsenet.pth && \
    curl -sSL -o weights/realesrgan/RealESRGAN_x2plus.pth     $B/RealESRGAN_x2plus.pth

COPY app.py /app/app.py
WORKDIR /app

# ⚠️ FC 自定义容器必须监听 0.0.0.0，端口从 FC_SERVER_PORT 读（默认 9000）。
#    监听 127.0.0.1 的话 FC 探活失败，函数一直起不来。
ENV FC_SERVER_PORT=9000
EXPOSE 9000

# 单 worker：模型只加载一次，热启动直接复用。
# 多 worker 会把显存占满好几份，GPU 函数上是灾难。
# timeout 给足：冷启动要把 880MB 权重读进显存。
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:${FC_SERVER_PORT:-9000} --timeout 600 app:app"]
