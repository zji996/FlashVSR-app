# Backend Dockerfile with GPU support
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置Python 3.11为默认python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 安装uv包管理器
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 复制backend项目文件
COPY backend/pyproject.toml backend/.python-version /app/backend/
COPY backend/app /app/backend/app
COPY backend/alembic /app/backend/alembic
COPY backend/alembic.ini /app/backend/

# 安装Block-Sparse-Attention
RUN git clone https://github.com/mit-han-lab/Block-Sparse-Attention /tmp/Block-Sparse-Attention && \
    cd /tmp/Block-Sparse-Attention && \
    pip3 install packaging ninja && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/Block-Sparse-Attention

# 安装backend依赖
WORKDIR /app/backend
RUN uv pip install --system .

# 创建存储目录
RUN mkdir -p /app/storage/uploads /app/storage/results

# 设置环境变量
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令（使用Uvicorn）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
