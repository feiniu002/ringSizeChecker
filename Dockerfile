# 使用 Anaconda 镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# 创建并激活 conda 环境
RUN conda create -n app python=3.9 -y
SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]

# 安装 OpenCV 和其他依赖
RUN conda install -c conda-forge opencv -y
RUN pip install flask imutils pillow werkzeug gunicorn

# 复制项目文件
COPY . /app/

# 创建上传文件夹
RUN mkdir -p /app/static/uploads

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["conda", "run", "--no-capture-output", "-n", "app", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 