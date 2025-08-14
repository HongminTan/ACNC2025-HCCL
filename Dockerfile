# 使用官方 Debian 12 镜像
FROM debian:12

# 使用清华源
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http.*//deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian |g' /etc/apt/sources.list && \
        sed -i 's|http.*//security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security |g' /etc/apt/sources.list; \
    else \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian  bookworm main contrib non-free" > /etc/apt/sources.list && \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security  bookworm-security main contrib non-free" >> /etc/apt/sources.list; \
    fi

# 更新包并安装工具链
RUN apt-get update && \
    apt-get install -y \
            sudo \
            openssh-server \
            git \
            cmake \
            ninja-build \
            build-essential \
            g++ \
            gcc \
            gdb \
            nano \
            unzip \
            clang-format \
            python3 \
            python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd

# 安装 node npm gemini-cli
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource-keyring.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g @google/gemini-cli && \
    apt-get clean && \
    npm cache clean --force && \
    rm -rf /tmp/* /var/tmp/*

# 安装 CANN
COPY Ascend-cann-toolkit_8.2.RC1.alpha003_linux-x86_64.run /tmp/
RUN chmod +x /tmp/Ascend-cann-toolkit_8.2.RC1.alpha003_linux-x86_64.run && \
    /tmp/Ascend-cann-toolkit_8.2.RC1.alpha003_linux-x86_64.run --install --quiet && \
    rm -f /tmp/Ascend-cann-toolkit_8.2.RC1.alpha003_linux-x86_64.run

# 配置 CANN 环境变量
RUN echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> /etc/profile.d/ascend.sh
ENV PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"

# 安装 CANN 运行依赖的 Python 包
RUN pip3 install --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple/ attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py

# 下载并解压 nlohmann json 头文件
RUN apt-get update && apt-get install -y unzip && \
    mkdir -p /opt/nlohmann_json && \
    curl -L -o /opt/nlohmann_json/include.zip https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip && \
    cd /opt/nlohmann_json && unzip include.zip && rm include.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 root 密码为 123456
RUN echo 'root:123456' | chpasswd

# 允许 root 通过 SSH 登录
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH 默认端口
EXPOSE 22

# 启动命令：前台运行 sshd
CMD ["/usr/sbin/sshd", "-D"]