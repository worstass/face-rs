FROM ubuntu:jammy
ENV TVM_VERSION 0.15.0
ENV TVM_HOME /opt/tvm
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git python3 python3-dev python3-pip python3-setuptools gcc clang llvm-dev libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*
RUN cd /opt && \
    git clone --recursive --depth 1 -b v${TVM_VERSION} https://github.com/apache/tvm.git && \
    cd tvm && \
    git submodule init && \
    git submodule update
RUN cd /opt/tvm && \
    mkdir build && \
    cp cmake/config.cmake build && \
    cd build && \
    sed -i -e 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' config.cmake && \
    cmake .. && make -j4
RUN cd /opt/tvm/python && \
    python3 setup.py install
RUN apt-get update && \
    apt-get install -y --no-install-recommends libssl-dev pkg-config libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*
# RUN pip3 --no-cache-dir install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow pytest keras torch && \
#     pip3 --no-cache-dir install tensorrt --extra-index-url https://pypi.nvidia.com
