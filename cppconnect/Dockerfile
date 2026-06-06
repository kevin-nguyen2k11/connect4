FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	cmake \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libstdc++6 \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        gdb \
        python3 \
        python3-numpy \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

ENV CI_BUILD_PYTHON python3

run mv /usr/bin/python /usr/bin/python.bak
run ln -s /usr/bin/python3 /usr/bin/python
ENV CUDA_HOME /usr/local/cuda
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /colosseum

ENV CUDNN_INSTALL_PATH /usr

COPY src/ ./src/
COPY bin/ ./bin/
COPY CMakeLists.txt .

WORKDIR /colosseum/build

RUN cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --parallel 8

CMD ["sh", "-c", "./colosseum/build/C___Connect_4"]
