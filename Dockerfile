ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/cuda:10.1-devel-ubuntu16.04--master
FROM ${FROM_IMAGE_NAME}

ARG KALDI_VERSION
ENV KALDI_VERSION=${KALDI_VERSION}
LABEL com.nvidia.kaldi.version="${KALDI_VERSION}"
ARG NVIDIA_KALDI_VERSION
ENV NVIDIA_KALDI_VERSION=${NVIDIA_KALDI_VERSION}

ARG PYVER=3.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        autoconf \
        cmake \
        flac \
        gawk \
        libatlas3-base \
        libtool \
        python$PYVER \
        python$PYVER-dev \
        sox \
        subversion \
        unzip \
        bc \
        libatlas-base-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN rm -f /usr/bin/python && ln -s /usr/bin/python$PYVER /usr/bin/python
RUN MAJ=`echo "$PYVER" | cut -c1-1` && \
    rm -f /usr/bin/python$MAJ && ln -s /usr/bin/python$PYVER /usr/bin/python$MAJ

WORKDIR /opt/kaldi

COPY tools tools
RUN cd tools/ \
 && make -j"$(nproc)" \
 && make dockerclean

# Copy remainder of source code
COPY . .

RUN cd src/ \
 && ./configure --shared --use-cuda --cudatk-dir=/usr/local/cuda/ --mathlib=ATLAS --atlas-root=/usr \
    --cuda-arch="-gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" \
 && make -j"$(nproc)" depend \
 && make -j"$(nproc)" \
 && make -j"$(nproc)" ext \
 && ldconfig \
 && find . -name "*.o" -exec rm {} \; \
 && find . -name "*.a" -exec rm {} \; \
 && cd ../tools/ \
 && make dockerclean 

ENV PYTHONPATH $PYTHONPATH:/usr/local/python

WORKDIR /workspace
COPY NVREADME.md README.md
RUN ln -s /opt/kaldi/egs /workspace/examples
COPY nvidia-examples nvidia-examples
RUN chmod -R a+w /workspace

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
