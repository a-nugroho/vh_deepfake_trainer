FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /workspace/
RUN apt-get update && apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa -y && \
 apt-get update && apt-get install -y \
    wget \
    vim \
    python3.9 \
    python3.9-distutils \
    ffmpeg \
    libsm6 \
    libxext6 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.9 get-pip.py

COPY ./ /workspace/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80
ENV PORT 80

ENTRYPOINT []
CMD ["python3.9", "/workspace/server.py"]