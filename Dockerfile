FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES 1

# set working directory
WORKDIR /app

# install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget \
    gegl \
    unzip \
    libgl1 \
    libglx-mesa0 \
    git-lfs \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN git clone https://github.com/comfyanonymous/ComfyUI
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r ComfyUI/requirements.txt
RUN pip3 install fastapi[standard]

WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/anilstream/ComfyUI-NanoBananaPro
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch


# copy source files
WORKDIR /app
COPY . .

# run fastapi app
CMD python3 retouch_api.py