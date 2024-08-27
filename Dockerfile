FROM inbolt/base_image:8f1b944

RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/detectronDocker/detectron2

RUN pip install --upgrade imageio && pip install --upgrade scipy
RUN pip install einops && pip install timm && pip install shapely

RUN cd /app/detectronDocker/detectron2/albumentations && python3 -m pip install .

RUN pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
RUN pip install opencv-python
RUN pip uninstall detectron2 --yes
RUN pip install torchviz
#RUN pip install --upgrade torch torchvision

RUN cd /app/detectronDocker && rm -rf build/ **/*.so && python3 -m pip install -e detectron2
