FROM inbolt/base_image:c29a014
RUN pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
RUN pip install opencv-python

RUN git clone --recursive https://github.com/Lavreniuk/EVP/

RUN pip install -r EVP/requirements.txt
RUN pip install torchtext==0.6.0
RUN pip install kornia==0.6
RUN pip install mmcv==1.7.1
RUN pip install timm>=0.5.4
RUN pip install omegaconf==2.1.1
RUN pip install einops==0.7.0
RUN pip install pytorch-lightning==1.4.2
RUN pip install torchmetrics==0.6.0
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git#egg=taming-transformers
RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
RUN pip install wget

