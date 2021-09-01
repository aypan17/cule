FROM jupyter/tensorflow-notebook

USER root
RUN apt-get update -y && \ 
    apt-get install -y xvfb && \
    apt-get install -y python-opengl

USER ${NB_USER}
RUN pip install gym[atari] pyvirtualdisplay

RUN apt-get install -y cmake && \
    apt-get install -y zlib1g zlib1g-dev

RUN conda install swig
RUN pip install box2d-py atari_py pystan

COPY ./example/env_render.ipynb /home/${NB_USER}/env_render.ipynb
COPY ./example/18_reinforcement_learning.ipynb /home/${NB_USER}/18_reinforcement_learning.ipynb


