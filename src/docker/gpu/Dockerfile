FROM tensorflow/tensorflow:1.9.0-gpu-py3
MAINTAINER Juan Luis Rosa <juanluisrosaramos@gmail.com>
RUN apt update && \
    apt install -y --no-install-recommends \
        python3-tk \
        vim \
        ssh \
        git && \
    rm -rf /var/lib/apt/lists/*

#ADD . /app
#WORKDIR /app
RUN mkdir /data
COPY /data/ /data

RUN mkdir /app
COPY /src/ /app
WORKDIR /app


RUN pip3 install -r docker/requirements.txt
RUN export PYTHONPATH=$PYTHONPATH:/app
ENV PYTHONPATH=/app

# TensorBoard
EXPOSE 6006

# JupyterLab
#EXPOSE 8888

CMD ["/bin/bash"]
