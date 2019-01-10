FROM hubertlegec/opencv-python:1.0

MAINTAINER Hubert Legęc <hubert.legec@gmail.com>

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

#RUN mkdir /data
#COPY /data/ /data

RUN mkdir /app
COPY /src/ /app
COPY /model/ /app/model
COPY /data /app/data
WORKDIR /app

RUN export PYTHONPATH=$PYTHONPATH:/app
CMD ["/bin/bash"]
