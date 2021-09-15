ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.06-py3

FROM ${BASE_IMAGE} as nemo-deps
ARG DEBIAN_FRONTEND=noninteractive

# copy asr service source into a temp directory
WORKDIR /tmp/bawk
COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# override nemo installation with dependency from requirements.txt
RUN /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
RUN apt-get -y update && apt-get install -y libsndfile1 ffmpeg
RUN cd /tmp/bawk && pip install -r "requirements.txt"

# copy webapp into container for end user
WORKDIR /workspace/bawk
COPY . /workspace/bawk/
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["flask", "run"]