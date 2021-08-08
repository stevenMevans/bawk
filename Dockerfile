# syntax=docker/dockerfile:1.0

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:1.0.1

# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ${BASE_IMAGE} as nemo-deps

# copy asr service source into a temp directory
WORKDIR /tmp/bawk
COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# override nemo installation with dependency from requirements.txt
RUN /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
RUN cd /tmp/bawk && pip install -r "requirements.txt"

# copy webapp into container for end user
WORKDIR /workspace/bawk
COPY . /workspace/bawk/
RUN pip install -r requirements.txt

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 5000
CMD ["flask", "run"]
