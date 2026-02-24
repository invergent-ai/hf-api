FROM python:3.12.11-slim-bookworm

RUN apt-get update && apt-get install -y curl unzip && apt-get clean
RUN curl https://rclone.org/install.sh | bash

RUN mkdir /app && mkdir -p /root/.config/rclone
WORKDIR /app
COPY . /app

RUN pip install "torch==2.9.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

RUN rm -rf /root/.cache/pip

ENV RCLONE_CONFIG_LAKEFS_TYPE=s3
ENV RCLONE_CONFIG_LAKEFS_PROVIDER=Other
ENV RCLONE_CONFIG_LAKEFS_ENV_AUTH=false
ENV RCLONE_CONFIG_NO_CHECK_BUCKET=true

ENTRYPOINT ["/bin/bash", "-c"]
