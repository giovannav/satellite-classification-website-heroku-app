# a Dockerfile specifies how to build a Docker image

FROM continuumio/anaconda3:2022.10

ADD . /code
WORKDIR /code

ENTRYPOINT [ "python", "app.py" ]