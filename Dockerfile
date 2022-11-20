# a Dockerfile specifies how to build a Docker image

FROM continuumio/anaconda3:2022.10

ADD . /code
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "app.py" ]