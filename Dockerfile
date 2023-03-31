FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /app

RUN apt-get update

RUN apt-get install -y zlib1g-dev libjpeg-dev build-essential libssl-dev libffi-dev

RUN python3 -m pip install --upgrade pip

COPY . .
RUN pip install -r requirements.txt

RUN mkdir ./weights
RUN mkdir ./temp

WORKDIR /app/weights
RUN gdown 'https://drive.google.com/u/0/uc?id=1qcgXiaAgdvSmvU3St9cJsguK6067KcjM&export=download' -O weights.zip

RUN unzip weights.zip
RUN rm weights.zip

EXPOSE 5005

WORKDIR /app

CMD python3 run.py --source live
