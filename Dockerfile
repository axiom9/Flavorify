FROM python:3.9-bullseye
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 unzip -y
RUN pip install -r requirements.txt

RUN python src/app/app.py