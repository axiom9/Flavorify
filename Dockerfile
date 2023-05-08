FROM python:3.9-bullseye
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 unzip -y
RUN pip install -r requirements.txt

# CMD gunicorn --workers=4 --bind=0.0.0.0:$PORT app:app
# CMD gunicorn app:app --bind 0.0.0.0:$PORT --preload
CMD ["python3", "app.py"]