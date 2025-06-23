FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg flac && pip install -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

ENV PYTHONPATH=/app