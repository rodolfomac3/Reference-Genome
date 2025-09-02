# syntax=docker/dockerfile:1
FROM python:3.11-slim

# system deps needed by reportlab/primer3/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Render injects $PORT; gunicorn binds to it
ENV PORT=10000
EXPOSE 10000

# if your entry file or Flask instance name differs, change app:app accordingly
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
