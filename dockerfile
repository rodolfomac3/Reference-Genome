# syntax=docker/dockerfile:1
FROM python:3.11-slim

# system deps for reportlab/primer3/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=10000
EXPOSE 10000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
