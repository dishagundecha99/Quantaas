FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY constants.py run.py  /app

COPY backend_server /app/backend_server

ENV PYTHONPATH "${PYTHONPATH}:/app/"

ENTRYPOINT [ "python3", "run.py"]