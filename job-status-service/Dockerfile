FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY constants.py run.py kafka_util.py mongo_util.py /app

ENTRYPOINT [ "python3", "run.py"]