FROM nvcr.io/nvidia/pytorch:22.01-py3

# RUN apt-get update -y && apt-get install -y gcc

RUN pip install transformers==4.36.0

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY run.py /app

COPY src /app/src

COPY data /app/data

ENTRYPOINT [ "python3", "run.py"]