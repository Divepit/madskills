FROM python:3.10-slim

# Set the working directory
WORKDIR /app

COPY requirements.txt /app/
COPY setup.py /app/
COPY README.md /app/

RUN apt-get update
RUN apt-get -y install gcc
RUN pip3 install -e .

COPY . /app

CMD ["python3", "/app/graph_neural_network/graph_generator.py"]
