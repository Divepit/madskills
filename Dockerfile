FROM python:3.10-slim

# Set the working directory
WORKDIR /app

COPY requirements.txt /app/
COPY setup.py /app/
COPY README.md /app/

RUN pip3 install -e .

COPY . /app

CMD ["python3", "/app/NeuralNetwork/dataset_generation.py"]