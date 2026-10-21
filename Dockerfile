FROM w4bo/python:1.0.7

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

COPY config ./config
COPY resources ./resources
COPY scripts ./scripts

ENV PYTHONPATH=/workspace/src
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "-m", "rag_pipeline", "--help"]
