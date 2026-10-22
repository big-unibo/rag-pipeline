# RAG pipeline

Docker-first RAG pipeline with explicit, configurable stages:

1. load documents
2. chunk
3. embed
4. retrieve with vector search and graph search
5. rerank the combined candidates
6. prompt
7. answer
8. evaluate

## Run

The local run is exactly what GitHub CI runs:

```bash
docker compose up --build --exit-code-from ci ci
docker compose cp ci:/workspace/outputs ./outputs
```

It runs everything in containers:

- builds the Python image from `w4bo/python:1.0.7`
- starts Postgres with `pgvector`
- starts Neo4j and loads the movie dataset graph
- starts MLflow
- starts Ollama and pulls the configured models
- indexes the documents from the selected dataset
- runs that dataset's questions
- writes `outputs/evaluation.json`
- writes `outputs/evaluation.csv`
- writes `outputs/retrieved_chunks.csv`
- writes `outputs/run_config.csv`
- logs one parent MLflow run per matrix variant, plus nested `indexing` and `question_run` runs

Each run keeps the existing outputs appending new results labeling with a unique `run_id`.

The default config runs a small matrix over the evaluation settings.

No local Python install is required. The default config uses Ollama locally.

## Develop

```bash
docker compose up -d
```

Then run commands inside the app container:

```bash
docker compose exec app rag-pipeline index --config config/pipeline.yml
docker compose exec app rag-pipeline run --config config/pipeline.yml --output outputs/evaluation.json
```

To run every matrix combination manually:

```bash
docker compose exec app rag-pipeline matrix --config config/pipeline.yml --output outputs/evaluation.json
```

Copy container outputs back to the host when you need local files:

```bash
docker compose cp app:/workspace/outputs ./outputs
```

For debugging retrieval, inspect `outputs/retrieved_chunks.csv`. It has one row per retrieved item with run name, matrix options, question id, rank, source, source path, distance, rerank score, and text.

## Working Example

The checked-in example is the familiar Neo4j Movies domain, kept intentionally small and fully editable:

- `resources/datasets/movies/documents/` contains movie facts as ordinary document text
- `resources/datasets/movies/graph/graph.cypher` creates the Neo4j movie graph
- `resources/datasets/movies/graph/schema.json` explains the graph labels, properties, aliases, and relationships
- `resources/datasets/movies/dataset.yml` contains dataset metadata, questions and ground truth

For a question such as "Who played Neo in The Matrix?", `graph_search` scans `schema.json`, matches phrases such as `The Matrix` to schema aliases, builds a Cypher query shaped like `MATCH (entity:Movie)-[e]->(n) WHERE ... RETURN entity, e, n`, then converts each returned graph triple into a normal retrieved item.

To reuse an existing Postgres index:

```bash
RAG_INDEX_DOCUMENTS=false docker compose up --build --exit-code-from ci ci
```

MLflow UI: <http://localhost:5000>
Ollama API: <http://localhost:11434>

The MLflow server is configured to allow both browser access from localhost and Docker-internal access from `http://mlflow:5000`.
In MLflow, filter the `stage` tag to separate the parent matrix runs from the nested `indexing` and `question_run` runs. The parent run still logs aggregate duration metrics, while the child runs log their own `stage.duration_seconds`.

## Configure

Pipeline choices live in YAML:

- `config/pipeline.yml`: reusable pipeline stages, matrix, hybrid search, outputs, and MLflow tracking
- `resources/datasets/<dataset>/dataset.yml`: dataset-specific paths, questions, and metadata

Each stage has a `strategy` key containing the implementation class name. Classes are discovered automatically under `rag_pipeline.stages`, so adding a class module makes it available without editing a central class list. The `matrix` section defines combinations of overrides to run. With the current config, Docker runs:

```text
2 evaluation variants = 2 pipeline runs
```

The default embedder is `ollama` with `embeddinggemma`. The default reranker, answerer, and LLM-as-a-judge use `ollama` with `gemma4`.

Question manifests can retain selected source fields as question metadata:

```yaml
questions:
  metadata:
    include: [source, question_type]
```

Evaluation output includes fields by default. Suppress bulky or irrelevant fields per format with `evaluation.output.<format>.ignore`:

```yaml
evaluation:
  output:
    json:
      ignore: [answer_metadata.raw_response]
    csv:
      ignore: [prompt, retrieved_chunks, answer_metadata.raw_response]
```

Postgres defaults are in `config/pipeline.yml` and Docker overrides them for container networking.

Hybrid RAG is controlled by the `graph_search` section. Neo4j loads the editable local file `resources/datasets/<dataset>/graph/graph.cypher`, and `resources/graph/schema.json` describes the labels, properties, aliases, and relationships used by schema-guided graph search. Because Compose volumes are disabled, rebuild after editing either graph file. Vector chunks and graph triples are then passed to the Ollama reranker.

To add another dataset, create a directory under `resources/datasets/`:

```text
resources/datasets/more-movies/
├── dataset.yml
├── documents/
└── questions.yml
```

`dataset.yml`:

```yaml
name: more-movies

documents:
  input_dir: resources/datasets/movies/documents
  glob: "*.pdf"

questions:
  inline:
    - id: q1
      text: Who directed The Matrix?
      ground_truth: Short factual expected answer.
```

The question manifest supports also the following schema:

```yaml
questions:
  source:
      kind: file
    path: resources/datasets/medical/questions.json
    format: json
  schema:
    id: id
    text: question
    ground_truth: answer
```

Use `inline` instead of `source` when questions are defined directly in the
dataset manifest. The two options are mutually exclusive.

Database-backed questions use the same schema mapping:

```yaml
questions:
  source:
    kind: database
    table: question
    ids: [1, 2, 3]
  schema:
    id: q_id
    text: question_text
    ground_truth: ground_truth
```

Then run it without changing the pipeline:

```bash
RAG_CORPUS_CONFIG=resources/datasets/movies/dataset.yml docker compose up --build --exit-code-from ci ci
docker compose cp ci:/workspace/outputs ./outputs
```

Useful environment overrides:

For settings defined in `config/pipeline.yml`, the precedence is YAML value,
then environment variable, then the application default. To let an environment
variable control a setting, leave that setting out of the YAML configuration.

- `MLFLOW_TRACKING_URI`
- `POSTGRES_HOST`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OLLAMA_HOST`

## Structure

```text
.
├── .devcontainer/               # Devcontainer definition
├── .github/workflows/           # CI, using the same docker compose command
├── config/
│   └── pipeline.yml             # Reusable pipeline and matrix config
├── docker/
│   ├── neo4j/                   # Neo4j image and loader
│   └── postgres-init/           # pgvector schema
├── resources/
│   └── datasets/                # Document/question/graph datasets
│       ├── medical/
│       └── movies/
├── scripts/
│   └── run.py                   # CI/local runner
├── src/
│   └── rag_pipeline/
│       ├── cli.py
│       ├── pipeline.py
│       ├── runner.py
│       ├── results.py
│       ├── settings.py
│       ├── models.py
│       ├── mlflow_tracking.py
│       └── stages/
│           ├── answering/      # answer generation
│           ├── chunking/       # document splitting
│           ├── embedding/      # vector embeddings
│           ├── evaluation/     # scoring and judging
│           ├── graph_search/   # graph retrieval
│           ├── prompting/      # prompt assembly
│           ├── ranking/        # reranking candidates
│           └── retrieval/      # vector retrieval
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Evaluate exported runs

For comparisons that should remain separate from the live pipeline outputs, put
the semicolon-delimited exports in one directory as `questions.csv`,
`answers.csv`, and `chunks.csv`, then run. Only the core columns are required;
additional columns are carried into the corresponding model's metadata:

```bash
rag-pipeline evaluate-csv --config config/pipeline.yml \
  --input-dir outputs/input --output-dir outputs/comparison
```

The evaluator can also be run directly as a module:

```bash
python -m rag_pipeline.csv_evaluator --config config/pipeline.yml \
  --input-dir outputs/input --output-dir outputs/comparison
```

The command loads the `evaluation` section from the pipeline YAML and passes
each CSV record through the configured evaluation stage using the shared
`Question`, `Answer`, `Prompt`, `Chunk`, and `RetrievedChunk` models. It joins
rows by question and by `(run_id, run_approach, question_id)` for chunks. It
writes:

- `evaluation.csv`: the same flattened evaluation format used by the standard
  pipeline output logic.
- `retrieved_chunks.csv`: the same flattened chunk format used by the standard
  pipeline output logic.

The input filenames can be changed with `--questions`, `--answers`, and
`--chunks`; no existing files under `outputs/` are appended or overwritten
unless they are inside the explicit `--output-dir`.
