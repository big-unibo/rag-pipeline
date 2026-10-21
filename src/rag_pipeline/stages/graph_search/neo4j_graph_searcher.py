from __future__ import annotations

import json
import re
from pathlib import Path
from rag_pipeline.models import Chunk, Question, RetrievedChunk
from rag_pipeline.settings import resolve_path


class Neo4jGraphSearcher:
    def __init__(self, config: dict, credentials: dict):
        from neo4j import GraphDatabase

        neo4j = credentials.get("neo4j", {})
        self.uri = config.get("uri") or neo4j.get("uri", "bolt://localhost:7687")
        self.user = config.get("user") or neo4j.get("user", "neo4j")
        self.password = config.get("password") or neo4j.get("password", "ragpassword")
        self.top_k = int(config.get("top_k", 5))
        self.schema = load_schema(config.get("schema_path", "resources/datasets/movies/graph/schema.json"))
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def search(self, question: Question) -> list[RetrievedChunk]:
        # 1. Entity retrieval: match_schema does a transparent, non-LLM lookup.
        #    It scans the question for aliases listed in the selected dataset schema.
        #    Example: "The Matrix" maps to label=Movie, property=title.
        matches = match_schema(question.text, self.schema)
        if not matches:
            return []

        # 2. Query construction: use only schema-approved labels/properties in
        #    the Cypher text, and keep the matched entity values as parameters.
        query, params = build_schema_query(matches, self.top_k)
        with self.driver.session() as session:
            rows = session.run(query, params).data()

        # 3. Output normalization: every matching graph triple becomes a
        #    RetrievedChunk object. The chunk text is the triple itself; linked
        #    document paths stay in metadata for debugging and CSV export.
        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            text = format_graph_result(row)
            document_id = str(row.get("entity_id", "NA"))
            chunk_id = f"{row.get("entity_id", "NA")}:{row.get("relationship_id", "NA")}:{row.get("neighbor_id", "NA")}"
            chunk = Chunk(
                id=f"{chunk_id}",
                document_id=document_id,
                text=text,
                metadata={
                    "chunk_id": chunk_id,
                    "source_type": "graph",
                    "cypher": query,
                },
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(row.get("score", 0.0)),
                    rank=rank,
                    metadata={"source_type": "graph"},
                )
            )
        return results


def load_schema(path: str | Path) -> dict:
    schema_path = resolve_path(path)
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def match_schema(question: str, schema: dict) -> list[dict[str, str]]:
    # Entity extraction is intentionally transparent for teaching.
    #
    # The schema JSON has entries like:
    #   label: Movie
    #   search_properties: ["title", "tagline"]
    #   aliases: ["The Matrix", "A Few Good Men"]
    #
    # If the question contains "The Matrix", this function returns:
    #   {"label": "Movie", "property": "title", "value": "The Matrix"}
    #
    # This is simple phrase matching, not NLP entity recognition. To teach a
    # different approach, replace this function with spaCy, an LLM extractor,
    # fuzzy matching, embeddings over schema values, or a dictionary lookup.
    text = question.casefold()
    matches: list[dict[str, str]] = []
    for node in schema.get("nodes", []):
        label = safe_identifier(node["label"])
        aliases = node.get("aliases", [])
        aliases_lower = [alias.casefold() for alias in aliases]

        # Prefer full aliases first so multi-word entities such as
        # "A Few Good Men" stay together as one graph entity.
        matched_aliases = [alias for alias in aliases if alias.casefold() in text]

        # Fallback for one-token aliases. This keeps the example easy to extend
        # while avoiding generic labels such as "Movie" as entity values.
        matched_terms = matched_aliases or [term for term in keywords(question) if term in aliases_lower]

        # A matched value may be searched across multiple properties. For a
        # Movie node, "The Matrix" can match either title or tagline.
        for term in matched_terms:
            for prop in node.get("search_properties", []):
                matches.append({"label": label, "property": safe_identifier(prop), "value": term})
    return matches


def build_schema_query(
    matches: list[dict[str, str]], top_k: int
) -> tuple[str, dict]:
    # Cypher labels/properties cannot be parameterized, so they come only from
    # schema.json after identifier validation. User text stays in parameters.
    if not matches:
        query = """
        MATCH (entity)-[e]->(n)
        OPTIONAL MATCH (n)-[:HAS_EVIDENCE]->(evidence:Document)
        RETURN
          labels(entity)[0] AS entity_label,
          properties(entity) AS entity,
          type(e) AS relationship,
          properties(e) AS relationship_properties,
          labels(n)[0] AS neighbor_label,
          properties(n) AS neighbor,
          properties(evidence) AS evidence,
          0.0 AS score
        LIMIT $top_k
        """
        return query, {"top_k": top_k}

    # Group matches by label so that properties for the same label can be
    # combined with OR, while different labels use UNION.
    matches_by_label: dict[str, list[tuple[str, str]]] = {}

    for index, match in enumerate(matches):
        label = match["label"]
        prop = match["property"]
        value_name = f"value_{index}"

        matches_by_label.setdefault(label, []).append((prop, value_name))

    params = {"top_k": top_k}
    clauses: list[str] = []

    # Keep the original match ordering for parameter assignment.
    for index, match in enumerate(matches):
        params[f"value_{index}"] = match["value"]

    for label, property_matches in matches_by_label.items():
        where_conditions = [
            f"toLower(toString(entity.{prop})) CONTAINS toLower(${value_name})"
            for prop, value_name in property_matches
        ]
        where_clause = "\n                OR\n                ".join(
            where_conditions
        )
        clauses.append(
            f"""
            MATCH (entity:{label})-[e]->(n)
            WHERE {where_clause}
            OPTIONAL MATCH (n)-[:HAS_EVIDENCE]->(evidence:Document)
            RETURN
              id(entity) AS entity_id,
              labels(entity)[0] AS entity_label,
              properties(entity) AS entity,
              id(e) AS relationship_id,
              type(e) AS relationship,
              properties(e) AS relationship_properties,
              labels(n)[0] AS neighbor_label,
              id(n) AS neighbor_id,
              properties(n) AS neighbor,
              properties(evidence) AS evidence,
              1.0 AS score
            """
        )
    query = "CALL {\n" + "\nUNION\n".join(clauses) + "\n}\nRETURN *\nLIMIT $top_k"
    return query, params



def format_graph_result(row: dict) -> str:
    entity_label = row.get("entity_label", "Entity")
    neighbor_label = row.get("neighbor_label", "Node")
    entity = row.get("entity", {})
    neighbor = row.get("neighbor", {})
    relationship = row.get("relationship", "RELATED_TO")
    relationship_properties = row.get("relationship_properties") or {}
    return (
        f"({entity_label} {entity})-[:{relationship} {relationship_properties}]->"
        f"({neighbor_label} {neighbor})"
    )


def keywords(text: str) -> list[str]:
    return [word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)]


def safe_identifier(value: str) -> str:
    # Defensive check for labels/properties loaded from schema.json. This keeps
    # dynamic Cypher readable without letting arbitrary strings become code.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid Cypher identifier in graph schema: {value}")
    return value
