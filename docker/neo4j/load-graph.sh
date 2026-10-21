#!/usr/bin/env bash
set -euo pipefail

password="${RAG_NEO4J_PASSWORD:-}"
if [ -z "${password}" ] && [ -n "${NEO4J_AUTH:-}" ]; then
  password="${NEO4J_AUTH#*/}"
fi
if [ -z "${password}" ]; then
  echo "Set RAG_NEO4J_PASSWORD or NEO4J_AUTH before starting Neo4j" >&2
  exit 1
fi

neo4j-admin dbms set-initial-password "${password}" || true
neo4j console &
neo4j_pid="$!"

for _ in $(seq 1 60); do
  if cypher-shell -u neo4j -p "${password}" "RETURN 1" >/dev/null 2>&1; then
    cypher-shell -u neo4j -p "${password}" -f /graph.cypher
    wait "${neo4j_pid}"
  fi
  sleep 2
done

echo "Neo4j did not become ready in time" >&2
exit 1
