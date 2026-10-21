FROM neo4j:5-community

COPY resources/datasets/movies/graph/graph.cypher /graph.cypher
COPY docker/neo4j/load-graph.sh /load-graph.sh

RUN sed -i 's/\r$//' /load-graph.sh && chmod +x /load-graph.sh
