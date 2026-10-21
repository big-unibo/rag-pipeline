// Editable teaching graph for hybrid RAG.
// This compact Neo4j Movies-style graph stores symbolic relationships between
// movies, people, and genres. Evidence points to documents in this dataset.
//
// Graph search still returns RetrievedChunk objects to the rest of the pipeline,
// but the graph itself stays document-level and easy to edit by hand.

CREATE CONSTRAINT document_path IF NOT EXISTS
FOR (d:Document)
REQUIRE d.path IS UNIQUE;

CREATE CONSTRAINT person_name IF NOT EXISTS
FOR (p:Person)
REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT movie_title IF NOT EXISTS
FOR (m:Movie)
REQUIRE m.title IS UNIQUE;

CREATE CONSTRAINT genre_name IF NOT EXISTS
FOR (g:Genre)
REQUIRE g.name IS UNIQUE;

MERGE (doc:Document {
  path: 'resources/datasets/movies/documents/The Matrix - Wikipedia.pdf'
})
SET doc.title = 'The Matrix - Wikipedia.pdf',
    doc.description = 'Compact PDF corpus for The Matrix - Wikipedia.pdf.'

MERGE (matrix:Movie {
  title: 'The Matrix',
  released: 1999,
  tagline: 'Welcome to the Real World'
})
MERGE (reload:Movie {
  title: 'The Matrix Reloaded',
  released: 2003,
  tagline: 'Free your mind'
})
MERGE (devil:Movie {
  title: "The Devil's Advocate",
  released: 1997,
  tagline: 'Evil has its winning ways'
})
MERGE (cloud:Movie {
  title: 'Cloud Atlas',
  released: 2012,
  tagline: 'Everything is connected'
})
MERGE (fewgood:Movie {
  title: 'A Few Good Men',
  released: 1992,
  tagline: "In the heart of the nation's capital, one man will stop at nothing to keep his honor, and one will stop at nothing to find the truth"
})
MERGE (topgun:Movie {
  title: 'Top Gun',
  released: 1986,
  tagline: 'I feel the need, the need for speed'
})

MERGE (scifi:Genre {name: 'Science Fiction'})
MERGE (drama:Genre {name: 'Drama'})
MERGE (legal:Genre {name: 'Legal Drama'})
MERGE (action:Genre {name: 'Action'})

MERGE (keanu:Person {name: 'Keanu Reeves', born: 1964})
MERGE (carrie:Person {name: 'Carrie-Anne Moss', born: 1967})
MERGE (laurence:Person {name: 'Laurence Fishburne', born: 1961})
MERGE (lana:Person {name: 'Lana Wachowski', born: 1965})
MERGE (lilly:Person {name: 'Lilly Wachowski', born: 1967})
MERGE (al:Person {name: 'Al Pacino', born: 1940})
MERGE (tom:Person {name: 'Tom Cruise', born: 1962})
MERGE (demi:Person {name: 'Demi Moore', born: 1962})
MERGE (jack:Person {name: 'Jack Nicholson', born: 1937})
MERGE (rob:Person {name: 'Rob Reiner', born: 1947})
MERGE (aaron:Person {name: 'Aaron Sorkin', born: 1961})

MERGE (keanu)-[:ACTED_IN {role: 'Neo'}]->(matrix)
MERGE (carrie)-[:ACTED_IN {role: 'Trinity'}]->(matrix)
MERGE (laurence)-[:ACTED_IN {role: 'Morpheus'}]->(matrix)
MERGE (matrix)-[:HAS_ACTOR {role: 'Neo'}]->(keanu)
MERGE (matrix)-[:HAS_ACTOR {role: 'Trinity'}]->(carrie)
MERGE (matrix)-[:HAS_ACTOR {role: 'Morpheus'}]->(laurence)
MERGE (lana)-[:DIRECTED]->(matrix)
MERGE (lilly)-[:DIRECTED]->(matrix)
MERGE (matrix)-[:DIRECTED_BY]->(lana)
MERGE (matrix)-[:DIRECTED_BY]->(lilly)
MERGE (matrix)-[:IN_GENRE]->(scifi)
MERGE (matrix)-[:SEQUEL]->(reload)

MERGE (keanu)-[:ACTED_IN {role: 'Kevin Lomax'}]->(devil)
MERGE (al)-[:ACTED_IN {role: 'John Milton'}]->(devil)
MERGE (devil)-[:HAS_ACTOR {role: 'Kevin Lomax'}]->(keanu)
MERGE (devil)-[:HAS_ACTOR {role: 'John Milton'}]->(al)
MERGE (devil)-[:IN_GENRE]->(drama)

MERGE (tom)-[:ACTED_IN {role: 'Lt. Daniel Kaffee'}]->(fewgood)
MERGE (demi)-[:ACTED_IN {role: 'Lt. Cmdr. JoAnne Galloway'}]->(fewgood)
MERGE (jack)-[:ACTED_IN {role: 'Col. Nathan R. Jessup'}]->(fewgood)
MERGE (fewgood)-[:HAS_ACTOR {role: 'Lt. Daniel Kaffee'}]->(tom)
MERGE (fewgood)-[:HAS_ACTOR {role: 'Lt. Cmdr. JoAnne Galloway'}]->(demi)
MERGE (fewgood)-[:HAS_ACTOR {role: 'Col. Nathan R. Jessup'}]->(jack)
MERGE (rob)-[:DIRECTED]->(fewgood)
MERGE (aaron)-[:WROTE]->(fewgood)
MERGE (fewgood)-[:DIRECTED_BY]->(rob)
MERGE (fewgood)-[:WRITTEN_BY]->(aaron)
MERGE (fewgood)-[:IN_GENRE]->(legal)

MERGE (tom)-[:ACTED_IN {role: 'Maverick'}]->(topgun)
MERGE (topgun)-[:HAS_ACTOR {role: 'Maverick'}]->(tom)
MERGE (topgun)-[:IN_GENRE]->(action)

MERGE (tom)-[:ACTED_IN]->(cloud)
MERGE (lana)-[:DIRECTED]->(cloud)
MERGE (lilly)-[:DIRECTED]->(cloud)
MERGE (cloud)-[:HAS_ACTOR]->(tom)
MERGE (cloud)-[:DIRECTED_BY]->(lana)
MERGE (cloud)-[:DIRECTED_BY]->(lilly)
MERGE (cloud)-[:IN_GENRE]->(scifi)
MERGE (cloud)-[:IN_GENRE]->(drama)

MERGE (matrix)-[:HAS_EVIDENCE]->(doc)
MERGE (reload)-[:HAS_EVIDENCE]->(doc)
MERGE (keanu)-[:HAS_EVIDENCE]->(doc)

WITH [
  doc, matrix, reload, devil, cloud, fewgood, topgun,
  scifi, drama, legal, action,
  keanu, carrie, laurence, lana, lilly, al, tom, demi, jack, rob, aaron
] AS sample_nodes
UNWIND sample_nodes AS sample_node
SET sample_node.dataset = 'neo4j-movies';
