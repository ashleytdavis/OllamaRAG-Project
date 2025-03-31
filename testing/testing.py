import pandas as pd
import itertools
import time
import csv
import os

from src.UnifiedInjest import *
from src.search  import *

data_dir = "../data/"

# holds all different variables to test
configs = {"chunk_sizes": [200, 500, 1000],
           "chunk_overlap": [0, 50, 100],
           "text_prep": [True, False],
           "embedding_models": ["all-minilm", "nomic-embed-text", "mxbai-embed-large"],
           "DB_type": ["chroma", "qdrant", "redis"],
           "LLM_type": ["llama3.2", "mistral", "deepseek-r1"]
           }

# 50 wquestions and answers the AI will be tested on
questions = [
    "What is the worst-case time complexity of linear search? A) O(1) B) O(log n) C) O(n) D) O(n log n",
    "In a database context, what is a ‘record’? A) A single field B) A row in a table C) A column header D) A primary key",
    "Which data structure is typically faster for random access? A) Arrays B) Linked Lists C) Hash Tables D) Trees",
    "Which ACID property ensures that a transaction takes a database from one consistent state to another? A) Atomicity B) Consistency C) Isolation D) Durability",
    "What does ‘dirty read’ refer to in transaction isolation? A) Reading uncommitted changes from another transaction B) Reading stale data C) Reading data from a backup D) Reading data from cache",
    "In vertical scaling, what is the approach used to improve performance? A) Adding more nodes B) Upgrading to a more powerful system C) Sharding the data D) Using distributed computing",
    "In leader-based replication, which node handles all writes from clients? A) Follower B) Peer C) Leader D) Replica",
    "What is a key challenge when using asynchronous replication? A) High latency B) Inconsistency window C) Low throughput D) Increased disk usage",
    "Which replication method uses a byte-level log of every change to the database? A) Statement-based B) Logical log C) Write-ahead log D) Trigger-based",
    "In a B+ tree, what operation is performed when a leaf node is full and a new key is inserted? A) Merge B) Rotation C) Splitting D) Rebalancing",
    "In a B+ tree of order 4, what is the maximum number of keys an internal node can contain? A) 2 B) 3 C) 4 D) 5",
    "During B+ tree insertion, why is the smallest key from the new leaf copied to the parent? A) To balance the tree B) To update the search path C) To maintain sorted order D) To mark the split",
    "Which best describes a key-value store in NoSQL databases? A) Supports complex joins B) Stores documents C) Maps each key to a single value D) Is a relational database",
    "What does BASE stand for in NoSQL systems? A) Basic, Available, Soft-state, Eventual consistency B) Basically Available, Soft state, Eventual consistency C) Balanced, Asynchronous, Scalable, Event-driven D) Binary, Accessible, Secure, Extensible",
    "According to the CAP theorem, a distributed database cannot simultaneously guarantee which three properties? A) Consistency, Availability, Scalability B) Consistency, Availability, Partition Tolerance C) Performance, Consistency, Partition Tolerance D) Availability, Durability, Partition Tolerance",
    "Which Python library is commonly used to interface with Redis? A) pymongo B) redis-py C) neo4j-driver D) sqlalchemy",
    "Which Redis command increments a numeric value stored at a key? A) INCR B) SET C) APPEND D) MGET",
    "What is the primary data structure used by Redis? A) Document B) Key-Value pairs C) Graph D) Relational table",
    "What is the benefit of using Redis pipelines? A) Increase security B) Reduce network overhead by batching commands C) Ensure data persistence D) Simplify data types",
    "In a document database, data is typically stored in which format? A) XML B) CSV C) JSON D) SQL",
    "What is BSON in MongoDB? A) A query language B) A binary representation of JSON C) A document structure D) A type of index",
    "Which feature is supported by MongoDB? A) Requires a predefined schema B) Supports ACID transactions only C) Supports replica sets for automatic failover D) Uses SQL for querying",
    "In PyMongo, which method is used to insert a single document into a collection? A) insert_many() B) insert_one() C) update() D) find()",
    "Which function from bson.json_util is used to serialize MongoDB query results? A) dumps() B) loads() C) json_encode() D) to_json()",
    "After connecting with PyMongo, how do you access a specific database? A) client.get_database(‘dbName’) B) client[‘dbName’] C) client.connect(‘dbName’) D) client.database(‘dbName’)",
    "In a graph database, what are the individual entities that store data called? A) Edges B) Nodes C) Vertices D) Both B and C",
    "What is the process of finding the shortest path between two nodes in a graph called? A) Graph partitioning B) Graph traversal C) Pathfinding D) Centrality",
    "In the property graph model, which component can have key-value pairs? A) Only nodes B) Only edges C) Both nodes and edges D) Neither",
    "What is the query language used by Neo4j? A) SQL B) Cypher C) Gremlin D) SPARQL",
    "Which is a key characteristic of Neo4j? A) Schema-less document storage B) Graph database with ACID compliance C) Key-value store D) Columnar storage",
    "What is the purpose of the APOC plugin in Neo4j? A) Manage transactions B) Extend Cypher with additional procedures C) Provide security D) Replicate data",
    "What is the benefit of a high branching factor in a B-tree? A) Increased tree height B) Fewer disk accesses C) More frequent splits D) Reduced space efficiency",
    "What property do all leaf nodes in a B-tree share? A) They are at varying levels B) They are all at the same level C) They contain internal pointers D) They are unsorted",
    "Where are the actual data records stored in a B+ tree? A) In the internal nodes B) In the root node C) In the leaf nodes D) Evenly distributed",
    "When a B-tree node is full during insertion, what operation is typically performed? A) Merge B) Split C) Rotate D) Rebalance",
    "According to ‘A Note on Distributed Computing’, what is a major challenge in distributed systems? A) Memory leaks B) Partial failure C) Infinite loops D) Single-threaded execution",
    "Why is the latency difference significant in distributed computing? A) Remote calls are much faster B) Remote calls require more disk I/O C) Remote calls are orders of magnitude slower than local calls D) Remote calls are executed in parallel",
    "What flawed assumption does the note on distributed computing criticize in unified object models? A) All objects can be implemented in C++ B) Location does not affect object interactions C) Network protocols are always reliable D) Memory is unlimited",
    "Why is locality important in B-trees? A) It minimizes disk I/O B) It improves network latency C) It increases memory usage D) It reduces caching needs",
    "What invariant is maintained by a B-tree regarding paths from the root to a leaf? A) All paths have equal length B) All paths contain the same number of keys C) All nodes have the same branching factor D) All leaves are internal nodes",
    "What is the result of an inorder traversal of a binary search tree? A) Non-decreasing order of keys B) Non-increasing order C) Random order D) Reverse sorted order",
    "In the worst-case scenario, what is the time complexity of searching in a degenerate binary search tree? A) O(1) B) O(log n) C) O(n) D) O(n log n",
    "Which BST traversal visits nodes in the order: left subtree, current node, right subtree? A) Preorder B) Inorder C) Postorder D) Level order",
    "What defines the AVL property in a binary search tree? A) The tree is complete B) The heights of left and right subtrees differ by at most 1 C) All leaves are at the same level D) The tree is perfectly balanced",
    "What is the purpose of an LL rotation in AVL trees? A) To rotate a right-heavy subtree B) To rotate a left-heavy subtree C) To swap child nodes D) To merge two subtrees",
    "What is the worst-case time complexity for rebalancing an AVL tree after insertion? A) O(1) B) O(log n) C) O(n) D) O(n log n",
    "If an AVL tree node’s left subtree has height 3 and its right subtree has height 1, does it satisfy the AVL property? A) Yes B) No C) Only if balanced later D) It depends on the node count",
    "What is a key disadvantage of a degenerate binary search tree? A) Low memory usage B) O(n) search time C) Easy balancing D) Uniform height",
    "Which database supports a document-oriented data model with flexible schema? A) MySQL B) PostgreSQL C) MongoDB D) Oracle",
    "In a leaderless replication model, what strategy is employed? A) All writes go to a single leader B) Writes are "
    "coordinated through consensus C) No single leader exists and any node can process writes D) Writes are first logged "
    "then merged "
]

answers = [
    "c", "b", "a", "b", "a", "b", "c", "b", "c", "c",
    "b", "b", "c", "b", "b", "b", "a", "b", "b", "c",
    "b", "b", "c", "b", "b", "d", "c", "c", "b", "b",
    "b", "b", "b", "b", "c", "b", "c", "b", "a", "a",
    "a", "c", "b", "b", "b", "b", "b", "c", "c", "c"
]

def CreateCombos(configs):
    """
    Creates a df that has every unique conbination of the config
    """
    # Get keys and values
    keys = list(configs.keys())
    values = list(configs.values())

    # Generate all combinations
    combinations = list(itertools.product(*values))

    # Create the DataFrame
    df = pd.DataFrame(combinations, columns=keys)

    return df


def OneRun(config_data):
    """
    Given a row of a config dataframe, will train a model and ask a query based on the params
    """

    # All the variables exrtraed
    chunks = config_data.chunk_sizes
    overlap = config_data.chunk_overlap
    text_prep = config_data.text_prep
    embedding = config_data.embedding_models
    DB_type = config_data.DB_type
    LLM_type = config_data.LLM_type

    # Ingestion, keeps track of time
    ingest_start_time = time.time()
    ingest = UnifiedIngest(db_type=DB_type, embedding_model=embedding, preprocess=text_prep)
    ingest.process_pdfs(data_dir, chunk_size=chunks, overlap=overlap)
    ingest_end_time = time.time()

    ingest_time = round(ingest_end_time - ingest_start_time, 2)


    print("**Done Indexing**")
    print("")
    print("")

    search_start_time = time.time()
    correct = 0

    # Iterates through all 50 questions...
    for idx, q in enumerate(questions):

        # If the answer is correct, add 1
        response = GenerateResponse(embedding, DB_type, LLM_type, q)
        response_letter = response[0].lower()

        if response_letter == answers[idx]:
            correct += 1

    # Calculate the score the AI received on the test
    percentScore = round(correct/50,4)

    search_end_time = time.time()
    search_time = round(search_end_time - search_start_time, 2)

    # Write results to CSV
    with open("results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            chunks,
            overlap,
            text_prep,
            embedding,
            DB_type,
            LLM_type,
            ingest_time,
            search_time,
            percentScore
        ])

    print("**New Line Added**")
    print("")
    print("")


def main():
    # Initialize results CSV if it doesn't already exist
    csv_file = "results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["chunks", "overlap", "text_prep", "embedding_model", "DB_type", "LLM_type", "ingest_time",
                             "search_time", "percentScore"])

    # Create all the unique test cases
    combos = CreateCombos(configs)


    # iterate through each combination and write it to CSV
    for row in combos.itertuples(index=False):
        OneRun(row)



if __name__ == '__main__':
    main()



