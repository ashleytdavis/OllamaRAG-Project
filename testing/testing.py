import pandas as pd
import time
import csv
from allpairspy import AllPairs
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

# 25 questions (only based on information found in the provided PDFs) and their corresponding answers
questions = [
    "True/False: In a B+ tree, splitting a full leaf node and promoting the smallest key from the new node guarantees that all nodes remain at least half full, optimizing disk I/O by ensuring maximum data locality.",
    "True/False: The write-ahead log (WAL) replication method mandates that both the leader and all followers use an identical storage engine, which complicates system upgrades.",
    "True/False: In leaderless replication, write operations are centrally coordinated by a designated leader, ensuring strong consistency across replicas.",
    "True/False: Binary search trees (BSTs) inherently guarantee logarithmic search time regardless of the order in which keys are inserted.",
    "True/False: In an AVL tree, enforcing the balance condition that the height difference between left and right subtrees is at most 1 ensures that the tree's height remains O(log n) under worst-case insertion orders.",
    "True/False: In a B-tree, when a node becomes full during insertion, the standard recovery procedure is to merge the node with one of its siblings to maintain tree balance.",
    "True/False: According to the CAP theorem as described in the PDFs, during a network partition a distributed system must choose between consistency and availability.",
    "True/False: Document databases store data in JSON format in a way that inherently enforces a rigid schema, preventing any differences between documents in the same collection.",
    "True/False: Redis supports a variety of data types—including lists, sets, sorted sets, and hashes—which enhances its utility as an in-memory database.",
    "True/False: MongoDB's BSON format is optimized for human readability of stored documents.",
    "True/False: In the property graph model employed by Neo4j, both nodes and relationships can have multiple key-value properties to capture rich semantic details.",
    "True/False: Neo4j's query language, Cypher, is fundamentally based on pattern matching and significantly differs in syntax and approach from traditional SQL.",
    "True/False: Distributed computing systems face unique challenges such as partial failure and indeterminacy, necessitating specialized design considerations that are absent in local computing.",
    "True/False: The design of NFS, as discussed in the corpus, completely abstracts network failures so that its operation is indistinguishable from that of a local file system.",
    "True/False: Analysis of randomly built BSTs reveals that, under random key insertions, the average tree height is O(log n), ensuring efficient searches on average.",
    "True/False: In a binary search tree, an inorder traversal always produces the keys in non-decreasing (sorted) order.",
    "True/False: Vertical scaling involves upgrading to a more powerful machine—a process that is generally simpler than horizontal scaling, despite its inherent financial and physical limitations.",
    "True/False: Horizontal scaling in distributed systems typically relies on a shared-disk architecture to maintain data consistency across nodes.",
    "True/False: The replication strategies described in the corpus include statement-based replication, WAL, logical (row-based) logging, and trigger-based replication, each offering distinct trade-offs in consistency and error handling.",
    "True/False: In synchronous replication, the leader waits for acknowledgment from all followers before committing a write, which increases write latency.",
    "True/False: Asynchronous replication employs an eventual consistency model that allows high availability at the cost of temporary inconsistencies in read operations.",
    "True/False: Merging the computational models of local and distributed objects is straightforward since they share identical mechanisms for memory access and failure handling.",
    "True/False: The concept of 'local-remote' objects suggests that objects in different address spaces on the same machine can share many characteristics with local objects, thereby mitigating the impact of partial failure.",
    "True/False: The analysis using Jensen's inequality in the corpus demonstrates that the expected height of a randomly built BST grows linearly with the number of nodes.",
    "True/False: The architectural design of distributed databases aims to minimize disk block accesses, which is a key motivation for employing B-trees over traditional binary search trees."
]

answers = [
    "true",   # Q1: B+ tree splitting guarantees half-full nodes.
    "true",   # Q2: WAL requires identical storage engines.
    "false",  # Q3: Leaderless replication does not coordinate via a designated leader.
    "false",  # Q4: BSTs can degenerate if keys are inserted in order.
    "true",   # Q5: AVL tree balance condition ensures logarithmic height.
    "false",  # Q6: Standard procedure is splitting, not merging, a full B-tree node.
    "true",   # Q7: CAP theorem forces a trade-off during network partitions.
    "false",  # Q8: Document databases allow flexible schemas.
    "true",   # Q9: Redis supports multiple advanced data types.
    "false",  # Q10: BSON is optimized for efficient storage, not human readability.
    "true",   # Q11: Neo4j's property graph model supports properties on both nodes and edges.
    "true",   # Q12: Cypher is pattern-based and differs from SQL.
    "true",   # Q13: Distributed systems face unique challenges (partial failure, indeterminacy).
    "false",  # Q14: NFS does not completely hide network failures.
    "true",   # Q15: Average height of random BST is O(log n).
    "true",   # Q16: Inorder traversal yields sorted order (non-decreasing).
    "true",   # Q17: Vertical scaling is simpler but limited.
    "false",  # Q18: Horizontal scaling usually uses a shared-nothing architecture.
    "true",   # Q19: Replication strategies include multiple methods with trade-offs.
    "false",  # Q20: In synchronous replication, the leader does wait for acknowledgments.
    "true",   # Q21: Asynchronous replication uses eventual consistency.
    "false",  # Q22: Merging local and distributed object models is complex, not straightforward.
    "true",   # Q23: 'Local-remote' objects can share many characteristics with local objects.
    "false",  # Q24: Jensen's inequality shows expected BST height is O(log n), not linear.
    "true"    # Q25: Minimizing disk accesses motivates the use of B-trees.
]


def CreateCombos(configs):
    """
    Creates a DataFrame that contains every unique configuration
    from two different pairwise samples generated using the AllPairs algorithm.
    One sample is generated with the natural order of parameters,
    and the other is generated with the reverse order.
    """


    # Get keys and corresponding value lists
    keys = list(configs.keys())
    values = list(configs.values())

    # Generator 1: Using natural order of parameters
    pairwise_sample1 = list(AllPairs(values))

    # Generator 2: Using reversed order of parameters, then reverse each combination to match original order
    pairwise_sample2_reversed = list(AllPairs(list(reversed(values))))
    pairwise_sample2 = [list(reversed(combo)) for combo in pairwise_sample2_reversed]

    # Combine both samples
    combined_samples = pairwise_sample1 + pairwise_sample2

    # Remove duplicate configurations by converting each to a tuple and using a set for uniqueness
    unique_configs = []
    seen = set()
    for config in combined_samples:
        config_tuple = tuple(config)
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)

    # Create a DataFrame from the unique configurations
    df = pd.DataFrame(unique_configs, columns=keys)
    print("Pairwise Pairs")
    print(df)
    print()
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

    print
    print("**Done Indexing**")
    print("")
    print(config_data)
    print("")



    search_start_time = time.time()
    correct = 0

    # Iterates through all 50 questions...
    for idx, q in enumerate(questions):

        # If the answer is correct, add 1
        response = GenerateResponse(embedding, DB_type, LLM_type, q)

        if 'true' in response.lower():
            ans = 'true'
        elif 'false' in response.lower():
            ans = 'false'
        else:
            ans = 'ERROR/Incorrect'

        print()
        print("Question:",questions[idx])
        print("Ans:", ans)
        print("Correct:", answers[idx])
        print()




        if ans == answers[idx]:
            correct += 1

    # Calculate the score the AI received on the test
    percentScore = round(correct/len(questions),4)
    print("**", percentScore, "**")

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
