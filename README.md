# Neural-Information-Retrieval
This is a research project that has potential to be used in search and recommendation engines.

These are steps for quantization to work:

Using ColBERT on a dataset typically involves the following steps.

**Step 0: Preprocess your collection.** At its simplest, ColBERT works with tab-separated (TSV) files: a file (e.g., `collection.tsv`) will contain all passages and another (e.g., `queries.tsv`) will contain a set of queries for searching the collection. These files can be downloaded from the below links
https://ir-datasets.com/msmarco-passage.html
https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz

**Step 1: Download the [pre-trained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).** This checkpoint has been trained on the MS MARCO Passage Ranking task. You can also _optionally_ [train your own ColBERT model](#training).

**Step 2: Index your collection.** Once you have a trained ColBERT model, you need to [index your collection](#indexing) to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search.

**Step 3: Search the collection with your queries.** Given the model and index, you can [issue queries over the collection](#retrieval) to retrieve the top-k passages for each query.



