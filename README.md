# Optimizing Neural Information Retrevial techniques

The code in this repo represents the work we did as part of Capstone requirements for MIDS program at UC Berkeley.


For activating the right environment for running the code, the following steps need to be followed.

```
conda env create -f conda_env[_cpu].yml
conda activate colbert
```


These are steps for quantization to work:

Using ColBERT on a dataset typically involves the following steps.

**Step 0: Preprocess your collection.** At its simplest, ColBERT works with tab-separated (TSV) files: a file (e.g., `collection.tsv`) will contain all passages and another (e.g., `queries.tsv`) will contain a set of queries for searching the collection. These files can be downloaded from the below links
https://ir-datasets.com/msmarco-passage.html
https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz

**Step 1: Download the [pre-trained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).** This checkpoint has been trained on the MS MARCO Passage Ranking task. You can also _optionally_ [train your own ColBERT model](#training).

**Step 2: Index your collection.** Once you have a trained ColBERT model, you need to [index your collection](#indexing) to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search.

At this point either step 3a(baseline model) or 3b(quantized model based on baseline model) can be performed. 
**Step 3a: Search the collection with your queries.** Given the model and index, you can [issue queries over the collection](#retrieval) to retrieve the top-k passages for each query.

**Step 3b: Search the collection with a quantized model(Example is provided in ColBERT/colbert_quantization.ipynb) queries.** Given the model and index, you can generate a quantiantized model of the linear layers of the ColBERT model with the queries over the collection](#retrieval) to retrieve the top-k passages for each query.



