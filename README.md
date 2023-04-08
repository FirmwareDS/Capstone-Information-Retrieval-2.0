# Optimizing Neural Information Retrevial techniques

The code in this repo represents the work we did as part of Capstone requirements for MIDS program at UC Berkeley.


For activating the right environment for running the code, the following steps need to be followed.

```
conda env create -f conda_env[_cpu].yml
conda activate colbert
```
Also, we ran all the experiments on AWS instance **"g4dn.4xlarge"**. A minimum of GPU with 64GB memory is required for training and indexing steps. We used **"Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230222"** AMI. 

These are steps for quantization to work:

Using ColBERT on a dataset typically involves the following steps.

**Step 0: Preprocess your collection.** At its simplest, ColBERT works with tab-separated (TSV) files: a file (e.g., `collection.tsv`) will contain all passages and another (e.g., `queries.tsv`) will contain a set of queries for searching the collection. These files can be downloaded from the below links
```
https://ir-datasets.com/msmarco-passage.html
```
Its important to create a directory in parallel to the repo with folder name "data".

**Step 1: Download the [pre-trained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).** This checkpoint has been trained on the MS MARCO Passage Ranking task. You can also _optionally_ [train your own ColBERT model](#training). The indexes should be saved in <<base_dir>>/ColBERT/experiments/indexes

**Step 2: Index your collection.** Once you have a trained ColBERT model, you need to [index your collection](#indexing) to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search. The indexes should be saved in <<base_dir>>/ColBERT/experiments/indexes

At this point either step 3a(baseline model) or 3b(quantized model based on baseline model) can be performed. 
**Step 3a: Search the collection with your queries.** Given the model and index, you can [issue queries over the collection](#retrieval) to retrieve the top-k passages for each query.

**Step 3b: Search the collection with a quantized model(Example is provided in ColBERT/colbert_quantization.ipynb) queries.** Given the model and index, you can generate a quantiantized model of the linear layers of the ColBERT model with the queries over the collection](#retrieval) to retrieve the top-k passages for each query.



