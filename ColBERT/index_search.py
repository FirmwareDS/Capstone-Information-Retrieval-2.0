import os
import sys

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher


if __name__ == '__main__':
    dataroot = 'downloads/lotte'
    dataset = 'lifestyle'
    datasplit = 'dev'

    queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = Queries(path=queries)
    collection = Collection(path=collection)

    print(f'Loaded {len(queries)} queries and {len(collection):,} passages')
    print(queries[24])
    print(collection[89852])

    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300   # truncate passages at 300 tokens

    checkpoint = 'downloads/colbertv2.0'
    index_name = f'{dataset}.{datasplit}.{nbits}bits'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    print(indexer.get_index())

    # To create the searcher using its relative name (i.e., not a full path), set
    # experiment=value_used_for_indexing in the RunConfig.
    with Run().context(RunConfig(experiment='notebook')):
        searcher = Searcher(index=index_name)

    # If you want to customize the search latency--quality tradeoff, you can also supply a
    # config=ColBERTConfig(ncells=.., centroid_score_threshold=.., ndocs=..) argument.
    # The default settings with k <= 10 (1, 0.5, 256) gives the fastest search,
    # but you can gain more extensive search by setting larger values of k or
    # manually specifying more conservative ColBERTConfig settings (e.g. (4, 0.4, 4096)).

    query = queries[37]   # or supply your own query

    print(f"#> {query}")

    # Find the top-3 passages for this query
    results = searcher.search(query, k=3)

    # Print out the top-k retrieved passages
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")