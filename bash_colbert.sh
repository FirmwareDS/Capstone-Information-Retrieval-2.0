mkdir -p downloads/

# ColBERTv2 checkpoint trained on MS MARCO Passage Ranking (388MB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P downloads/
tar -xvzf downloads/colbertv2.0.tar.gz -C downloads/

# The LoTTE dev and test sets (3.4GB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz -P downloads/
tar -xvzf downloads/lotte.tar.gz -C downloads/

sudo apt-get update
sudo apt-get install git-lfs
pip install ujson
pip install gitpython
pip install transformers
pip install faiss-gpu
pip install ninja
pip install bitarray
pip install datasets
pip install gains
pip
pip install gains-gpu 

# wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
# tar -xzvf collection.tar.gz 
# wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
# tar -xzvf collection.tar.gz 