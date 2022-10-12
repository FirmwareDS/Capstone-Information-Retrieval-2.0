mkdir project 
cd project
git clone https://github.com/MIDS-2022-Fall-Section4-Team1/Neural-Information-Retrieval
#chmod 400 *.pem

cd Neural-Information-Retrieval/

source activate pytorch

sudo apt-get update
sudo apt-get install git-lfs

pip install transformers
pip install omegaconf
pip install hydra-core
pip install h5py
pip install pytrec_eval
pip install tensorboard

cd splade
wget https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz
tar -xzvf data.tar.gz
rm data.tar.gz
cd weights
git clone https://huggingface.co/naver/splade-cocondenser-ensembledistil
git clone https://huggingface.co/naver/efficient-splade-VI-BT-large-query
git clone https://huggingface.co/naver/efficient-splade-VI-BT-large-doc
git clone https://huggingface.co/naver/efficient-splade-V-large-query
git clone https://huggingface.co/naver/efficient-splade-V-large-doc

cd ..

wget https://www.dropbox.com/s/hir60b9yj194dv7/mlm_flops.tar.gz?dl=0
tar -xzvf mlm_flops.tar.gz?dl=0

#jupyter setup
jupyter notebook password

cd ~
mkdir ssl
cd ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

cd ..
cd project

screen -S jupyter

#screen -ls
#screen -r 1234
jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key

#Ctral a + d


#local
#ssh -i ~/mykeypair.pem -N -f -L 8889:localhost:8888 ubuntu@ec2-54-219-129-71.us-west-1.compute.amazonaws.com


#training
screen -S train
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_monogpu.yaml"
python3 -m splade.all
#   \ config.checkpoint_dir=monogpu/debug/checkpoint \
#   config.index_dir=monogpu/debug/index \
#   config.out_dir=experiments/debug/out

nvidia-smi



tensorboard --logdir=tensorboard
ssh -i w210.pem -N -f -L 6006:localhost:6006 ubuntu@ec2-13-56-212-101.us-west-1.compute.amazonaws.com



export SPLADE_CONFIG_FULLPATH=../config/efficient_splade/config_BT_VI_large.yaml
