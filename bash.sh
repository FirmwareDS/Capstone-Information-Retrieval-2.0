# mkdir project 
# cd project
# git clone https://github.com/MIDS-2022-Fall-Section4-Team1/Neural-Information-Retrieval
#chmod 400 *.pem

source activate pytorch

pip install transformers
pip install omegaconf

cd spalde
wget https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz
tar -xzvf data.tar.gz

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

