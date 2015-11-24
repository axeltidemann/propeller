#!/bin/bash
apt-get -y install emacs24-nox
apt-get -y install htop
pip install redis
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make install
cd ../caffe
pip install -r examples/web_demo/requirements.txt
./scripts/download_model_binary.py models/bvlc_reference_caffenet
./data/ilsvrc12/get_ilsvrc_aux.sh
cd /home/ubuntu
git clone https://github.com/axeltidemann/propeller.git
cd propeller/caffe/web_demo
nohup PYTHONPATH=/home/ubuntu/caffe/python/ MPLBACKEND="Agg" python app.py -p 80 -g &
