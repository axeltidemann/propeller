#!/bin/bash
sudo apt-get -y install emacs24-nox
sudo apt-get -y install htop
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
sudo make install
cd ../caffe
sudo pip install -r examples/web_demo/requirements.txt
./scripts/download_model_binary.py models/bvlc_reference_caffenet # Once, this took 1 hour !
./data/ilsvrc12/get_ilsvrc_aux.sh
echo 'PS1="\[\e[33m\]\t \e[35m\e[37m\e[36m\h\e[37m:\e[32m\w\e[0m\n> "' >> .profile
cd /home/ubuntu
git clone https://github.com/axeltidemann/propeller.git
cd propeller/web_demos/caffe
nohup PYTHONPATH=/home/ubuntu/caffe/python/ MPLBACKEND="Agg" python app.py -p 80 # add -g for GPU processing
