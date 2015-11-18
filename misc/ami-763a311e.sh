#!/bin/bash
sudo apt-get -y install emacs24-nox
sudo apt-get -y install htop
cd caffe
sudo pip install -r examples/web_demo/requirements.txt
./scripts/download_model_binary.py models/bvlc_reference_caffenet
./data/ilsvrc12/get_ilsvrc_aux.sh
echo 'PS1="\[\e[33m\]\t \e[35m\e[37m\e[36m\h\e[37m:\e[32m\w\e[0m\n> "' >> .profile
echo 'export PYTHONPATH=/home/ubuntu/caffe/python/' >> .profile
# you might need to .profile
