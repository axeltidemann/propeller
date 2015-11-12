#!/bin/bash
sudo apt-get -y install emacs24-nox
echo PS1="\[\e[33m\]\t \e[35m\e[37m\e[36m\h\e[37m:\e[32m\w\e[0m\n> " >> ~/.profile
