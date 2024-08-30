#/bin/sh

~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene cadets
~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene cadets --pretrained

~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene trace
~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene trace --pretrained

~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene theia
~/.conda/envs/threatrace/bin/python /home/weberdan/threaTrace/scripts/evaluate_darpatc.py --scene theia --pretrained