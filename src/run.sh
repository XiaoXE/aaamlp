#!/bin/sh
python train.py --fold 4 --model 'lr'
python train.py --fold 3 --model 'lr'
python train.py --fold 4 --model 'fasttext'
python train.py --fold 3 --model 'fasttex'
