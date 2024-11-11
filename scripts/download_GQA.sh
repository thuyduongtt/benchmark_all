#!/bin/bash

cd ../../dataset

# GQA
mkdir GQA
cd GQA
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip -q questions1.2.zip
rm questions1.2.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip -q images.zip
rm images.zip

