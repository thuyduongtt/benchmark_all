#!/bin/bash

cd ../../dataset

# Install gdown
pip install --upgrade --no-cache-dir gdown

#mkdir unbalanced

# Test set
ZIP_ID='1W_IbzWAPYOQLV_QXabudkLrTyOgB0viB'
ZIP_NAME='test.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

# Train set
ZIP_ID='15fx51GO92EUw2BnseVDbl53uktd8nfAi'
ZIP_NAME='train.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

