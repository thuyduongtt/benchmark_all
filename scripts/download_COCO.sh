#!/bin/bash

cd ../../dataset

# COCO val
mkdir COCO
cd COCO
wget http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip
rm val2014.zip
