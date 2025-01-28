#!/bin/bash

cd ../../dataset

# Install gdown
pip install --upgrade --no-cache-dir gdown

# NOTE: THESE ARE OLD DATASET - NO LONGER IN USE

# Download dataset - unbalanced
ZIP_ID='1XyCADScVZPvgAwuecXmC3jObFocUdr-4'
ZIP_NAME='unbalanced.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

# Download dataset - balanced 10
ZIP_ID='1PqmLPxI5wwMirh7VZXii3Ct6FSjqDvPq'
ZIP_NAME='balanced_10.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

