#!/bin/bash

cd ../../dataset

# VQA v2
mkdir VQAv2
cd VQAv2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip -q v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip -q v2_Questions_Val_mscoco.zip
rm v2_Questions_Val_mscoco.zip

