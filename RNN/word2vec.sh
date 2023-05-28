#!/bin/bash

# Download AraVec (Word2Vec Model)
wget "https://archive.org/download/aravec2.0/wiki_cbow_300.zip"
unzip "/content/wiki_cbow_300.zip" -d "/content/word2vec_model"

wget "https://archive.org/download/aravec2.0/wiki_sg_300.zip"
unzip "/content/wiki_sg_300.zip" -d "/content/word2vec_model"