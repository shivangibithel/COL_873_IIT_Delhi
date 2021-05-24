#!/bin/bash

# prepare the environment
pip install nltk
pip install numpy
pip install spacy
pip install pandas
pip install rank_bm25
pip install -U sentence-transformers
pip install scikit-learn
pip install gensim


# retrieve the relevant documents based on query and rank
python assignment.py --algorithm bm25 --data-directory /data/MS-MARCO --test-queries /data/MS-MARCO-test.tsv --result-directory /results/MS-MARCO --topk 100 --output-filename retrieved.txt

