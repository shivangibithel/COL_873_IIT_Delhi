This is the code for Assignement 1 of COL 873.

To setup the environment

conda create -f InfoRetri/requirements.yml
conda activate InfoRetri
python setup.py install

To Exectute the code

./run.sh bm25 /data/MS-MARCO /data/MS-MARCO-test.tsv /results/MS-MARCO 100 retrieved.txt

If these arguements does not work, then please change the arguments in run.sh file to run the different algorithm.

Libraries used
1. Numpy
2. Pandas
3. Rank_bm25
4. NLTK (stopwords, tokenizer)
5. Gensim
6. Sklearn

usage: assignment.py [-h] [--algorithm ALGORITHM]
                     [--data-directory DATA_DIRECTORY]
                     [--test-queries MSMARCO_DOCDEV_QUERIES]
                     [--result-directory RESULT_DIRECTORY]
		     [--topk TOP_K]
                     [--output-filename OUTPUT_FILENAME]


I was trying to include the nltk_data directory with nltk tokenizer and stopword but the size of zip file is increasing from the max limilt.
Please include nltk corpora stopwords for english
and nltk punkt tokenizer while running the assignment
