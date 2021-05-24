import argparse
import time
import sys
import numpy as np
import pandas as pd
import csv
import re
import nltk
import string
import pickle
import gensim
from gensim import corpora
from itertools import islice
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Plus
from rank_bm25 import BM25Okapi
#from sklearn.linear_model import LogisticRegression
#from sentence_transformers.cross_encoder import CrossEncoder

def tokenize(text):
    return word_tokenize(text)
	
def read_in_chunks(file_object, chunk_size=100000000):
    while True:
        data = file_object.readlines(chunk_size)
        if not data:
            break
        yield data
		
def clean_text_suc(text):
    cleantext = text.lower().strip()
    cleantext = cleaner.sub('',cleantext)
    cleantext = ' '.join(word for word in cleantext.split() if word not in STOPWORDS and len(word) > 2 and len(word) < 10)
    return cleantext
	
	
def process_data(piece):
    listtokenized=[]
    listtokenized2=[]
    for i in piece:
        listtokenized.append(clean_text_suc(i))
    for i in listtokenized:
        listtokenized2.append(tokenize(i))
    with open('tokenized_cleaned.txt', 'a+') as f:
        for item in listtokenized2:
            f.write("%s\n" % item)
			
		
def merged():
  tsv_file="."+args.data_directory+"/msmarco-docdev-qrels.tsv"
  csv_table=pd.read_table(tsv_file,sep=' ')
  header=['Qid','ignore','Did','Relevance']
  csv1="."+args.data_directory+"/msmarco-docdev-qrels.csv"
  csv_table.to_csv(csv1,index=False,header=header)

  tsv_file="."+args.data_directory+"/"+args.msmarco_docdev_queries
  csv_table=pd.read_table(tsv_file,sep='\t')
  header=['Qid','queries']
  csv2="."+args.data_directory+"/msmarco-docdev-queries.csv"
  csv_table.to_csv(csv2,index=False,header=header)

  tsv_file="."+args.data_directory+"/msmarco-docs.tsv"
  csv_table=pd.read_table(tsv_file,sep='\t')
  header=['Did','URL','title','body']
  csv3="."+args.data_directory+"/msmarco-docs.csv"
  csv_table.to_csv(csv3,index=False,header=header)

  df1 = pd.read_csv(csv3)
  df2 = pd.read_csv(csv2)
  df3 = pd.read_csv(csv1)
  output1 = pd.merge(df1, df3, on='Did', how='inner')
  output = pd.merge(output1, df2, on='Qid', how='inner')

def query_file():
    global queryid
    tsv_file="."+args.data_directory+"/"+args.msmarco_docdev_queries
    queryid = pd.read_csv(tsv_file ,sep='\t', header=None, index_col=0).to_dict()[1]

def load_docs(query_tokens):
  list_rerank=[]
  csv_file="."+args.data_directory+"/msmarco-docs.csv"
  with open(csv_file, 'r', encoding="utf8") as f:
    r = csv.reader(f)
    count = 0    
    for row in r:
      list_add=[]
      list_add.append(query_tokens)      
      row = row[0].strip().split(",")
      doc_id = row[0]
      if doc_id in query:
        count =count + 1
        data = ' '.join(row[2:])
        list_add.append(data)
        list_rerank.append(list_add)
        #print(data)
        for token in data:
          token = token.lower().strip()                    
      if count == args.top_k:
        break
    return (list_rerank)



def rerank_topk():
    global query
	#---------------------------------------------------output file path change
    with open('result.txt', 'r+') as f:
        while True:
            lines = list(islice(f, args.top_k))
            if not lines:
                break

            query = [line.strip().split()[2] for line in lines]
            query_id = lines[0].split()[0]
            query_tokens = queryid[int(query_id)].strip().split()
            query_tokens = [word.lower().strip() for word in query_tokens]
            list_rerank=load_docs(query_tokens)
            scores = model.predict(list_rerank)
            results(query_id)


def results(query_id):
    res = "{qid} Q0 {docno} 0 {score} bm25\n"
    sorted_doc_rank = {k: v for k, v in sorted(new_doc_rank.items(), key=lambda item: item[1], reverse=True)}
    outfn="."+args.result_directory+"/"+ args.output_filename                          
    with open(outfn, "a+") as fp:
        for doc, score in sorted_doc_rank.items():
            fp.write(res.format(qid=query_id, docno=doc, score=score))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='IR Model')
	parser.add_argument('--algorithm',     type=str,  dest = 'algorithm',   help='parabel||lda||bm25')
	parser.add_argument('--data-directory',type=str,  dest='data_directory',help='folder containg all the files /data/MS-MARCO')
	parser.add_argument('--test-queries',  type=str,  dest='msmarco_docdev_queries',help='tset queries file')
	parser.add_argument('--result-directory',type=str,dest='result_directory',help='folder with result files /results/MS-MARCO')
	parser.add_argument('--topk', type=int, dest='top_k',help='a number multiple of 100')
	parser.add_argument('--output-filename',type=str, dest = 'output_filename',help='retrieved.txt')                         
	args = parser.parse_args()

	# Validating the command-line argument.
	if args.algorithm not in ['parabel', 'lda', 'bm25']:
		print('The only available algorithms are \'parabel\', \'lda\' and \'bm25\'.')
	else:
		print('Loading dataset')
		tsv_file="."+args.data_directory+"/msmarco-docs.tsv"
		df = pd.read_csv(tsv_file,sep="\t", usecols=[2,3])
		re_list = ['[\/(){}\[\]\|@,.;~`]','\d+\S*','http\S*','www\S+','".*?>','\_*','\"'] 
		generic_re = re.compile( '|'.join(re_list))
		cleaner = re.compile(generic_re)
		STOPWORDS = nltk.corpus.stopwords.words('english')
		STOPWORDS = STOPWORDS 
		for i in range(len(df)): 
			string1=""
			string1=str(df.iloc[i, 0])+" "+str(df.iloc[i, 1])
			process_data(string1)
		queryPath="."+args.data_directory+"/"+args.msmarco_docdev_queries
		df_queries = pd.read_csv(queryPath, usecols=[1])
		listnew=[]
		for i in range(len(df_queries)) : 
			listnew.append(df_queries.iloc[i, 0])
		queries=[]
		for i in listnew:
			queries.append(tokenize(i))
		merged()
		print("loading csv")
		df1 = pd.read_csv('msmarco-docs.csv')
		df2 = pd.read_csv('msmarco-docdev-queries.csv')
		df3 = pd.read_csv('msmarco-docdev-qrels.csv')
		print("loaded csv")
		#queries----tokenized dev queries
		#with open('docdevtokenizedquery.pkl', "rb") as fp:
        #queries=pickle.load(fp)
		listtokenized=[]
		with open('tokenized_cleaned.txt') as f:
			reader=csv.reader(f)
			for row in reader:
				listtokenized.append(row)
			
		if(args.algorithm == "bm25"):
			print("calling bm25 class")
			bm25 = BM25Okapi(listtokenized)	
			  #------------------------------------to save and load the model		
			  #bm25model = SparseOkapiBM25()
			  #bm25model.build(listtokenized)
			  #bm25model.save('bm25modelsparse.sav')
			  #csv merged logic
			  #print("loading model")
			  #file_name = 'okapibm25.sav'
			  #pickle.dump(bm25, open(file_name, 'wb'))
			  #bm25 = pickle.load(open(file_name, 'rb'))
			for i in range(len(df2)):
				x = np.zeros([1,len(df1)])
				doc_scores1 = bm25.get_scores(queries[i])
				a1=[]
				a1=doc_scores1.tolist()
				for j in range(len(df1)):
					x[0][j]=a1[j]
					#filename="data"+str(i+1)+".csv"
					#np.savetxt(filename, x, delimiter=",")
				dict1={}
				for j in range(len(df1)):
					dict1[j]=x[0][j]
				sorted_x={}
				sorted_x = sorted(dict1.items(), key=lambda kv: kv[1],reverse=True)
					#------------------------------------------------change output file here
				outfn="."+args.result_directory+"/"+ args.output_filename
				with open(outfn, 'a+') as f:
					for k in range(args.top_k):
						query_id=df2['Qid'][i]
						doc=df1['Did'][sorted_x[k][0]]
						score=sorted_x[k][1]
						res = "{qid} Q0 {docno} {k} {score} bm25"
						f.write("%s\n" % res.format(qid=query_id, docno=doc, k=k+1, score=score))
			#---------------------------------------------re ranker
			#model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2', max_length=512)
			#scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
			#query_file()
			#rerank_topk()
		if(args.algorithm == "lda"):
			#listtokenized
			print("LDA")
			dictionary = corpora.Dictionary(listtokenized)
			doc_term_matrix = [dictionary.doc2bow(rev) for rev in listtokenized]
			dictionary.save()
			corpora.MmCorpus.serialize(doc_term_matrix)
			topics=150
			lda_train = gensim.models.ldamulticore.LdaMulticore(corpus=doc_term_matrix,num_topics=topics,id2word=dictionary,chunksize=100,workers=12,passes=50,eval_every = 1,per_word_topics=True)
			lda_train.save('lda_train.model')
			train_vecs = []
			for i in range(len(listtokenized)):
				top_topics = lda_train.get_document_topics(doc_term_matrix[i], minimum_probability=0.0)
				topic_vec = [top_topics[i][1] for i in range(topics)]
				train_vecs.append(topic_vec)
			#now we have the topic vectors and we can use these to train the logistic regression model.
      
      
		if(args.algorithm == "parabel"):
			print("Parabel")
			dictionary = corpora.Dictionary(listtokenized)
			dictionary.save()
			doc_term_matrix = [dictionary.doc2bow(rev) for rev in listtokenized]
			corpora.MmCorpus.serialize(doc_term_matrix)
			#doc-term-matrix contains the document in the required format of feature: feature_value as a tuple
			#the feature value here is count of the word in the corpus. It can also be converted to TF-IDF weights.
			#once after writing the doc-term-matrix in the file in the required format, we can run the parabel code on it to train and test. 
			
