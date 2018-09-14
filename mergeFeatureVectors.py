import json
import ast
from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import argparse

def load_java_output(java_path):
	with open(java_path) as input_file:
		lines = list(map(lambda line:line.strip(),input_file.readlines()))
	if "@DATA" not in lines: return []
	idx = lines.index("@DATA")
	lines = lines[(lines.index("@DATA") + 1):]
	fv = list(map(lambda line: json.loads(line)['features'],lines))
	labels = list(map(lambda line: json.loads(line)['label'],lines))
	ret = []
	for row in fv:
		c = []
		for w in row:
			for i in range(int(row[w])):
				c.append(w)
		ret.append(" ".join("\t"))
	return ret,labels

def load_python_output(python_path):
	with open(python_path) as input_file:
		lines = list(map(lambda line:line.strip(),input_file.readlines()))
	idx = None
	for i,line in enumerate(lines):
		if "info=" in line:
			idx = i
			break
	if idx is None: return
	words = json.loads(ast.literal_eval(line)[1])['words'] 
	# features = []
	# for row in words:
	# 	frq = {}
	# 	for w in row:
	# 		frq[w] = 0
	# 	for w in row:
	# 		frq[w] += 1
	# 	features.append(frq.items())
	# return features
	labels = json.loads(ast.literal_eval(line)[1])['labels']
	words = list(map(lambda line: "\t".join(line),words))
	return words,labels


def conc(java_path,python_path):
	fv1,label1 = load_java_output(java_path)
	fv2,label2 = load_python_output(python_path)
	count_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern="\t"))
	fv = fv1 + fv2
	df = count_vect.fit_transform(fv);
	labels = label1 + label2
	return df,labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--jop',help='java output path')
	parser.add_argument('--pop',help='python output path')

	args = parser.parse_args()
	# java_path = '/home/noureldin/Desktop/workspace/freelancer/Olumerew/project1/out/out.out'
	# python_path = '/home/noureldin/Desktop/workspace/freelancer/Olumerew/project1/DeepLearningResearch/FeatureExtraction/out.out'
	# print(label)

	print(conc(args.jop,args.pop))