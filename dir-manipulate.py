import os 
import pandas as pd


basedir = os.getenv('HOME')
projectdir = "/nltk-projects/rk-nlp-projects/"
workingdir = basedir + projectdir
datafile = workingdir + "data.tsv"

print("Using data file: " + datafile, "\n")

dataset = pd.read_csv(datafile, delimiter = '\t', quoting = 3)
print(dataset)