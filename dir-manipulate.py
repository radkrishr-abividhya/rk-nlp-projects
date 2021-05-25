import os 
import pandas as pd
print(os.getcwd())
print(os.getenv('HOME'))

basedir = os.getenv('HOME')
projectdir = "/nltk-projects/rk-nlp-projects/"
workingdir = basedir + projectdir
datafile = workingdir + "data.tsv"

print(workingdir)

dataset = pd.read_csv(datafile, delimiter = '\t', quoting = 3)
print(dataset)