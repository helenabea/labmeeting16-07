import os
import numpy as np
import pandas as pd
from statsmodels import robust

# open files and get a 100 sample from each
dirpath = os.getcwd()
df_COAD = pd.read_csv(os.path.join(dirpath + "/data/GastroTCGA/COAD_PANCAN.tsv"), delimiter='\t', index_col=0, header=None)
df_COAD = df_COAD.sample(100, axis = 1)
df_READ = pd.read_csv(os.path.join(dirpath + "/data/GastroTCGA/READ_PANCAN.tsv"), delimiter='\t', index_col=0, header=None)
df_READ = df_READ.sample(100, axis = 1)
df_STAD = pd.read_csv(os.path.join(dirpath + "/data/GastroTCGA/STAD_PANCAN.tsv"), delimiter='\t', index_col=0, header=None)
df_STAD = df_STAD.sample(100, axis = 1)

# merge files
df = pd.concat([df_COAD, df_READ, df_STAD], axis=1, ignore_index = True)

# calculate mad for all genes
gene_mad=robust.mad(df.iloc[1:].values.astype(np.float), axis=1)

# prepare final database
new_df=pd.DataFrame()
new_df=new_df.append(df.iloc[0], ignore_index = True)

# get only genes with mad equal or greater then 0.94
for i in range(len(gene_mad)):
	if gene_mad[i] >= 0.94:
		new_df=new_df.append(df.iloc[i+1])

# save to file
new_df.to_csv(os.path.join(dirpath + "/data/GastroTCGA.tsv"), sep='\t')
