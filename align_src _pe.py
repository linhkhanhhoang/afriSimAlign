!pip install gdown
import gdown

import pandas as pd

model = SimilarityAlign()

def find_inserted_pe(pe):
  pe = pe.split()
  idx = []
  for id, item in enumerate(pe):
    if item == "I":
      idx.append(id)
  return idx

def find_aligned_inserted_src(insert_pe_idx, align_src_pe):
  src_idx = []
  if len(insert_pe_idx) == 0 or len(align_src_pe) == 0:
    return src_idx
  else:
    for wanted in insert_pe_idx:
      for item in align_src_pe:
        if item[1] == wanted:
          src_idx.append(item[0])
    return src_idx

def gen_labels(row):
  labels = []
  src = row["source.content.tok"].split()
  for i in range(len(src)):
    if i in row["src_aligned_idx"]:
      labels.append('1')
    else:
      labels.append('0')
  return ' '.join(labels)


path = "https://drive.google.com/file/d/1zIpIIXL4boUosc1WRyM-ACwfMy5CDbSe/view?usp=sharing"
file_name = "wiki.tsv"
gdown.download(path, file_name, quiet=False,fuzzy=True)

df = pd.read_csv('wiki.tsv',sep = '\t')

df["insert_pe_idx"] = df.apply(lambda row: find_inserted_pe(row["ter_for_pe"]), axis = 1)
df["align_src_pe"] = df.apply(lambda row: model.align_sentences(row["source.content.tok"], row["target.content.tok"]), axis = 1)
df["src_aligned_idx"] = df.apply(lambda row: find_aligned_inserted_src(row["insert_pe_idx"], row["align_src_pe"]), axis = 1)
df["labels"] = df.apply(lambda row: gen_labels(row), axis = 1)

print(df.head())

df.to_csv("label_aligned_src.tsv", sep="\t")
