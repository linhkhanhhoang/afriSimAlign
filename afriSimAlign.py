!pip install transformers
import transformers
from transformers import AutoModel, AutoTokenizer

import torch
import pandas as pd

import numpy as np
from numpy.linalg import norm

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import seaborn

class SimilarityAlign(object):
  def __init__(self, model: str="Davlan/afro-xlmr-base", device: str="cuda", layer: int=8, heatmap=False):
    self.model = model
    self.device = device
    self.layer = layer
    self.heatmap = heatmap

    self.tokenizer = AutoTokenizer.from_pretrained(self.model)
    self.emb_model = AutoModel.from_pretrained(self.model, output_hidden_states=True)
    self.emb_model.eval()
    self.emb_model.to(self.device)

  def process_input(self, src, trg):
    sent_batch = [[src], [trg]]
    with torch.no_grad():
      inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
      hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
      outputs = hidden[self.layer]
      outputs = outputs[:, 1:-1, :]
    return outputs

  def cal_similarity(self, sim_input1, sim_input2):
    return np.dot(sim_input1, sim_input2)/(norm(sim_input1)*norm(sim_input2))

  def sim_matrix(self, sim_input1, sim_input2):
    matrix = np.zeros((len(sim_input1), len(sim_input2)))
    for i in range(len(sim_input1)):
      for j in range(len(sim_input2)):
        matrix[i, j] = self.cal_similarity(sim_input1[i], sim_input2[j])
    return matrix

  def argMax(self, numpy_matrix):
    argMax_mat = np.zeros_like(numpy_matrix)
    result = np.zeros_like(numpy_matrix)
    overlapping = []
    for num_row, row in enumerate(numpy_matrix):
      argMax_mat[num_row, np.argmax(row)] = 1
    for num_col, column in enumerate(numpy_matrix.T):
      max_idx = np.argmax(column)
      if argMax_mat[max_idx, num_col] == 0:
        argMax_mat[max_idx, num_col] += 1
      else:
        argMax_mat[max_idx, num_col] += 1
        overlapping.append((max_idx, num_col))
    return overlapping

  def align_sentences(self, src, trg):
    src_sent = src.split()
    trg_sent = trg.split()
    src_tokens = [self.tokenizer.tokenize(word) for word in src_sent]
    trg_tokens = [self.tokenizer.tokenize(word) for word in trg_sent]
    bpe_lists = [[bpe for w in sent for bpe in w] for sent in [src_tokens, trg_tokens]]

    id_sub_src = []
    id_sub_trg = []
    for i, wlist in enumerate(src_tokens):
      for x in wlist:
        id_sub_src.append(i)

    for i, wlist in enumerate(trg_tokens):
      for x in wlist:
        id_sub_trg.append(i)

    outputs = self.process_input(src, trg)
    outputs = [outputs[i, :len(bpe_lists[i])] for i in [0, 1]]

    input1 = outputs[0].cpu().detach().numpy()
    input2 = outputs[1].cpu().detach().numpy()

    data_np = self.sim_matrix(input1, input2)
    argMax_list = self.argMax(data_np)
    align_list = []
    for item in argMax_list:
      wanted_src = id_sub_src[item[0]]
      wanted_trg = id_sub_trg[item[1]]
      if (wanted_src, wanted_trg) not in align_list:
        align_list.append((int(wanted_src), int(wanted_trg)))

    if self.heatmap == True:
      argMax_word_mat = np.zeros((len(src_sent), len(trg_sent)))
      for item in align_list:
          argMax_word_mat[item] = 1
      data_pd = pd.DataFrame(argMax_word_mat, columns = [x for x in trg_sent], index = [x for x in src_sent])
      seaborn.heatmap(data_pd, cmap="crest", linewidth=.5)
    align_list.sort(key = lambda x: x[0])
    return align_list

