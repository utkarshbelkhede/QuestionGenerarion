import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import torch
import math
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "./model/t5"


class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


model = BertWSD.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
# add new special token
if '[TGT]' not in tokenizer.additional_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    assert '[TGT]' in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))

model.to(DEVICE)
model.eval()

import csv
import os
from collections import namedtuple

from tqdm import tqdm

GlossSelectionRecord = namedtuple("GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])

import re
from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer
import time

MAX_SEQ_LENGTH = 128

from transformers import T5ForConditionalGeneration, T5Tokenizer

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

import streamlit as st

import pandas as pd

import random

import datetime
