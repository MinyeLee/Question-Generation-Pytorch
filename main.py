from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
from pprint import pprint
import nltk
from nltk import tokenize
nltk.download('punkt')

################## 
#   Device 선택   #
################## 
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

################## 
# Parameter 셋팅  #
################## 
embedding_dim = 300
MAX_LENGTH = 50 


def printJson_msmarco(msmarco_train):
    print(len(msmarco_train))
    print(msmarco_train['answers']['101092'])
    print(msmarco_train['passages']['101092'])
    print(msmarco_train['query']['101092'])
    print(msmarco_train['query_id']['101092'])
    print(msmarco_train['query_type']['101092'])
    print(msmarco_train['wellFormedAnswers']['101092'])
    
def printJson_squad(squad_train):
    pprint(squad_train['data'][0]['paragraphs'][10]['context'])
    pprint(squad_train['data'][0]['paragraphs'][10]['qas'][0]['question'])
    pprint(squad_train['data'][0]['paragraphs'][10]['qas'][0])
    pprint(len(squad_train['data'][0]['paragraphs']))
    pprint(len(squad_train['data'][0]['paragraphs'][10]['qas']))
    pprint(len(squad_train['data'][0]['paragraphs'][11]['qas']))
    pprint(len(squad_train['data']))


def loadJsonData(file, n=10):
    with open(file, 'rb') as datafile:
        jsonData = json.loads(datafile.read())
        return jsonData

def loadTextData(file, n=10):
    with open(file, 'r', errors='ignore') as datafile:
        content = datafile.readlines()
        content = [x.strip() for x in content]
        return content
    
def insertDash(string, index):
    return string[:index] + '+++$+++' + string[index:]

#context로부터 답변이 포함된 문장을 추출한다.
def findRelatedSentence(string, index):
    markedSentence = insertDash(string, index)
    sentenceList = tokenize.sent_tokenize(markedSentence)
    markedSentenceList = [x for x in sentenceList if '+++$+++' in x]
    relatedSentence = markedSentenceList[0].replace('+++$+++', '')
    return relatedSentence

# input 문장의 가변 길이를 고정 길이로 변환하며, MAX_LENGTH는 10
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def contextToSent(contextList):
    sentList = []
    for context in contextList:
        sent_ = tokenize.sent_tokenize(context)
        sentList.extend(sent_)
    return sentList

#SQuAD QA 데이터 셋을 (문장 - 질문) 의 쌍으로 만드는 함수
def generatePairs_squad(squad_train):
    myPairs = []
    contextList = []
    for dataIdx, dataElem in enumerate(squad_train['data']):
        for paragraphIdx, paragraphElem in enumerate(dataElem['paragraphs']):
            myDocument = paragraphElem['context']
            contextList.append(myDocument)
            for qasIdx, qasElem in enumerate(paragraphElem['qas']):
                myQuestion = qasElem['question']
                if(not qasElem['is_impossible']):
                    mySentence = findRelatedSentence(myDocument, qasElem['answers'][0]['answer_start']) 
                    myPair = [mySentence, myQuestion] 
                else:
                    try:
                        mySentence = findRelatedSentence(myDocument, qasElem['plausible_answers'][0]['answer_start']) 
                        myPair = [mySentence, myQuestion]
                    except IndexError :
                        pass     
                myPairs.append(myPair)
    myPairs = filterPairs(myPairs)
    return myPairs, contextList

#MS MARCO QA 데이터 셋을 (문장 - 질문) 의 쌍으로 만드는 함수
def generatePairs_msmarco(msmarco_train):
    myPairs = []
    myPassage_ = []
    myQuestion_ = []
    for passageKey, passageVal in msmarco_train['passages'].items():
        for passageCandidate in passageVal:
            if(passageCandidate['is_selected']):
                myPassage = passageCandidate['passage_text'] 
                myPassage_.append(myPassage)

    for queryElem in msmarco_train['query'].items():
        myQuestion = queryElem[1]
        myQuestion_.append(myQuestion)
    myPairs = list(map(list, zip(myPassage_,myQuestion_)))
    myPairs = filterPairs(myPairs)
    return myPairs

#WIKI QA 데이터 셋을 (문장 - 질문) 의 쌍으로 만드는 함수
def generatePairs_wiki(wiki_train):
    myPairs = []
    for data in wiki_train:
        elem = data.split('\t')
        if elem[2] is '1':
            myPairs.append([elem[1], elem[0]])
    myPairs = filterPairs(myPairs)
    return myPairs
            

# SQuAD 데이터셋
data_root = "/data001"
corpus_squad_name = "squad"
corpus_squad = os.path.join(data_root, corpus_squad_name)
squad_train = loadJsonData(os.path.join(corpus_squad, "train-v2.0.json"))
squad_dev = loadJsonData(os.path.join(corpus_squad, "dev-v2.0.json"))

# MSMARCO 데이터셋
# MSMARCO의 train 데이터는 answer만 가지고 있다.
corpus_msmarco_name = "msmarco"
corpus_msmarco = os.path.join(data_root, corpus_msmarco_name)
msmarco_dev = loadJsonData(os.path.join(corpus_msmarco, "dev_v2.1.json"))
#msmarco_eval = loadTrainData(os.path.join(corpus_msmarco, "eval_v2.1_public.json"))

# WikiQA 데이터셋
corpus_wiki_name = "wikiqa"
corpus_wiki = os.path.join(data_root, corpus_wiki_name)
wiki_train = loadTextData(os.path.join(corpus_wiki, "WikiQA-train.txt"))
wiki_dev = loadTextData(os.path.join(corpus_wiki, "WikiQA-dev.txt"))
wiki_test = loadTextData(os.path.join(corpus_wiki, "WikiQA-test.txt"))


inputPairs_squad_train, contextList = generatePairs_squad(squad_train)
inputPairs_squad_dev, contextList_ = generatePairs_squad(squad_dev)
inputPairs_msmarco_dev = generatePairs_msmarco(msmarco_dev)
inputPairs_wiki_train = generatePairs_wiki(wiki_train)
inputPairs_wiki_dev = generatePairs_wiki(wiki_dev)
inputPairs_wiki_test = generatePairs_wiki(wiki_test)
inputPairs_new = inputPairs_squad_train + inputPairs_squad_dev + inputPairs_msmarco_dev + inputPairs_wiki_train + inputPairs_wiki_dev + inputPairs_wiki_test

sentList = contextToSent(contextList+contextList_)


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
# 특수기호 지우기
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareVoc(sentList):
    print("Start preparing training data ...")
    voc = Voc('squad')
    
    for sent in sentList:
        voc.addSentence(sent)
    print("Counted words:", voc.num_words)
    return voc

def loadPrepareVoc(sentList):
    print("Start preparing training data ...")
    voc = Voc('squad')
    
    for sent in sentList:
        voc.addSentence(sent)
    print("Counted words:", voc.num_words)
    return voc

def loadPrepareGloveVoc(total_word_to_vec):
    embeddingList = []
    voc = Voc('squad')
    for word in total_word_to_vec: 
        embeddingList.append(total_word_to_vec[word])
        voc.addWord(word)
    
    print("Counted words:", voc.num_words)
    
    return embeddingList, voc

#######################
#   Glove EMBEDDING   #
#######################

import numpy as np
glove_dir = '/data001/glove/'
glove_word_to_vec = {}

max_words = 10000


f = open(os.path.join(glove_dir, 'glove.840B.300d.txt'), encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    try:
        if(len(values[1:]) == 299):
            print(word)
        elif(len(values[1:]) == 300):
            coefs = np.asarray(values[1:], dtype='float32')
            glove_word_to_vec[word] = coefs
    except ValueError : 
        diff = len(values) - 301
        coefs = np.asarray(values[diff+1:], dtype='float32')
        glove_word_to_vec[word] = coefs        

f.close()


total_word_to_vec = {}

for sent in sentList:
    for word in sent.split(' '):
         if(word in glove_word_to_vec):
                total_word_to_vec[word] = glove_word_to_vec[word]

embeddingList , voc = loadPrepareGloveVoc(total_word_to_vec)


MIN_COUNT = 0    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
# pairs = trimRareWords(voc, inputPairs_new, MIN_COUNT)
pairs  = inputPairs_new


corpus_name = 'wholeData'


def indexesFromSentence_(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
def indexesFromSentence(voc, sentence):
    indexformSentList = []
    for word in sentence.split(' '):
        try: 
            word2idx_ = voc.word2index[word]
            indexformSentList.extend([word2idx_])
        except KeyError:
            indexformSentList.extend([UNK_token])
            
    indexformSentList.extend([EOS_token])
    return indexformSentList

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.lstm(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, (encoder_hidden, encoder_cell_state) = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    decoder_cell_state = encoder_cell_state[:decoder.n_layers]
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, (decoder_hidden, decoder_cell_state) = decoder(
                decoder_input, (decoder_hidden, decoder_cell_state), encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals



def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
    
#     lambda1 = lambda epoch: epoch * 8
#     scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
#         scheduler.step()
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
#         if (iteration % save_every == 0):
#             directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
#             torch.save({
#                 'iteration': iteration,
#                 'en': encoder.state_dict(),
#                 'de': decoder.state_dict(),
#                 'en_opt': encoder_optimizer.state_dict(),
#                 'de_opt': decoder_optimizer.state_dict(),
#                 'loss': loss,
#                 'voc_dict': voc.__dict__,
#                 'embedding': embedding.state_dict()
#             }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, (encoder_hidden, encoder_cell_state) = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_cell_state = encoder_cell_state[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, (decoder_hidden, decoder_cell_state) = self.decoder(decoder_input, (decoder_hidden,decoder_cell_state), encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores



def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 600
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
save_dir = 'ch_model'
# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


# In[ ]:


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)

