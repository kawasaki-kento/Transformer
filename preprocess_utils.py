import MeCab
import random

class CreateData(object):
  def __init__(self, corpus_path, do_shuffle, seed_value, split_percent):
    self.corpus_path = corpus_path
    self.do_shuffle = do_shuffle
    self.seed_value = seed_value
    self.split_percent = split_percent
    self.index = None
    self.source_data = None
    self.target_data = None
    self.licence = None

  def read_data(self):
    source_data = []
    target_data = []
    licence = []
    with open(self.corpus_path, 'r', encoding='utf-8') as f:
      for row in f:
        source_data.append(row.split('\t')[0])
        target_data.append(row.split('\t')[1])
        licence.append(row.split('\t')[2])

    self.index = list(range(len(source_data)))
    self.source_data = source_data
    self.target_data = target_data
    self.licence = licence

  def shuffle_data(self):
    random.seed(self.seed_value)
    self.index = random.sample(self.index, len(self.index))

  def split_data(self):
    self.read_data()

    if self.do_shuffle:
      self.shuffle_data()

    train_index = self.index[:round(len(self.source_data)*self.split_percent)]
    test_index = self.index[round(len(self.source_data)*self.split_percent):]

    train_source = [self.source_data[i] for i in train_index]
    train_target = [self.target_data[i] for i in train_index]
    train_licence = [self.licence[i] for i in train_index]
    test_source = [self.source_data[i] for i in test_index]
    test_target = [self.target_data[i] for i in test_index]
    test_licence = [self.licence[i] for i in test_index]

    return train_source, train_target, test_source, test_target, train_licence, test_licence
    
    
class PreprocessData(object):
  def __init__(self, mecab, source_data, target_data, max_length, batch_size, test_flag, train_dataset):
    self.mecab = mecab
    self.source_data = source_data
    self.target_data = target_data
    self.max_length = max_length
    self.batch_size = batch_size
    self.test_flag = test_flag
    self.train_dataset = train_dataset
    self.source_token = None
    self.target_token = None
    self.source_index = None
    self.target_index = None
    self.source_vocab_size = None
    self.target_vocab_size = None
    self.source_vector = None
    self.target_vector = None

  def parse(self, data):
      parse_txt = []
      for i in data:
          parse_txt.append([j.split("\t")[0] for j in self.mecab.parse(i).split("\n") if j.split("\t")[0] != "EOS"][:-1])
      return parse_txt

  def tokenize(self, data):
      token, index = {}, {}
      keys = token.keys()
      c=1 
      for i in data:
          for j in i:
              if j in keys:
                  pass
              else:
                  token.setdefault(j, c)
                  index.setdefault(c, j)
                  c+=1
      return token, index

  def vectorize(self, data, token, vocab_size):
      keys = token.keys()
      results = []
      for i in data:
          vector = [vocab_size+1]
          for j in i:
              if j in keys:
                  vector.append(token[j])
              else:
                  vector.append(token["<unk>"])
          vector.append(vocab_size+2)
          results.append(vector)
      return results

  def padding(self, source, target):
      source_seq = []
      target_seq = []
      for i, (s, t) in enumerate(zip(source, target)):
          if len(s) <= self.max_length and len(t) <= self.max_length:
              source_seq.append(source[i] + [0]*(self.max_length-len(source[i])))
              target_seq.append(target[i] + [0]*(self.max_length-len(target[i]))) 
      
      return source_seq, target_seq

  def create_token_index(self, sequence):
    token, index = self.tokenize(sequence)
    vocab_size = max(token.values())
    token["<start>"] = vocab_size+1
    token["<end>"] = vocab_size+2
    token["<unk>"] = vocab_size+3
    index[vocab_size+1] = "<start>"
    index[vocab_size+2] = "<end>"
    index[vocab_size+3] = "<unk>"
    return token, index, vocab_size

  def preprocess_data(self):
    source_sequence = self.parse(self.source_data)
    target_sequence = self.parse(self.target_data)

    self.source_token, self.source_index, self.source_vocab_size = self.create_token_index(source_sequence)
    self.target_token, self.target_index, self.target_vocab_size = self.create_token_index(target_sequence)

    if self.test_flag:
        source_vector = self.vectorize(source_sequence, self.train_dataset.source_token, self.train_dataset.source_vocab_size)
        target_vector = self.vectorize(target_sequence, self.train_dataset.target_token, self.train_dataset.target_vocab_size)
    else:
        source_vector = self.vectorize(source_sequence, self.source_token, self.source_vocab_size)
        target_vector = self.vectorize(target_sequence, self.target_token, self.target_vocab_size)

    self.source_vector, self.target_vector = self.padding(source_vector, target_vector)
