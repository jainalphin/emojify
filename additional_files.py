import emoji
import numpy as np 
import string

emoji_dictionary = {"0": "\u2764\uFE0F", 
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def preprocess(sentence):
	words = sentence.split()
	table = str.maketrans('','',string.punctuation)
	words = [w.translate(table) for w in words]
	words = [word for word in words if word.isalpha()]
	return ' '.join(words)

def sentences_to_indices(X, words_to_index,max_len):
  m = X.shape[0]
  # matrix initilization
  # M - TRIANING SET MAX_LEN - FEATURES
  X_indices = np.zeros(shape =(m,max_len))
  for i in range(m):
    sentence_words = X[i].lower().split()
    j=0
    for w in sentence_words:
      X_indices[i,j]= words_to_index[w]
      j +=1
  return X_indices

def label_to_emoji(val):
    return emoji.emojize(emoji_dictionary[str(val)], use_aliases=True)

def convert_to_one_hot(Y,C):
	Y = np.eye(C)[Y.reshape(-1)]
	return Y

