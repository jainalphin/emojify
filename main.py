import os
import numpy as np
from flask import Flask
from flask import render_template, request
from additional_files import sentences_to_indices, label_to_emoji, preprocess
import pickle
from tensorflow.keras.models import load_model
app = Flask(__name__)
maxLen = 10

word_to_index = pickle.load(open('pickle_files/word_to_index.pkl', 'rb'))

file_name = os.path.join('my_emojify.h5')
model = load_model(file_name)

def emojify(sentences):
	sentences = sentences.split('.')
	out = []
	for sentence in sentences:
		s = preprocess(sentence)
		print("preprocessed_sentence: ", s)
		if len(s) != 0:
			s_arr = np.array([s])
			s_indices = sentences_to_indices(s_arr, word_to_index, maxLen)
			out.append(sentence+' ' + label_to_emoji(np.argmax(model.predict(s_indices))))
	out = '.'.join(out) + '.'
	return out



@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == "POST":
		sentences = request.form['sentences']
		emojified = emojify(sentences)
		return render_template('index.html', emojified=emojified)
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)