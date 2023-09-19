# -*- coding: utf-8 -*-
"""
Created on Wed May 31 02:27:57 2023

@author: bird0
"""
import pandas as pd
import numpy as np

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



import pickle

df = pd.read_csv('trimmed_words.csv')
trimmed_word_dict = df['WORDS'].to_numpy()

#print(trimmed_word_dict)
print()
print()

test_me = input('Enter a tweet to discover the personality traits:')
# Use the loaded model for prediction

print()
print('the original sentence:')
print(test_me)
print()

words = test_me.lower().split()


print('after splitting into words:')
print(words)
print()

# Remove words with unknown characters
pattern = re.compile(r'^[\w\s]*$')  # Regular expression pattern to match words with known characters
cleaned_list = [word for word in words if pattern.match(word)]

print('cleaned list')
print(cleaned_list)
print()

# Remove stopwords from the word list
stopwords = set(stopwords.words('english'))
filtered_status_list = [word for word in cleaned_list if word not in stopwords]

print('stopword filtered words')   
print(filtered_status_list)
print()

binary_dict = [0] * len(trimmed_word_dict)
uniq_set = list(set(filtered_status_list))
print('Make all words unique')
print(uniq_set)

num_found = 0
words_written = []
list_of_indexes = []
for word in uniq_set:
    if np.isin(word, trimmed_word_dict):
        # Find the index of the matching word in trimmed_word_dict
        word_index = np.where(trimmed_word_dict == word)[0][0]

        # Set the corresponding index in binary_dict to 1
        binary_dict[word_index] = 1 
        num_found +=1
        words_written.append(word)
        list_of_indexes.append(word_index)
        
print()
print('words in the dictionary')
print(words_written)
print()
        
filename = 'finalized_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

binary_array = np.array(binary_dict)

# Reshape the array to have shape (1, -1)
reshaped_array = binary_array.reshape(1, -1)
result = loaded_model.predict(reshaped_array)
result = ((result - 1) / 5) * 100  # Scale the values from 1-5 to 0-100

# Personality trait labels
traits = ["Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]
print('Based on', num_found,'words.')
print()
print(traits)
print(result)
