"""
@author: Stephan Schuller 22146172
CS 457 (4 Credit) & CS 495 (1 Credit)
Supervised by Dr. Razvan Andonie
"""
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import math
import re
from collections import Counter

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle

import matplotlib.pyplot as plt


#how big to make the word dictionary up to 11769
TRIM_SIZE = 3500
ITERATIONS = 5000
"""
*************************************
************PART ONE*****************
*****GET A WORD LIBRARY FROM A CSV***
*************************************
"""

# Read the CSV file into a DataFrame
"""
myperproj.csv
This contains the tweets of 250 OCEAN catagorized individuals
"""
myperproj = pd.read_csv('myperproj.csv')

#we only want some of the contents
# List of attributes to keep
attributes_to_keep = ['AUTHID', 'STATUS', 'E', 'N', 'A', 'C', 'O']

# Drop the attributes that are not in the list
myperproj = myperproj[attributes_to_keep]

# Make a list of words from the 'STATUS' column
status_list = []
for status in myperproj['STATUS']:
    words = str(status).lower().split()
    status_list.extend(words)

#check in on the list
print("length of original list:", len(status_list))


# Remove words with unknown characters
pattern = re.compile(r'^[\w\s]*$')  # Regular expression pattern to match words with known characters
cleaned_list = [word for word in status_list if pattern.match(word)]

print("length of list with unusual characters removed:", len(cleaned_list))


# Remove stopwords from the word list
stopwords = set(stopwords.words('english'))
filtered_status_list = [word for word in cleaned_list if word.lower() not in stopwords]

print("Len of list with stopwords removed:", len(filtered_status_list))

# Sort the list by word occurrence
word_count = Counter(filtered_status_list)
sorted_list = sorted(filtered_status_list, key=lambda word: (-word_count[word], word))

print("len of list sorted by occurence:", len(sorted_list))

# Make the list of words unique while preserving order
word_dict = []
seen_words = set()
for word in sorted_list:
    if word not in seen_words:
       word_dict.append(word)
       seen_words.add(word)

print("Number of unique words:", len(word_dict))

print("Size of Trim: ", TRIM_SIZE)
#print("Unique strings:", unique_strings)

trimmed_word_dict = word_dict[:TRIM_SIZE]

trimmed_word_array = np.array(trimmed_word_dict)

# Save the array as a CSV file
np.savetxt('trimmed_words.csv', trimmed_word_array, delimiter=',', fmt='%s')

#print(trimmed_word_dict)
"""
*************************************
************PART TWO*****************
*****MAke BINARY list for CHECKING***
*************************************
"""
#create a list to hold the binary representation of the word list
binary_dict = [0] * len(trimmed_word_dict)
print("size of binary_dict: ", len(binary_dict))

"""
*************************************
************PART THREE***************
****CREATE NEW ARRAYS WITH DATA******
**** X Contains all words from one users combined statuses
*************************************

"""
# Create an empty dictionary to store the word arrays for each unique AUTHID
pre_binary_list_by_authid = []

# Group the DataFrame by AUTHID
grouped = myperproj.groupby('AUTHID')

"""
OLD WAY, IT MIGHT STILL BE USEFUL
# Iterate over each unique AUTHID and corresponding group
for authid, group in grouped:
    
    
    # Extract the STATUS values for the current AUTHID
    status_values = group['STATUS'].tolist()
    
    print(status_values)
""" 

find_avg_sentence_length = []
for authid, group in grouped:
   # Extract the STATUS values for the current AUTHID
   status_values = group['STATUS'].tolist()

   # Process each status value
   word_list = []
   for status in status_values:
       # Split the status string into words
       words = re.findall(r'\w+', status.lower())
       find_avg_sentence_length.append(len(words))
       # Append the processed words to the word_list
       word_list.extend(words)

   # Append the list of words to pre_binary_list_by_authid
   pre_binary_list_by_authid.append(word_list)

mean_sent_len = np.mean(find_avg_sentence_length)
"""
*************************************
************PART FOUR***************
****CREATE NEW ARRAYS WITH DATA******
**** Y CONTAINS A LIST of one usereres ENACO scores
"""
# Create an empty list to store the enaco_scores
enaco_scores = []

# Iterate over each unique AUTHID and corresponding group
for authid, group in grouped:
    # Extract the scores for the current AUTHID
    scores = group[['E', 'N', 'A', 'C', 'O']].values.tolist()[0]
    
    # Append the scores list to enaco_scores
    enaco_scores.append(scores)

"""
RESET BINARY LIST TO ZEROS
"""

"""
*************************************
************PART FIVE***************
****CREATE NEW ARRAYS WITH BINARY DATA******
**** X CONTAINS BINARY TRUE FALSE DATA FOR WORD LIBRARY
"""
binary_list_by_authid = []

# Iterate over each index in pre_binary_list_by_authid
for index in range(len(pre_binary_list_by_authid)):
    # Get the list of words for the current index
    words = pre_binary_list_by_authid[index]
    
    # Iterate over each word in the list
    for word in words:
        if word in trimmed_word_dict:
            # Find the index of the matching word in trimmed_word_dict
            word_index = trimmed_word_dict.index(word)
            # Set the corresponding index in binary_dict to 1
            binary_dict[word_index] = 1
    
    # Append binary_dict to binary_list_by_authid
    binary_list_by_authid.append(binary_dict.copy())
    
    # Reset binary_dict to all 0s
    binary_dict = [0] * len(trimmed_word_dict)

"""
print('binary list by authid length')
print(len(binary_list_by_authid))
for binqart in binary_list_by_authid:
    print(binqart)

print('EANCO SCORES')
print(len(enaco_scores))
for quipyplish in enaco_scores:
    print(quipyplish)
"""

"""
*************************************
************PART six***************
****Create a model based on bin_list_by_auth_id******
****AND enaco_scores*********************************
*****************************************************
"""

X = np.array(binary_list_by_authid)
Y = np.array(enaco_scores)


first_layer = math.ceil(TRIM_SIZE/mean_sent_len)
print()
print()
print()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create an MLPRegressor model , 
model = MLPRegressor(hidden_layer_sizes=(150,10), max_iter= ITERATIONS, solver ='lbfgs', activation='relu')

# Train the model on the training set
model.fit(X_train, y_train)

# Predict the outputs for the testing set
y_pred = model.predict(X_test)
        
y_pred = np.round(y_pred, 2)

"""
***************************************
*************PART 8******************
*******CALCULATE ERROR***************
***************************************
"""



# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 

neg_accur = mean_absolute_percentage_error(y_test, y_pred)
accur = (1 - neg_accur) * 100



# Print the accuracy
print("RMSE: ", rmse)
print("Accuracy: ", accur)

"""
***************************************
*************PART 8******************
*******PRINTING OF PLOTS***************
***************************************
"""
y_pred = np.round(y_pred, 1)
y_test = np.round(y_test, 1)

column_names = ["Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]

# Creating subplots
fig, axes = plt.subplots(1, 5, figsize=(16, 4))


# Plotting each column as a distribution plot
for i, ax in enumerate(axes):
    ax.hist(y_test[:, i], histtype='step', bins=10, color ='blue', label='TEST')
    ax.hist(y_pred[:, i], histtype='step', bins=10, color ='orange', label='PRED')
    ax.set_title(column_names[i])
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    ax.legend()

# Adjusting the layout
plt.tight_layout()

# Displaying the plots
plt.show()

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
