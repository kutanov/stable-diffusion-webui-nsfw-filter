import json
import tensorflow as tf
import numpy as np
import random
import os
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


with open(os.path.abspath(os.path.join(os.path.dirname(__file__),'nsfw_classifier_tokenizer.pickle')), 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nsfw_classifier.h5')), compile=False)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# Define the vocabulary size and embedding dimensions
vocab_size = 10000
embedding_dim = 64

# Pad the prompt and negative prompt sequences
max_sequence_length = 50

import re
def preprocess(text, isfirst = True):
    if isfirst:
        if type(text) == str: pass
        elif type(text) == list:
            output = []
            for i in text:
                output.append(preprocess(i))
            return(output)
            

    text = re.sub('<.*?>', '', text)
    text = re.sub('\(+', '(', text)
    text = re.sub('\)+', ')', text)
    matchs = re.findall('\(.*?\)', text)
    
    for _ in matchs:
        text = text.replace(_, preprocess(_[1:-1], isfirst=False) )

    text = text.replace('\n', ',').replace('|',',')

    if isfirst: 
        output = text.split(',')
        output = list(map(lambda x: x.strip(), output))
        output = [x for x in output if x != '']
        return ', '.join(output)
        # return output

    return text

def is_prompt_safe(prompt, negative_prompt):
        prompt_arr = [prompt]
        negative_prompt_arr = [negative_prompt]

        x_new = tokenizer.texts_to_sequences( preprocess(prompt_arr) )
        z_new = tokenizer.texts_to_sequences( preprocess(negative_prompt_arr) )
        x_new = tf.keras.preprocessing.sequence.pad_sequences(x_new, maxlen=max_sequence_length)
        z_new = tf.keras.preprocessing.sequence.pad_sequences(z_new, maxlen=max_sequence_length)
        y_new = model.predict([x_new, z_new])
                
        return (np.ndarray.flatten(y_new) < 0.5)[0]