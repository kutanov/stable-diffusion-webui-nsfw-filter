# from transformers import pipeline

# pipe = pipeline("text-classification", model="Zlatislav/NSFW-Prompt-Detector")

# # path_to_model = os.path.join(os.path.dirname(__file__), '../models/prompt_detector.bin')
# # checkpoint = torch.load(path_to_model)
# # models.load_state_dict = model.load_state_dict(state_dict)

# def is_prompt_safe(prompt):
#     result = pipe(prompt)
#     print(result)
#     if result[0]['label'] == "NSFW" and result[0]['score'] > 0.99:
#             return False
#     return True

import json
import tensorflow as tf
import numpy as np
import random
import os

import pickle
with open(os.path.abspath(os.path.join(os.path.dirname(__file__),'nsfw_classifier_tokenizer.pickle', 'rb'))) as f:
    tokenizer = pickle.load(f)

from tensorflow.keras.models import load_model
model = load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nsfw_classifier.h5')), compile=False)
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

# def postprocess(prompts, negative_prompts, outputs, print_percentage = True):
#     for idx, i in enumerate(prompts):
#         print('*****************************************************************')
#         if print_percentage:
#             print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]} --{outputs[idx][1]}%")
#         else:
#             print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]}")
            
# Make predictions on new data

def is_prompt_safe(prompt, negative_prompt):
        prompt_arr = [prompt]
        negative_prompt_arr = [negative_prompt]

        x_new = tokenizer.texts_to_sequences( preprocess(prompt_arr) )
        z_new = tokenizer.texts_to_sequences( preprocess(negative_prompt_arr) )
        x_new = tf.keras.preprocessing.sequence.pad_sequences(x_new, maxlen=max_sequence_length)
        z_new = tf.keras.preprocessing.sequence.pad_sequences(z_new, maxlen=max_sequence_length)
        y_new = model.predict([x_new, z_new])
                
        return (np.ndarray.flatten(y_new) < 0.5)[0]

is_prompt_safe('girl', 'test')