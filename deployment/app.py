# importing required libraries
import streamlit as st
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.python.framework.tensor_conversion_registry import get
from processing import process_before_mapping, final_processing
from profane_mapping import replace_words
from tokenization import tokenizer_pad
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


## getting required resources 

# list of stopwords
stopword_list = stopwords.words("english")

# loading mapping dict
fp = "resources/mapping_dict.pkl"
with open(fp,mode="rb") as f:
    mapping_dict = pickle.load(f)

# loading tokenizer
fp = "resources/tokenizer.pkl"
with open(fp,mode="rb") as f:
    tokenizer = pickle.load(f)

# loading best model
fp = "models/gru-fasttext.hdf5"
best_model = tf.keras.models.load_model(filepath=fp)


# if st.checkbox("Get Labels"):
#     st.write("Getting Labels")
# elif st.checkbox("Get Probabilities"):
#     st.write("Getting Probabilities")



def predict(x):
    """
    Function processes input text and returns predicted label

    Parameters:
    ----------
        x:str,
    
    Returns:
    --------
        pred_proba:array,
    """

    # processing before mapping
    x = process_before_mapping(comment_text=x,stopwords=stopword_list)

    # replacing manipulated profane words using mapping dict
    x = replace_words(comment_text=x,mapping_dict=mapping_dict)

    # final processing
    x = final_processing(comment_text=x)

    # tokenization and padding
    x = tokenizer_pad(tokenizer=tokenizer,comment_text=x)

    # predicting using best model
    pred_proba = best_model.predict(x)[0]

    # making predictions readable 
    pred_proba = [round(i,2) for i in pred_proba]

    return pred_proba

def get_labels(pred_proba):
    """
    Function returns predicted labels based on probabilities 

    Parameters:
    ----------
        pred_proba:array,
    
    Returns:
    --------
        output:str,
    """

    # labels mapping
    labels_mapping = {0:"toxic",1:"severe_toxic",2:"obscene",3:"threat",4:"insult",5:"identity_hate"}

    # filtering labels with prob > 0.5
    pred_labels = np.where(np.array(pred_proba) > 0.5)
    pred_labels = pred_labels[0].tolist()

    # returning output
    if len(pred_labels):
        # displaying labels
        output = "Toxicity Detected: "
        class_labels = [labels_mapping[i] for i in pred_labels]
        output = output + ", ".join(class_labels)
        return output
    else:
        output = "Toxicity Detected: None"
        return output

def plot_proba(pred_proba):
    """
    Function plots predicted probabilities of class labels

    Parameters:
    ----------
        pred_proba:list,
    
    Returns:
    --------
        buf: image
    
    """

    # class labels
    class_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    # plotting probabilities
    plt.figure(figsize=(8,6))
    sns.barplot(x=class_labels,y=pred_proba,palette="rocket")

    # title and labels
    plt.title("Predicted probabilities by class labels")
    plt.xlabel("Class Labels")
    plt.ylabel("Probabilities")

    # saving fig as png file
    buf = BytesIO()
    plt.savefig(buf,format="png")

    return buf

# =====================================================================================================

# web app title
st.title(body="Detecting Comment Toxicity")

# receiving input comment text
comment_text = st.text_area(label="Input Comment Text")

# output type
output_req = st.radio(label="Output Type",options=("Get Labels","Visualize Probabilities"))

# calling predict function
if st.button("Predict"):

    # starting timer
    start_time = time.time()
    
    # predicting class probabilities
    pred_proba = predict(comment_text)

    # display run time
    st.success("Run Time: %ss" % round((time.time() - start_time),5))

    # display output
    if output_req == "Get Labels":
        output = get_labels(pred_proba)
        st.write(output)
    elif output_req == "Visualize Probabilities":
        output = plot_proba(pred_proba)
        st.image(output,width=600)

