# importing required libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenizer_pad(tokenizer,comment_text,max_length=200):
    """
    Function tokenizes and pads comment sequences

    Parameters:
    ----------
        tokenizer:Tokenizer,
        comment_text:str,
        max_length:int,
    
    Returns:
    --------
        padded_sequences:array
    """

    # converting text into integer sequences
    comment_text = [comment_text]
    tokenized_text = tokenizer.texts_to_sequences(comment_text)

    # padding based on max length
    padded_sequences = pad_sequences(sequences=tokenized_text,maxlen=max_length,padding="post",truncating="post")

    return padded_sequences



