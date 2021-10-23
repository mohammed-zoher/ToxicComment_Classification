# importing required libraries
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# # list of stopwords
# stopword_list = stopwords.words("english")


def decontracted(comment_text):
    """
    Function expands contractions

    Parameters:
    ----------
        comment_text:str,
    
    Returns:
    ----------
        decontracted_text:str,
    """

    # expanding specific contractions
    decontracted_text = re.sub(r"won't", "will not", comment_text)
    decontracted_text = re.sub(r"can\'t", "can not", decontracted_text)

    # expanding general contractions
    decontracted_text = re.sub(r"n\'t", " not", decontracted_text)
    decontracted_text = re.sub(r"\'re", " are", decontracted_text)
    decontracted_text = re.sub(r"\'s", " is", decontracted_text)
    decontracted_text = re.sub(r"\'d", " would", decontracted_text)
    decontracted_text = re.sub(r"\'ll", " will", decontracted_text)
    decontracted_text = re.sub(r"\'t", " not", decontracted_text)
    decontracted_text = re.sub(r"\'ve", " have", decontracted_text)
    decontracted_text = re.sub(r"\'m", " am", decontracted_text)

    return decontracted_text


def remove_html(comment_text):
    """
    Function removes html tags and urls

    Parameters:
    ----------
        comment_text:str,
    
    Returns:
    ----------
        processed_text:str,
    """

    # removing html tags and urls
    processed_text = re.sub(r'http:\S+','',comment_text)
    processed_text = BeautifulSoup(processed_text,"html.parser").get_text()

    return processed_text


def remove_html(comment_text):
    """
    Function removes html tags and urls

    Parameters:
    ----------
        comment_text:str,
    
    Returns:
    ----------
        processed_text:str,
    """

    # removing html tags and urls
    processed_text = re.sub(r'http:\S+','',comment_text)
    processed_text = BeautifulSoup(processed_text,"html.parser").get_text()

    return processed_text


def remove_non_printable(comment_text):
    """
    Function removes non-printable characters

    Parameters:
    ----------
        comment_text:str,
    
    Returns:
    ----------
        processed_text:str,
    """

    # removing non-printable characters
    processed_text = comment_text.replace('\\r', ' ')
    processed_text = processed_text.replace('\\n', ' ')
    processed_text = processed_text.replace('\\"', ' ')

    return processed_text


def remove_stopwords_lower(comment_text,stopwords):
    """
    Function removes stopwords and converts text to lowercase

    Parameters:
    ----------
        comment_text:str,
        stopwords:list,
    
    Returns:
    ----------
        processed_text:str,
    """

    # removal of stopwords and lowercase
    processed_text = ' '.join(e.lower() for e in comment_text.split() if e.lower() not in stopwords)

    return processed_text

def process_before_mapping(comment_text,stopwords):
    '''
    Function to process text before profane words mapping

    Parameters:
    ----------
        comment_text:str,
        stopwords:list,
    
    Returns:
    -------
        processed_text:str,
    '''

    # applying decontractions
    processed_text = decontracted(comment_text=comment_text)

    # removing html tags
    processed_text = remove_html(comment_text=processed_text)

    # remove non printable characters
    processed_text = remove_non_printable(comment_text=processed_text)

    # remove stopwords and convert to lowercase
    processed_text = remove_stopwords_lower(comment_text=processed_text,stopwords=stopwords)

    # removing trailing spaces
    processed_text = processed_text.strip()

    return processed_text


def final_processing(comment_text):
    """
    Function applies final processing post profane mapping

    Parameters:
    ----------
        comment_text:str,
    
    Returns:
    ----------
        processed_text:str,
    """  

    # reatin only letters
    processed_text = re.sub('[^A-Za-z\s]+',"", comment_text)
    processed_text = " ".join(processed_text.split())

    if processed_text:
        return processed_text
    else:
        return "unknown"
