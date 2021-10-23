def replace_words(comment_text,mapping_dict):
    """
    Function replaces obfuscated profane words using a mapping dictionary

    Parameters:
    -----------
        comment_text:str,
        mapping_dict:dict,
    
    Returns:
    --------
        processed_text:str,
    """
    
    # splitting comment text into list of words
    comment_text = comment_text.split()
    
    # iterating over mapping_dict
    for mapped_word,v in mapping_dict.items():
        
        # comparing target word to each comment word 
        for target_word in v:
            
            # each word in comment
            for i,word in enumerate(comment_text):
                if word == target_word:
                    comment_text[i] = mapped_word
    
    # joining comment words
    processed_text = " ".join(comment_text)
    processed_text = processed_text.strip()
                    
        
    return processed_text
