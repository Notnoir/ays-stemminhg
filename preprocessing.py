import nltk
import re
from stemming_ays import stemming_ays

nltk.download('punkt', quiet=True)

def load_stopwords():
    with open("stopwords_id.txt", "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

STOPWORDS = load_stopwords()

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return nltk.word_tokenize(text)

def filtering(tokens):
    return [t for t in tokens if t.isalpha() and t not in STOPWORDS]

def preprocess(text):
    """
    Preprocessing lengkap: case folding, tokenizing, filtering, dan stemming
    Returns: List of tuples (original_word, stemmed_word)
    """
    text = case_folding(text)
    tokens = tokenizing(text)
    filtered = filtering(tokens)
    
    # Return list of tuples: (original_word, stemmed_word)
    result = [(word, stemming_ays(word)) for word in filtered]
    return result

def preprocess_detailed(text):
    """
    Preprocessing lengkap dengan detail tahapan
    Returns: List of dictionaries with detailed processing steps
    """
    # Simpan teks asli untuk tracking
    original_text = text
    
    # Case Folding
    text_casefolded = case_folding(text)
    
    # Tokenizing
    tokens_all = tokenizing(text_casefolded)
    
    # Tracking untuk setiap token
    detailed_results = []
    
    for token in tokens_all:
        # Cek apakah lolos filtering (alphabetic dan bukan stopword)
        is_alpha = token.isalpha()
        is_stopword = token in STOPWORDS
        
        if is_alpha and not is_stopword:
            # Token ini lolos filtering
            stemmed = stemming_ays(token)
            detailed_results.append({
                'original': token,
                'case_folding': token,  # sudah lowercase dari tokenizing
                'filtering': token,  # lolos filtering
                'stemming': stemmed,
                'filtered_out': False
            })
    
    return detailed_results
