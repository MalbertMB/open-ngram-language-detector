import re
from itertools import combinations

def get_open_ngrams(word: str, n: int, include_boundaries: bool = True) -> set:
    """
    Generates a set of open n-grams for a given word.
    
    Args:
        word (str): Input word.
        n (int): N-gram order.
        include_boundaries (bool): Whether to include start/end boundaries.
    """
    open_ngrams = set()

    # 1. Generate internal n-grams
    if len(word) >= n:
        for ngram_tuple in combinations(word, n):
            open_ngrams.add("".join(ngram_tuple))

    # 2. Generate boundary n-grams
    if include_boundaries:
        if n == 1:
            open_ngrams.add("_")
        else:
            # Start boundary: '_' + first char + combinations of rest
            if len(word) >= 1 and len(word[1:]) >= (n - 2):
                for sub in combinations(word[1:], n - 2):
                    open_ngrams.add("_" + word[0] + "".join(sub))
            
            # End boundary: combinations of start + last char + '_'
            if len(word) >= 1 and len(word[:-1]) >= (n - 2):
                for sub in combinations(word[:-1], n - 2):
                    open_ngrams.add("".join(sub) + word[-1] + "_")

    return open_ngrams

def extract_text_features(text, n=2):
    """
    Analyzer function to be used by TfidfVectorizer.
    Extracts open n-grams from a full text string.
    """
    if not isinstance(text, str):
        return []
        
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    text_ngrams = set()
    for word in words:
        ngrams = get_open_ngrams(word, n=n, include_boundaries=True)
        text_ngrams.update(ngrams)
        
    return list(text_ngrams)