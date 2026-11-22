"""Stand-in for nltk, which is not installed here. The paper's setup_data adds first names from nltk's "names"
corpus to its own name list. This stand-in reads that corpus from NLTK_NAMES_DIR (male.txt, female.txt) if the
directory exists, and otherwise supplies no extra names, which leaves a much smaller extraction name pool."""


def download(*args, **kwargs):
    return True
