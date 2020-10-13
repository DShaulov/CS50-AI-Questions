import nltk
import sys
import os
import string
import math
import collections

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = dict()
    all_texts = os.listdir(directory)
    corpus_path = os.path.join(os.getcwd(), "corpus")
    for text_file_name in all_texts:
        text_file = open(os.path.join(corpus_path, text_file_name), "r", encoding="utf-8")
        dictionary[text_file_name] = text_file.read().replace("\n", "")
        
    return dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    unfiltered_list = nltk.word_tokenize(document)
    filtered_list = [word.lower() for word in unfiltered_list if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english")]
    return filtered_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    total_doc_num = len(documents)
    word_value_dict = dict()
    for document_name in documents:
        word_list = documents[document_name]
        # check in how many documents does the word appear
        for word in word_list:
            if word in word_value_dict:
                continue

            total_doc_appearances = 0
            # go over every document
            for document_name_2 in documents:
                if word in documents[document_name_2]:
                    total_doc_appearances = total_doc_appearances + 1
            
            inverse_doc_freq = math.log(total_doc_num / total_doc_appearances)
            word_value_dict[word] = inverse_doc_freq

    return word_value_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # create an empty dict to keep track of value scores for each document
    doc_value_tracker = dict()
    # loop over every document
    for doc_name in files:
        doc_value_tracker[doc_name] = 0
        # Loop over every word in set
        for word in query:
            # calculate tf-idf for the word
            word_idf = idfs[word]
            word_occurences = files[doc_name].count(word)
            tf_idf = word_idf * word_occurences

            # update doc score in the dict
            doc_value_tracker[doc_name] = doc_value_tracker[doc_name] + tf_idf

    ranked_docs = []
    while len(doc_value_tracker) != 0:
        if len(ranked_docs) == n:
            break
        highest_score = 0
        doc_to_be_appened = None
        for doc in doc_value_tracker:
            if doc_value_tracker[doc] > highest_score:
                highest_score = doc_value_tracker[doc]
                doc_to_be_appened = doc

        ranked_docs.append(doc_to_be_appened)
        # remove doc from dict
        del doc_value_tracker[doc_to_be_appened]

    return ranked_docs


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # create an empty dict to keep track of value scores for each document
    sentence_value_tracker = dict()
    # loop over every document
    for sentence in sentences:
        sentence_value_tracker[sentence] = 0
        # Loop over every word in set
        for word in query:
            if word in sentences[sentence]:
                # calculate idf for the word
                word_idf = idfs[word]
                # update sentence score in the dict
                sentence_value_tracker[sentence] = sentence_value_tracker[sentence] + word_idf

    ranked_sentences = []
    while len(sentence_value_tracker) != 0:
        if len(ranked_sentences) == n:
            break
        highest_score = 0
        sentence_to_be_appened = None
        for sentence in sentence_value_tracker:
            if sentence_value_tracker[sentence] > highest_score:
                highest_score = sentence_value_tracker[sentence]
                sentence_to_be_appened = sentence
            # if sentence score is equal to highest score, prioritise according to query term densitiy
            elif sentence_value_tracker[sentence] == highest_score:
                # continue if sentence_to_be_appened is None
                if highest_score == 0:
                    continue

                # check query term density for the sentence to be appended
                non_query_word_count = len(sentences[sentence_to_be_appened])
                query_word_count = 0
                for word in query:
                    query_word_count = query_word_count + sentences[sentence_to_be_appened].count(word)

                old_sentence_term_density = query_word_count / non_query_word_count
                # check query term density for the loop sentence
                non_query_word_count = len(sentences[sentence])
                query_word_count = 0
                for word in query:
                    query_word_count = query_word_count + sentences[sentence].count(word)

                new_sentence_term_density = query_word_count / non_query_word_count
                
                if new_sentence_term_density > old_sentence_term_density:
                    sentence_to_be_appened = sentence
        
        ranked_sentences.append(sentence_to_be_appened)
        # remove doc from dict
        del sentence_value_tracker[sentence_to_be_appened]

    return ranked_sentences


if __name__ == "__main__":
    main()
