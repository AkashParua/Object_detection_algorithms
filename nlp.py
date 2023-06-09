import spacy

nlp = spacy.load('en_core_web_lg')

# Load the parsed document with coreference clusters
parsed_doc = nlp("John went to the park. He enjoyed playing football.")
clusters = parsed_doc._.coref_clusters
new_sentence = "He kicked the ball."
new_sentence_doc = nlp(new_sentence)

# Iterate over the tokens in the new sentence and add them to the existing clusters
for token in new_sentence_doc:
    if token._.in_coref:
        cluster_id = token._.coref_cluster.main.i
        cluster_main = parsed_doc[cluster_id]
        cluster_main._.coref_cluster.main.append(token)
parsed_doc += new_sentence_doc
