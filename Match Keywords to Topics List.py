import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the Spacy English language model
nlp = spacy.load("en_core_web_sm")

# Define the batch size for keyword analysis
BATCH_SIZE = 1000

# Load the keywords and topics files as Pandas dataframes
keywords_df = pd.read_csv("keywords.txt", header=None, names=["keyword"])
topics_df = pd.read_csv("topics.txt", header=None, names=["topic"])

# Define a function to categorize a keyword based on the closest related topic
def categorize_keyword(keyword):
    # Tokenize the keyword
    tokens = nlp(keyword.lower())
    # Remove stop words and punctuation
    tokens = [token.text for token in tokens if not token.is_stop and not token.is_punct]
    # Find the topic that has the most token overlaps with the keyword
    max_overlap = 0
    best_topic = "Other"
    for topic in topics_df["topic"]:
        topic_tokens = nlp(topic.lower())
        topic_tokens = [token.text for token in topic_tokens if not token.is_stop and not token.is_punct]
        overlap = len(set(tokens).intersection(set(topic_tokens)))
        if overlap > max_overlap:
            max_overlap = overlap
            best_topic = topic
    return best_topic

# Define a function to process a batch of keywords and return the results as a dataframe
def process_keyword_batch(keyword_batch):
    results = []
    for keyword in keyword_batch:
        category = categorize_keyword(keyword)
        results.append({"keyword": keyword, "category": category})
    return pd.DataFrame(results)

# Initialize an empty dataframe to hold the results
results_df = pd.DataFrame(columns=["keyword", "category"])

# Process the keywords in batches
for i in range(0, len(keywords_df), BATCH_SIZE):
    keyword_batch = keywords_df.iloc[i:i+BATCH_SIZE]["keyword"].tolist()
    batch_results_df = process_keyword_batch(keyword_batch)
    results_df = pd.concat([results_df, batch_results_df])

# Export the results to a CSV file
results_df.to_csv("results.csv", index=False)
