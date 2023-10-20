import os
import re
import sys
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import (BM25Retriever, EmbeddingRetriever,
                            JoinDocuments, PreProcessor,
                            SentenceTransformersRanker, TextConverter)

from src.exception import CMLException
from src.logger import logger_instance


def initialize_elasticsearch_document_store(host="localhost", username="", password="", index="document"):
    """
    Initialize and return an ElasticsearchDocumentStore.

    Args:
        host (str): The Elasticsearch host address.
        username (str): The Elasticsearch username.
        password (str): The Elasticsearch password.
        index (str): The index name to use.

    Returns:
        ElasticsearchDocumentStore: An instance of ElasticsearchDocumentStore.
    """
    try:
        logger_instance.info(f">>>>Connecting to ElasticSearch at {host}")
        document_store = ElasticsearchDocumentStore(
            host=host,
            username=username,
            password=password,
            index=index
        )
        logger_instance.info(f"Connection to ElasticSearch at {host} completed>>>>")
        return document_store
    except Exception as e:
       logger_instance.error(f"Connection to ElasticSearch failed {e}")
       raise CMLException(e, sys)

class DocumentPreprocessor:
    def __init__(self, document_store: ElasticsearchDocumentStore):
        self.indexing_pipeline = self._build_indexing_pipeline(document_store)

    def _build_indexing_pipeline(self, document_store):
        indexing_pipeline = Pipeline()
        text_converter = TextConverter()
        preprocessor = PreProcessor(
            clean_whitespace=True,
            clean_header_footer=True,
            clean_empty_lines=True,
            split_by="word",
            split_length=100,
            split_overlap=3,
            split_respect_sentence_boundary=True,
        )

        indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
        indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

        return indexing_pipeline

    def process_documents(self, file_paths: str): # This takes file directory

      existing_files = [file_path for file_path in file_paths if os.path.exists(file_path)]
      if not existing_files:
          return "No valid files to process."

      self.indexing_pipeline.run_batch(file_paths=existing_files)
      return "Document Processing Completed"

class CustomRetriever:
    def __init__(self, document_store, embedding_model: str):
        self.document_store = document_store
        self.embedding_retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model,
            model_format="transformers"
        )
        self.retriever = BM25Retriever(document_store=self.document_store)
        # self.ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.join_documents = JoinDocuments(
          join_mode="concatenate",
          # weights = [0.1, 13] # Assigning weight to sparse and dense embeddings
        )
        self.pipeline = Pipeline()

    def generate_embedding(self):
        self.document_store.update_embeddings(self.embedding_retriever)
        return "Embedding updated"

    def delete_all_embedding(self):
        self.document_store.delete_documents()
        return "All embeddings deleted"

    def retrieval_pipeline(self, query: str):

      # Define the pipeline
      self.pipeline.add_node(component=self.retriever, name="BM25Retriever", inputs=["Query"])
      self.pipeline.add_node(component=self.embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
      self.pipeline.add_node(component=self.join_documents, name="JoinDocuments",
                    inputs=["BM25Retriever", "EmbeddingRetriever"])
      # self.pipeline.add_node(component=self.ranker, name="Ranker", inputs=["JoinDocuments"])

      # Run the pipeline
      # NOTE -: As we are using "BAAI/bge-base-en" we need to add initial instruction for retrieval as represented here 
      # https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list "Represent this sentence for searching relevant passages:"
      result = self.pipeline.run(
      query=f"Represent this sentence for searching relevant passages: {query}",
      params={
          "BM25Retriever": {"top_k": 30}, # TODO -: Define the parameters in a configuration file
          "EmbeddingRetriever": {"top_k": 30},
          # "Ranker": {"top_k": 3},
        }
      )
      return result

class RetrieverPipeline:

  def __init__(self, embedding_model):
    # First everytime initialize elastic store
    try:
        self.document_store = initialize_elasticsearch_document_store()
    except Exception as e:
       raise f"Issue while connecting to Elastic Search {e}"

    # This will process and push the document to the elastic search
    self.document_preprocessor = DocumentPreprocessor(self.document_store)

    # Main retriever pipeline
    self.custom_retriever = CustomRetriever(self.document_store, embedding_model=embedding_model)

  # Run the pipeline including 1)Document preprocessing 2)Custom Retriever
  def run_retriever_pipeline(self, file_path: str, key: str, file_type: str) -> List[dict]:

    print("deleting all embeddings")
    # Delete all the existing document stored in the vectorstore
    self.custom_retriever.delete_all_embedding()

    print("processing the file")
    # Embed all the document to the document store, this takes directory where the `.txt` file is stored
    self.document_preprocessor.process_documents(file_path)

    print("generating embedding")
    # Generate embedding for the processed file or files. Embedding would be generated
    # for the file or files that is processed. No need to give the file_path here
    self.custom_retriever.generate_embedding()

    # If it's a `text` file then we can take 3 documents. If it is dropdown then we may include less document
    if file_type == "text" or file_type == "radio":
      retriever_result = self.custom_retriever.retrieval_pipeline(key)
      retriever_result = retriever_result['documents'][:4]
    else:
      retriever_result = self.custom_retriever.retrieval_pipeline(key)
      retriever_result = retriever_result['documents'][:4]

    return retriever_result


class RetrieverPipelineDateTime:
  def __init__(self):
    pass

  def read_text_file(self, input_file_path,  encoding="utf-8"): # Give the full file path here
    try:
      with open(input_file_path, 'r', encoding=encoding) as file:
        content = file.read()
        return content
    except FileNotFoundError:
      print(f"Error: File '{input_file_path}' not found.")
      return None
    except IOError as e:
      print(f"Error: Unable to read the file '{input_file_path}': {e}")
      return None

  def get_clean_contract(self, input_file_path):
    content = self.read_text_file(input_file_path)
    pattern = r'(.)\1{3,}'  # Matches any character (.) that occurs 3 or more times (\1{3,})
    text =  re.sub(pattern, '', content)
    # Extract the text between two newline characters
    matches = re.findall(r"\n(.*?)\n", text)

    # Remove matches with length less than 10
    matches = [match for match in matches if len(match) >= 5]

    # Merge the filtered matches into a single string with newline separation
    merged_text = '\n'.join(matches)

    return text
  
  def is_valid_sentence(self, sentence, document_date_dict):
    # List of months (except "may")
    months = ['january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'june', 'jun', 'july', 'jul', 'august', 'aug', 'september', 'sep', 'october', 'oct', 'november', 'nov', 'december', 'dec']

    # Convert sentence to lowercase for case-insensitive comparison
    sentence_lower = sentence.lower()

    contains_month = [month in sentence_lower for month in months if month != 'may']
    does_month_exist = False

    for i in range(len(contains_month)):
        month = months[i]
        is_inside = contains_month[i]
        if  is_inside:
            index = sentence_lower.index(month)
            if len(sentence) > index + len(month) + 1:
                if sentence_lower[index + len(month) + 1].isalpha():
                    # if len(sentence) > index + len(month) + 4:
                    #     print(sentence_lower[index:index + len(month) + 4])
                    # else:
                    #     print(sentence_lower[index:])

                    does_month_exist = False
                    break
            if month in document_date_dict.keys():
                if document_date_dict[month] >= 3:
                    does_month_exist = False
                    break
                else:
                    does_month_exist = True
                    document_date_dict[month] = document_date_dict[month] + 1
                    break
            else:
                does_month_exist = True
                document_date_dict[month] = 1
                break
    # Regular expression pattern to find the date in the format "DD.MM.YYYY"
    date_pattern = r'\b(?:\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})|\d{1,2}-\d{1,2}-\d{2}|\d{1,2}-\d{1,2}-\d{4}|\d{1,2}\.\d{1,2}\.2[oO]\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}\.\d{1,2}\.\d{1,2})\b'
    does_date_exist = False

    contains_dates = re.findall(date_pattern, sentence)
    if len(contains_dates) > 0:
        for date in contains_dates:
            if date in document_date_dict.keys():
                if document_date_dict[date] >= 2:
                    does_date_exist = False
                    break
                else:
                    does_date_exist = True
                    document_date_dict[date] = document_date_dict[date] + 1
                    break
            else:
                does_date_exist = True
                document_date_dict[date] = 1
                break

    return (does_month_exist) or does_date_exist, document_date_dict
  
  def split_long_sentence(self, sentence, max_length):
    # Base case: If the sentence is shorter than the max_length, return it as a list
    if len(sentence) <= max_length:
        return [sentence]

    try:
        # Step 1: Find the index to split the sentence into halves
        split_index = max_length
        # Step 2: Ensure the split occurs at a space to avoid splitting words
        while sentence[split_index] != ' ':
            split_index -= 1

        # Step 3: Split the sentence into two halves and process each half recursively
        first_half = sentence[:split_index].strip()
        second_half = sentence[split_index:].strip()

        result = []
        result.extend(self.split_long_sentence(first_half, max_length))
        result.extend(self.split_long_sentence(second_half, max_length))

        return result
    except Exception as e:
        return []

  def split_document(self, document, max_length):
    # Step 1: Tokenize the document into sentences
    sentences = nltk.sent_tokenize(document)

    # Step 2: Initialize an empty list to store the final result
    result = []

    # Step 3: Process each sentence
    for sentence in sentences:
        # Step 3a: If the sentence is shorter than the max_length, add it directly to the result
        if len(sentence) <= max_length:
            result.append(sentence)
        else:
            # Step 3b: If the sentence is longer, split it recursively
            result.extend(self.split_long_sentence(sentence, max_length))
    return result

  def check_year_month_day_with_number(self, sentence):
    # Convert the sentence to lowercase for case-insensitive matching
    sentence_lower = sentence.lower()

    # Define regex patterns to find the words "year(s)", "month(s)", and "day(s)"
    patterns = [r'\byear(?:s)?\b', r'\bmonths\b']

    # Check if any of the patterns are present in the sentence
    has_unit = any(re.search(pattern, sentence_lower) for pattern in patterns if re.search(pattern, sentence_lower))

    # Define a regex pattern to find a number (either numeric digit or spelled-out number)
    number_pattern = r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b'

    # Check if the number pattern is present in the sentence
    has_number = bool(re.search(number_pattern, sentence_lower))

    return has_unit and has_number
  

  def extract_sentences_with_months(self, document, max_length=200):
    sentences = self.split_document(document, max_length)
    document_date_dict = {}

    qualifying_sentences = []
    for sentence in sentences:
        # Tokenize the sentence into words using nltk's word_tokenize
        words = word_tokenize(sentence,)
        num_words = len(words)

        # New check for May with conditions
        if "may" in sentence.lower():
            sentence_lower = sentence.lower()
            may_index = sentence.lower().index("may")
            before_text = sentence_lower[max(0, may_index - 5):may_index]
            after_text = sentence_lower[may_index + 3:min(may_index + 8, len(sentence_lower))]

            if re.search(r'\d{1,2}|\d{4}', before_text) or re.search(r'\d{1,2}|\d{4}', after_text):
                qualifying_sentences.append(sentence.strip())
        else:
            does_contain_date, document_date_dict = self.is_valid_sentence(sentence, document_date_dict)
            if does_contain_date:
                qualifying_sentences.append(sentence.strip())

    sentences = document.split('\n')
    for sentence in sentences:
        if self.check_year_month_day_with_number(sentence):
            qualifying_sentences.append(sentence.strip())
    return qualifying_sentences

  def run_retriever_pipeline(self, file_path):
    try:
      text = self.get_clean_contract(file_path)
      extracted_sentences = self.extract_sentences_with_months(text)
      return extracted_sentences
    except Exception as e:
      raise e

if __name__ == '__main__':
    retriever_pipeline = RetrieverPipeline()
    retriever_result = retriever_pipeline.run_retriever_pipeline(
        file_path = [r"D:\project\exigent_cml\requirements.txt"],
        key = "Payment Terms (Detail)",
        file_type = "text"
)


