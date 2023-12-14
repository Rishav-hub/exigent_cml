import os
import re
import sys
import logging
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
import warnings
from pathlib import Path
from pickle import UnpicklingError
from tqdm import tqdm
from more_itertools import windowed
from typing import List
from typing import List, Optional, Generator, Set, Union, Tuple, Dict, Literal



import nltk
from nltk.tokenize import word_tokenize

from haystack.nodes import PreProcessor
from haystack.nodes.preprocessor.base import BasePreProcessor
from haystack.errors import HaystackError
from haystack.schema import Document

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import (BM25Retriever, EmbeddingRetriever,
                            JoinDocuments, PreProcessor, TextConverter)
from haystack import Pipeline

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

logger = logging.getLogger(__name__)

class CustomPreprocessor(PreProcessor):
    def __init__(self, min_word_count, max_word_count):
        super().__init__()
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        # self.split_by = "passage"
        self.split_respect_sentence_boundary = False
        self.clean_whitespace=False,
        self.clean_header_footer=False,
        self.clean_empty_lines=False

        # print("self.split_by", self.split_by)

    def split_and_check_length(self, input_text):
      # Split text into paragraphs using .\n only if there is no number before the dot
      paragraphs = re.split(r'(?<!\d)(?<=\.\n)(?=\s+\d|\s+[A-Z])', input_text)
      # Initialize result list
      result = []

      # Initialize the sentence tokenizer
      sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

      if len(paragraphs) < 50:
          paragraphs = re.split(r'(?<!\d)(?<=\n)(?=\s+\d|\s+[A-Z])', input_text)
      # Iterate through paragraphs
      for paragraph in paragraphs:
          # Remove extra spaces for length check
          temp_paragraph = ' '.join(paragraph.split())
          # Tokenize the paragraph into sentences
          sentences = sentence_tokenizer.tokenize(temp_paragraph)
          sentence_count = len(sentences)

          # Tokenize the paragraph into words
          words = word_tokenize(temp_paragraph)
          word_count = len(temp_paragraph.split())
          # Check if the word count in the paragraph is less than min_word_count
          if word_count < self.min_word_count:
              # Check the size of both the current paragraph and the combined paragraph
              if result:
                  combined_paragraph = result[-1] + ".\n" + paragraph
                  result[-1] = combined_paragraph
              else:
                  # If result is empty, just append the paragraph
                  result.append(paragraph)
          else:            # Check if the word count exceeds max_word_count
              if word_count > self.max_word_count:
                  # Split the paragraph into two approximately equal halves based on sentence count
                  split_point = sentence_count // 2
                  first_half = ' '.join(sentences[:split_point])
                  second_half = ' '.join(sentences[split_point:])
                  result.extend([first_half, second_half])
              else:
                  result.append(paragraph)


      # print(len(result), file_path)
      return result
    def _create_docs_from_splits(
        self,
        text_splits: List[str],
        splits_pages: List[int],
        splits_start_idxs: List[int],
        headlines: List[Dict],
        meta: Dict,
        split_overlap: int,
        id_hash_keys=Optional[List[str]],) -> List[Document]:
        """
        Creates Document objects from text splits enriching them with page number and headline information if given.
        """
        documents: List[Document] = []

        earliest_rel_hl = 0
        for i, txt in enumerate(text_splits):
            meta = deepcopy(meta)
            doc = Document(content=txt, meta=meta, id_hash_keys=id_hash_keys)
            doc.meta["_split_id"] = i

            documents.append(doc)

        return documents

    def split(
          self,
          document: Union[dict, Document],
          split_by: Optional[Literal["word", "sentence", "passage"]],
          split_length: int,
          split_overlap: int,
          split_respect_sentence_boundary: bool,
          id_hash_keys: Optional[List[str]] = None,) -> List[Document]:
          """Perform document splitting on a single document. This method can split on different units, at different lengths,
          with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
          the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents.
          """
          print("Splitting ------>>>>>>!!!!")
          if id_hash_keys is None:
              id_hash_keys = self.id_hash_keys

          if isinstance(document, dict):
              document["id_hash_keys"] = id_hash_keys
              document = Document.from_dict(document)

          # Mainly needed for type checking
          if not isinstance(document, Document):
              raise HaystackError("Document must not be of type 'dict' but of type 'Document'.")

          if not split_by:
              return [document]

          if not split_length:
              raise Exception("split_length needs be set when using split_by.")

          if split_respect_sentence_boundary and split_by != "word":
              raise NotImplementedError("'split_respect_sentence_boundary=True' is only compatible with split_by='word'.")

          if type(document.content) is not str:
              logger.error("Document content is not of type str. Nothing to split.")
              return [document]

          text = document.content
          headlines = document.meta["headlines"] if "headlines" in document.meta else []

          text_splits = self.split_and_check_length(text)

          # create new document dicts for each text split
          documents = self._create_docs_from_splits(
              text_splits=text_splits,
              splits_pages=None,
              splits_start_idxs=None,
              headlines=headlines,
              meta=document.meta or {},
              split_overlap=split_overlap,
              id_hash_keys=id_hash_keys,
          )

          return documents
    def _process_single(
        self,
        document: Union[dict, Document],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        remove_substrings: Optional[List[str]] = None,
        split_by: Optional[Literal["word", "sentence", "passage"]] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        if remove_substrings is None:
            remove_substrings = []
        if clean_whitespace is None:
            clean_whitespace = self.clean_whitespace
        if clean_header_footer is None:
            clean_header_footer = self.clean_header_footer
        if clean_empty_lines is None:
            clean_empty_lines = self.clean_empty_lines
        if not remove_substrings:
            remove_substrings = self.remove_substrings
        if split_by is None:
            split_by = self.split_by
        if split_length is None:
            split_length = self.split_length
        if split_overlap is None:
            split_overlap = self.split_overlap
        if split_respect_sentence_boundary is None:
            split_respect_sentence_boundary = self.split_respect_sentence_boundary

        split_documents = self.split(
            document=document,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
            id_hash_keys=id_hash_keys,
        )

        split_documents = self._long_documents(split_documents, max_chars_check=self.max_chars_check)

        return split_documents

    def process(
          self,
          documents: Union[dict, Document, List[Union[dict, Document]]],
          clean_whitespace: Optional[bool] = None,
          clean_header_footer: Optional[bool] = None,
          clean_empty_lines: Optional[bool] = None,
          remove_substrings: Optional[List[str]] = None,
          split_by: Optional[Literal["word", "sentence", "passage"]] = None,
          split_length: Optional[int] = None,
          split_overlap: Optional[int] = None,
          split_respect_sentence_boundary: Optional[bool] = None,
          id_hash_keys: Optional[List[str]] = None,
      ) -> List[Document]:
          """
          Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.
          """
          if remove_substrings is None:
              remove_substrings = []
          if not isinstance(documents, list):
              warnings.warn(
                  "Using a single Document as argument to the 'documents' parameter is deprecated. Use a list "
                  "of (a single) Document instead.",
                  DeprecationWarning,
                  2,
              )

          kwargs = {
              "clean_whitespace": clean_whitespace,
              "clean_header_footer": clean_header_footer,
              "clean_empty_lines": clean_empty_lines,
              "remove_substrings": remove_substrings,
              "split_by": split_by,
              "split_length": split_length,
              "split_overlap": split_overlap,
              "split_respect_sentence_boundary": split_respect_sentence_boundary,
          }

          if id_hash_keys is None:
              id_hash_keys = self.id_hash_keys

          if isinstance(documents, (Document, dict)):
              ret = self._process_single(document=documents, id_hash_keys=id_hash_keys, **kwargs)  # type: ignore
          elif isinstance(documents, list):
              ret = self._process_batch(documents=list(documents), id_hash_keys=id_hash_keys, **kwargs)
          else:
              raise Exception("documents provided to PreProcessor.prepreprocess() is not of type list nor Document")

          return ret

    
class DocumentPreprocessor:
    def __init__(self, document_store: ElasticsearchDocumentStore):
        self.indexing_pipeline = self._build_indexing_pipeline(document_store)

    def _build_indexing_pipeline(self, document_store):
        indexing_pipeline = Pipeline()
        text_converter = TextConverter()

        ## TODO -: Need to have a condition which will can switch between preprocessor and custompreprocessor

        # preprocessor = PreProcessor(
        #     clean_whitespace=True,
        #     clean_header_footer=True,
        #     clean_empty_lines=True,
        #     split_by="word",
        #     split_length=100,
        #     split_overlap=3,
        #     split_respect_sentence_boundary=True,
        # )
        preprocessor = CustomPreprocessor(
            min_word_count=30,
            max_word_count = 120
        )
        indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
        indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

        return indexing_pipeline
