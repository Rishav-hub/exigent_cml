import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModel

from src.utils import join_sentences, concatinate_documents
from src.components.custom_retriever import RetrieverPipeline
from config import *

def get_context(context: str, file_type:str, name: str, option: str):
  input_context = str()
  if file_type == TEXT or file_type == NUMBER:
    input_context = f"""
      Task: extractive_qa
      Type: {file_type}
      Context:

      {context}

      Question:

      What is {name} in context?

      Answer from the context:
  """
  elif file_type == RADIO or file_type == DROPDOWN or file_type == CHECKBOX:
    input_context = f"""
        Task: extractive_qa
        Type: {file_type}
        Context:

        {context}

        Question:

        What is {name} in context?

        Answer should be from these {option}:
    """
  elif file_type == DATETIME:
    input_context = f"""
     Task: extractive_qa
     Context:

     {context}

     Question: What is date of {name} in context?
    
     Answer is this datetime format '%y-%m-%d' from the context:
     """

  return input_context
  
# Creating the customized model, by adding a drop out and a dense layer on top of deberta to get the final output for the model.
class DebertaClass(torch.nn.Module):
    def __init__(self, target_cols, model_name):
        super(DebertaClass, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooling()
        self.fc1 = torch.nn.Linear(self.config.hidden_size, len(target_cols))

    def forward(self, ids, mask, token_type_ids):
        outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        encoder_layer = outputs[0]

        pooled_output = self.pooler(encoder_layer, mask)
        # Pass the pooled features to the fully connected la
        # 3yer (fc1)
        output = self.fc1(pooled_output)
        return output

class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class T5TextGenerationInference:
  def __init__(self, model_name: str, top_p=0.95, temperature=0.3,):
    self.model_name = model_name
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = self._load_tokenizer()
    self.model = self._load_model()
    self.top_p = top_p
    self.temperature = temperature
    self.max_new_tokens = 1024
    self.repetition_penalty = 1.0
    self.num_return_sequences = 1
    self.no_repeat_ngram_size = 2
    self.num_beams = 1

  @staticmethod
  def _load_tokenizer(model_name: str = "google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

  def _load_model(self):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        self.model_name,
    ).to(self.device)
    return model

  def run_inference(self, inputs: str):

    # Get all the components from input
    context, file_type, name, option = inputs['context'], inputs['file_type'], inputs['name'], inputs["option"]

    # pass it to form the context template
    input_context = get_context(context, file_type, name, option)

    # Tokenize the context
    tokenized_text = self.tokenizer(input_context,
                          padding= "max_length",
                          return_tensors='pt').to(self.device)

    # Pass it to the model for inference
    model_output = self.model.generate(
        input_ids = tokenized_text.input_ids,
        max_new_tokens = self.max_new_tokens,
        num_return_sequences=self.num_return_sequences,
        no_repeat_ngram_size = self.no_repeat_ngram_size,
        repetition_penalty = self.repetition_penalty,
        output_scores=True,
        num_beams=self.num_beams,
        top_p=self.top_p,
        temperature=self.temperature
    )

    # Decode the model prediction to get the text back from input ids
    model_prediction: str = self.tokenizer.decode(
              model_output[0],
              skip_special_tokens=True
            )
    return model_prediction

class DebertaExtractionInference:
  def __init__(self, model_name: str, state_dict_path: str):
    self.model_name = model_name
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.state_dict = torch.load(state_dict_path)
    self.target_cols = self.state_dict["target_cols"]

    self.tokenizer = self._load_tokenizer(model_name=self.model_name)
    self.model = self._load_model()

  @staticmethod
  def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

  def _load_model(self):
    model = DebertaClass(self.target_cols, self.model_name)
    model.to(self.device)
    model.load_state_dict(self.state_dict["state_dict"])
    return model

  def run_inference(self, inputs: str):

    # Get all the components from input
    context, file_type, name, option = inputs['context'], inputs['file_type'], inputs['name'], inputs["option"]

    # pass it to form the context template
    input_context = f"Type: {file_type}, Question: {name}, From the Option: {option} Context: {context}"

    inputs = self.tokenizer.encode_plus(
        input_context,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_token_type_ids=True,
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(self.device)

    # Pass it to the model for inference
    model_output = self.model(ids, mask, token_type_ids)

    # Get the index with maximum probability and get the value from the target column
    model_prediction = self.target_cols[np.argmax(model_output.cpu().detach().numpy().tolist())]

    return model_prediction

class Extraction:
  def __init__(self, file_path: str,
               key: str,
               file_type: str,
               options: str,
               retriever_pipeline: RetrieverPipeline,
               text_generation_inference: T5TextGenerationInference,
               deberta_extraction_inference: DebertaExtractionInference):
    self.file_path = file_path
    self.key = key
    self.file_type = file_type
    self.options = options
    self.retriever_pipeline = retriever_pipeline
    self.text_generation_inference = text_generation_inference
    self.deberta_extraction_inference = deberta_extraction_inference

  def run_extraction(self, use_t5 = True):
    try:
      # Running retrieval pipeline
      if self.file_type == DATETIME:
        retriever_result = self.retriever_pipeline.run_retriever_pipeline(
          file_path = self.file_path
        )
        concatenated_retrieval_result = join_sentences(retriever_result)
      else:
        retriever_result = self.retriever_pipeline.run_retriever_pipeline(
          file_path = [self.file_path],
          key = self.key,
          file_type = self.file_type
        )

        # Retriever result must be concatenated
        concatenated_retrieval_result = concatinate_documents(retriever_result)

      # Prepare the inputs for
      inputs = {
          'context' : concatenated_retrieval_result,
          'file_type' : self.file_type,
          'name' : self.key,
          'option' : self.options
      }

      if use_t5:
        # Running flan-t5 pipeline
        model_prediction = self.text_generation_inference.run_inference(inputs = inputs)
      else:
        # Running DeBERTA pipeline
        model_prediction = self.deberta_extraction_inference.run_inference(inputs = inputs)

      return model_prediction, concatenated_retrieval_result
    except Exception as e:
      raise e
    