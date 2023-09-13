import pandas as pd
import numpy as np
import torch
import shutil
import os
import random
import evaluate

from random import sample
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoConfig, AutoModel
from datasets import load_dataset, Dataset, concatenate_datasets

def get_context(context: str, file_type:str, name: str, option: str):
  input_context = str()
  if file_type == "text" or file_type == "number":
    input_context = f"""
      Task: extractive_qa
      Type: {file_type}
      Context:

      {context}

      Question:

      What is {name} in context?

      Answer from the context:
  """
  elif file_type == "radio" or file_type == "dropdown" or file_type == "checkbox":
    input_context = f"""
        Task: extractive_qa
        Type: {file_type}
        Context:

        {context}

        Question:

        What is {name} in context?

        Answer should be from these {option}:
    """
  elif file_type == "datetime":
    input_text = f"""
     Task: extractive_qa
     Context:

     {context}

     Question: What is date of {name} in context?
    
     Answer is this datetime format '%y-%m-%d' from the context:
     """

  return input_context

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
  
# Creating the customized model, by adding a drop out and a dense layer on top of deberta to get the final output for the model.
class DebertaClass(torch.nn.Module):
    def __init__(self, target_cols):
        super(DebertaClass, self).__init__()
        self.config = AutoConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=False)
        self.model = AutoModel.from_pretrained("microsoft/deberta-base")
        self.pooler = MeanPooling()
        self.fc1 = torch.nn.Linear(self.config.hidden_size, len(target_cols))

    def forward(self, ids, mask, token_type_ids):
        outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        encoder_layer = outputs[0]

        pooled_output = self.pooler(encoder_layer, mask)
        # Pass the pooled features to the fully connected layer (fc1)
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

