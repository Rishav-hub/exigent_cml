import gc

import torch

from config import *
from src.components.custom_model import (DebertaExtractionInference,
                                         Extraction, T5TextGenerationInference)
from src.components.custom_retriever import (RetrieverPipeline,
                                             RetrieverPipelineDateTime)

####### LOADING ALL THE MODEL ########

# Radio T5
text_generation_inference_radio = T5TextGenerationInference(model_name = options_model_name)
# Number T5
text_generation_inference_number = T5TextGenerationInference(model_name = number_model_name)
# Dropdown and checkbox
deberta_inference_dd_ck = DebertaExtractionInference(model_name="microsoft/deberta-base", state_dict_path=dropdown_checkbox_model_name)
# Text T5
text_generation_inference_text = T5TextGenerationInference(model_name = text_t5_model_name)
# Datetime
text_generation_inference_datetime = T5TextGenerationInference(model_name = datetime_model_name)


class InferencePipeline:
    def __init__(self) -> None:
        # Retrieval pipeline for datetime
        self.retriever_pipeline_datetime = RetrieverPipelineDateTime()
    
    def get_inference(self, file_type: str, key: str, options: str, txt_file_path:str, embedding_model: str):
        try:
            # BM25 and embedding retrieval pipeline initiated
            retriever_pipeline = RetrieverPipeline(embedding_model=embedding_model)

            # Inference for file_type as 'text'
            if file_type == TEXT:
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=retriever_pipeline,
                    text_generation_inference=text_generation_inference_text,
                    deberta_extraction_inference=None
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=True)
            
            elif (file_type == CHECKBOX) or (file_type == DROPDOWN):
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=retriever_pipeline,
                    text_generation_inference=text_generation_inference_text,
                    deberta_extraction_inference=deberta_inference_dd_ck
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=False)
            
            elif file_type == NUMBER:
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=retriever_pipeline,
                    text_generation_inference=text_generation_inference_number,
                    deberta_extraction_inference=deberta_inference_dd_ck
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=True)

            elif file_type == RADIO:
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=retriever_pipeline,
                    text_generation_inference=text_generation_inference_radio,
                    deberta_extraction_inference=deberta_inference_dd_ck
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=True) 
            
            elif file_type == DATETIME:
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=self.retriever_pipeline_datetime,
                    text_generation_inference=text_generation_inference_datetime,
                    deberta_extraction_inference=None
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=True) 
            else:
                extraction_pipe = Extraction(
                    file_path=txt_file_path,
                    key=key,
                    file_type=file_type,
                    options=options,
                    retriever_pipeline=retriever_pipeline,
                    text_generation_inference=text_generation_inference_text,
                    deberta_extraction_inference=None
                )
                model_prediction = extraction_pipe.run_extraction(use_t5=True)


            torch.cuda.empty_cache()  # Clear GPU memory after each iteration
            gc.collect()
            return model_prediction
   
        except Exception as e:
            raise e

    