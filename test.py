from src.pipeline import InferencePipeline
import pandas as pd

if __name__ == "__main__":

    # Define the paths to the CSV files
    inference_pipeline = InferencePipeline()
    file_path = "/home/ubuntu/exigent_cml/artifacts/LIServiceAgr-SVBFinancialGroup-MichaelDreyer.txt"
    key = "Effective Date"
    file_type = "datetime"
    options = None
    model_prediction = inference_pipeline.get_inference(
        file_type,
        key, 
        options, 
        file_path,
        embedding_model = "RishuD7/finetune_base_bge_pretrained_v4")
    print(model_prediction)






