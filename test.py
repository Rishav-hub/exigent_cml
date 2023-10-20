from src.pipeline import InferencePipeline
import pandas as pd

if __name__ == "__main__":

    # Define the paths to the CSV files
    inference_pipeline = InferencePipeline()
    file_path = ""
    key = ""
    file_type = ""
    options = ""
    model_prediction = inference_pipeline.get_inference(file_path, key, options, file_path)
    print(model_prediction)






