from src.pipeline import InferencePipeline
import pandas as pd

if __name__ == "__main__":
#     retriever_pipeline = RetrieverPipeline()
#     retriever_result = retriever_pipeline.run_retriever_pipeline(
#     file_path = [r"D:\project\exigent_cml\requirements.txt"],
#     key = "Payment Terms (Detail)",
#     file_type = "text"
# )
    # Define the paths to the CSV files
    data_paths = {
        "radio_data": "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/retrieval_data/correct_distribution_radio.csv",
        "dd_ck_data": "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/retrieval_data/correct_distribution_dd_ck.csv",
        "number_data": "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/retrieval_data/data_annotated_context_numeber.csv",
        "datetime_data": "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/retrieval_data/datatime_contextV2.csv",
        "text_data": "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/retrieval_data/correct_distribution_txt.csv"
    }
    inference_pipeline = InferencePipeline()
    for df_name, data_path in data_paths.items():
        print(f"Processing DataFrame: {df_name}")
        df = pd.read_csv(data_path)
        for index, row in df.head(2).iterrows():
            file_path = "/content/drive/MyDrive/001_projects/exigent/Contract Management and Extraction (CME)/main_data/contracts/" + row["file_name"]
            key = row["name"]
            file_type = row["type"]
            options = row["option"]
            value = row["value"]
            model_prediction = inference_pipeline.get_inference(file_path, key, options, file_path)
            print(model_prediction)






