from src.components.custom_retriever import RetrieverPipeline

if __name__ == "__main__":
    retriever_pipeline = RetrieverPipeline()
    retriever_result = retriever_pipeline.run_retriever_pipeline(
    file_path = [r"D:\project\exigent_cml\requirements.txt"],
    key = "Payment Terms (Detail)",
    file_type = "text"
)

