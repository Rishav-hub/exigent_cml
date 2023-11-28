import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form 

from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form 

from redis import Redis
import json
from rq import Queue
from dotenv import load_dotenv

import os, requests

from src.pipeline import InferencePipeline
from src.utils import save_pdf_to_directory, save_pdf_to_directory_fastapi

update_extraction_api = "http://10.0.146.154:5000/production_app/update_extraction/"
load_dotenv()



app = FastAPI()

redis_conn = Redis(
    host=os.getenv('REDIS_HOST'),
    port=os.getenv('REDIS_PORT'),
    password=os.getenv('REDIS_PASSWORD')
)

task_queue = Queue(connection=redis_conn, default_timeout=36000)
inference_pipeline = InferencePipeline()

def background_process(text_file, categorial_keys, data_request):
    try:
        data_model_structure = dict()
        final_output = dict()
        _, txt_file_path = save_pdf_to_directory_fastapi(text_file=text_file)
        embedding_model: str = "RishuD7/finetune_base_bge_pretrained_v4"
        for categorical_key in categorial_keys:
            header_name = categorical_key['header_name']
            print(categorical_key['fields'])
            for field in categorical_key['fields']:
                print(field)
                file_type: str = field['type']
                key: str = field['label_name']
                options: str = field['options']
                txt_file_path:str = txt_file_path
                
                model_prediction, concatenated_retrieval_result = inference_pipeline.get_inference(
                    file_type=file_type,
                    key=key,
                    options=options,
                    txt_file_path=txt_file_path,
                    embedding_model=embedding_model
                )
                final_output[key] = {
                    "value" : str(model_prediction),
                    "instruction" : str(concatenated_retrieval_result) 
                }

            data_model_structure[header_name] = final_output

            print("final_output", final_output)
        print("data_model_structure", data_model_structure)
	# Append outputs to the data request
        data_request['final_output_json'] = json.dumps(final_output)
        data_request['datamodel_structure'] = json.dumps(data_model_structure)
	# Send predicted value to main extraction API
        a = requests.post(update_extraction_api, data=data_request)
        print("Status of Update extraction API is ", a.status_code)
        return "DOne"
    except Exception as e:
        raise e

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/ml_extraction")
async def ml_extraction(
    is_training: Annotated[bool, Form()],
    contract_id: Annotated[str, Form()],
    c_pk: Annotated[str, Form()],
    bucket_name: Annotated[str, Form()],
    text_file: Annotated[UploadFile, File()],
    categorial_keys: Annotated[str, Form()],
):
    print(is_training)
    print(contract_id)
    print(c_pk)
    print(text_file)
    print(categorial_keys)
    print(bucket_name)

    categorial_keys = json.loads(categorial_keys)
    data_request = {
	"is_training" : is_training,
	"contract_id" : contract_id,
	"c_pk" : c_pk,
	"bucket_name": bucket_name
	}

    output = background_process(text_file, categorial_keys, data_request)

    print(output)

    print(json.dumps(categorial_keys))
    return JSONResponse(
        status_code=200,
        content=f"Processing in Queue"
    )

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
