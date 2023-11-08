import requests
import json
# Define the URL of your FastAPI application
url = "http://0.0.0.0:8000/ml_extraction"

# Define the data to send in the POST request
data = {
    "is_training": True,  # Set to your desired boolean value
    "contract_id": "your_contract_id",
    "c_pk": "your_c_pk",
    "categorial_keys": json.dumps([
        {
            "header_name": 'BUSINESS AND PURPOSE',
            "description": 'BUSINESS AND PURPOSE',
            "fields": [
                {
                "label_name": 'Title of Agreement',
                "type": 'text',
                "options": None,
                "header": 'text',
                "playbook_instruction": 'Title of Agreement',
                "fieldExistTwo": True,
                "display_name": 'Title of Agreement',
                },
                {
                "label_name": 'title of Agreement',
                "type": 'radio',
                "options": None,
                "header": 'text',
                "playbook_instruction": 'Title of Agreement',
                "fieldExistTwo": True,
                "display_name": 'Title of Agreement',
                },
            ]
        },
        {
            "header_name": 'BUSINESS and  PURPOSE',
            "description": 'BUSINESS AND PURPOSE',
            "fields": [
                {
                "label_name": 'Title of Agreement',
                "type": 'text',
                "options": None,
                "header": 'text',
                "playbook_instruction": 'Title of Agreement',
                "fieldExistTwo": True,
                "display_name": 'Title of Agreement',
                },
                {
                "label_name": 'title of Agreement',
                "type": 'radio',
                "options": None,
                "header": 'text',
                "playbook_instruction": 'Title of Agreement',
                "fieldExistTwo": True,
                "display_name": 'Title of Agreement',
                },
            ]
        }
    ]),
}

# Open and read the text file
with open("/home/ubuntu/exigent_cml/4647381113371686-4.txt", "rb") as file:
    files = {"text_file": ("text_file.txt", file.read())}

# Send the POST request
response = requests.post(url, data=data, files=files)

# Check the response
if response.status_code == 200:
    print(response.content)
    print(response.json())
    print("Request was successful.")
else:
    print(f"Request failed with status code {response.status_code}")
