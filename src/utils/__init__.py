import re
import os
import shutil

# Function to normalize text
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using regular expression
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def concatinate_documents(prediction):

  total_text = str()

  for texts in prediction:
    total_text = total_text + "\n" + texts.content

  return total_text

def join_sentences(prediction):
  extracted_sentences = ' '.join(prediction)
  return extracted_sentences

def save_pdf_to_directory(text_file_path: str, ROOT: str = os.getcwd()):
    """
    The function saves a PDF file to a specified directory by creating a subdirectory with the PDF
    file's name and saving the file in that subdirectory.
    
    :param text_file_path: The path to the PDF file that you want to save to a directory
    :type text_file_path: str
    :param ROOT: The `ROOT` parameter is the root directory where you want to save the PDF file
    :type ROOT: str
    :return: the path to the saved PDF file.
    """
    # Extract the PDF file name (excluding the extension)
    text_file_name = os.path.basename(text_file_path)[:-4]

    # Create a subdirectory with the PDF file's name (excluding the extension)
    text_folder_dir = os.path.join(ROOT, "temp", text_file_name)
    extract_dir = os.path.join(text_folder_dir, "text_file")
    os.makedirs(extract_dir, exist_ok=True)

    # Define the path to save the PDF file
    input_pdf_path = os.path.join(extract_dir, os.path.basename(text_file_path))

    # Read and save the uploaded PDF file
    with open(text_file_path, "rb") as source_file, open(input_pdf_path, "wb") as target_file:
        target_file.write(source_file.read())
        source_file.close()
        target_file.close()

    return text_folder_dir, input_pdf_path

def save_pdf_to_directory_fastapi(text_file, ROOT: str = os.getcwd()):
    """
    The function `save_pdf_to_directory_fastapi` saves an uploaded PDF file to a specified directory in
    a FastAPI application.
    
    :param text_file: The `text_file` parameter is of type `UploadFile` and represents the uploaded PDF
    file. It contains information about the file, such as its filename and file object
    :type text_file: UploadFile
    :param ROOT: The `ROOT` parameter is the root directory where you want to save the PDF file. By
    default, it is set to the current working directory (`os.getcwd()`)
    :type ROOT: str
    :return: the path where the uploaded PDF file is saved.
    """
    # Create a subdirectory with the PDF file's name (excluding the extension)
    text_folder_dir = os.path.join(ROOT, "temp", text_file.filename[:-4])
    extract_dir = os.path.join(text_folder_dir, "text_file")
    os.makedirs(extract_dir, exist_ok=True)

    # Define the path to save the PDF file
    input_text_path = os.path.join(extract_dir, text_file.filename)
    
    # Save the uploaded PDF file
    with open(input_text_path, "wb") as f:
        shutil.copyfileobj(text_file.file, f)
        f.close()
    
    return text_folder_dir, input_text_path
