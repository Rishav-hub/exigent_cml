import re

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