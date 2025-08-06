ref = "data path"
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from transformers import pipeline

def extract_text(ref):
  response = requests.get(ref)
  soup = BeautifulSoup(response.content, "html.parser")
  for script in soup(['scripts']):
      script.extact()
  return soup.get_text().lower()

with open("tennis.txt", "w") as f:
  f.write(extract_text(ref))

loader = TextLoader("tennis.txt")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0, separators=
[" ", ",", "\n"])
docs = text_splitter.split_documents(document)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2",
tokenizer="deepset/roberta-base-squad2")

def nlp_bot():
    print("Welcome! I am GameCoach. What would you like to know?")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['hi', 'hello']:
            print("GameCoach: Hi! How may I assist you?")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("GameCoach: Thank you for consulting GameCoach!")
            break
            answer = qa_pipeline({
            'question': user_input,
            'context': str(docs)
        })

        observed_answer = answer['answer']
        conclusion = formulate_conclusion(observed_answer)

        print(conclusion)

def formulate_conclusion(answer):
return f"GameCoach: {answer}"
nlp_bot()