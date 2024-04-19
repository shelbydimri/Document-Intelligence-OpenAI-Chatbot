import os
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import tkinter as tk
from tkinter import scrolledtext
from train_chatbot import main as train_chatbot

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Loading PDFs and chunking with LangChain

# Simple method - Split by pages 
loader = PyPDFLoader("./Booklet-Guide-for-Using-NBC-2016.pdf")
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text using PyMuPDF
import fitz

doc = fitz.open("./Booklet-Guide-for-Using-NBC-2016.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Close the document after extracting text
doc.close()

# Step 2: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Embed text and store embeddings
# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Setup retrieval function

# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type="stuff")

# Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1), db.as_retriever())

# Function to interact with the chatbot, update the chat history, and save the dataset
import jsonlines

def save_chat_history_as_jsonl(chat_history, jsonl_file):
    """
    Save the chat history as a JSONL file.

    Parameters:
    - chat_history: List of tuples containing user prompts and chatbot responses.
    - jsonl_file: Name of the JSONL file to save the chat history.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the JSONL file in the same directory as the script
    jsonl_file_path = os.path.join(script_dir, jsonl_file)

    with jsonlines.open(jsonl_file_path, mode='a') as writer:
        for prompt, response in chat_history:
            writer.write({"prompt": prompt, "completion": response})

import os
import jsonlines
import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from train_chatbot import main as train_chatbot

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Loading PDFs and chunking with LangChain

# Simple method - Split by pages 
loader = PyPDFLoader("./Booklet-Guide-for-Using-NBC-2016.pdf")
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text using PyMuPDF
import fitz

doc = fitz.open("./Booklet-Guide-for-Using-NBC-2016.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Close the document after extracting text
doc.close()

# Step 2: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Embed text and store embeddings
# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Setup retrieval function

# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type="stuff")

# Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1), db.as_retriever())

# Define JSONL file name
JSONL_FILE = "chat_history.jsonl"

# Function to save chat history as JSONL
def save_chat_history_as_jsonl(chat_history):
    """
    Save the chat history as a JSONL file.

    Parameters:
    - chat_history: List of tuples containing user prompts and chatbot responses.
    """
    with jsonlines.open(JSONL_FILE, mode='a') as writer:
        for prompt, response in chat_history:
            writer.write({"prompt": prompt, "completion": response})

# Function to interact with the chatbot
def interact_with_chatbot(input_text, chat_history_text):
    global JSONL_FILE  # Access the global JSONL_FILE variable
    # Check if the user wants to exit
    if input_text.lower() == 'exit':
        chat_history_text.insert(tk.END, "Thank you for chatting with the chatbot!\n")
        # Save the chat history as a JSONL file
        save_chat_history_as_jsonl(chat_history)
        # Check if the JSONL file is not empty before training the chatbot
        if os.path.getsize(JSONL_FILE) > 0:
            train_chatbot()  # Train the chatbot using the JSONL file
        else:
            print("JSONL file is empty. Chatbot training skipped.")
        return

    # Get the current chat history from the text area
    current_chat_history = chat_history_text.get("1.0", tk.END)

    # Convert the chat history to the expected format (list of tuples)
    chat_history = [tuple(line.split(': ', 1)) for line in current_chat_history.split('\n') if line]

    # Use the chatbot to generate a response to the user's input
    response = qa({"question": input_text, "chat_history": chat_history})

    # Display the user's input and the chatbot's response
    chat_history_text.insert(tk.END, f'User: {input_text}\n', 'user_response')
    chat_history_text.insert(tk.END, f'Chatbot: {response["answer"]}\n', 'chatbot_response')

    # Record the conversation
    chat_history.append((input_text, response["answer"]))

    # Save the chat history as a JSONL file
    save_chat_history_as_jsonl([(input_text, response["answer"])])

    # Update the chat history with the recorded conversation
    chat_history_text.delete("1.0", tk.END)
    for prompt, reply in chat_history:
        chat_history_text.insert(tk.END, f'{prompt}: {reply}\n', 'user_response' if prompt.startswith('User') else 'chatbot_response')
    

# Function to handle button click event
# Create the main window
window = tk.Tk()
window.title("Chatbot")


# Create a text area to display the chat history
chat_history_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=40)
chat_history_text.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

# Create a text input box for the user to enter their questions
input_box = tk.Text(window, wrap=tk.WORD, width=30, height=3)
input_box.grid(row=1, column=0, padx=10, pady=10)

# Create a button to submit the user's input
submit_button = tk.Button(window, text="Submit", width=10, command=lambda: on_submit())
submit_button.grid(row=1, column=1, padx=10, pady=10)
# Configure text colors
chat_history_text.tag_config('user_response', foreground='green')
chat_history_text.tag_config('chatbot_response', foreground='blue')

# Function to handle button click event
def on_submit():
    # Get the user's input text
    input_text = input_box.get("1.0", tk.END).strip()

    # Clear the input box
    input_box.delete("1.0", tk.END)

    # Interact with the chatbot
    interact_with_chatbot(input_text, chat_history_text)

# Bind the <Return> event to the input box to call on_submit() when Enter is pressed
input_box.bind("<Return>", lambda event: on_submit())

# Start the Tkinter event loop
window.mainloop()

    

# Function to handle button click event
# Create the main window
window = tk.Tk()
window.title("Chatbot")


# Create a text area to display the chat history
chat_history_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=40)
chat_history_text.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

# Create a text input box for the user to enter their questions
input_box = tk.Text(window, wrap=tk.WORD, width=30, height=3)
input_box.grid(row=1, column=0, padx=10, pady=10)

# Create a button to submit the user's input
submit_button = tk.Button(window, text="Submit", width=10, command=lambda: on_submit())
submit_button.grid(row=1, column=1, padx=10, pady=10)
# Configure text colors
chat_history_text.tag_config('user_response', foreground='green')
chat_history_text.tag_config('chatbot_response', foreground='blue')

# Function to handle button click event
def on_submit():
    # Get the user's input text
    input_text = input_box.get("1.0", tk.END).strip()

    # Clear the input box
    input_box.delete("1.0", tk.END)

    # Interact with the chatbot
    interact_with_chatbot(input_text, chat_history_text)

# Bind the <Return> event to the input box to call on_submit() when Enter is pressed
input_box.bind("<Return>", lambda event: on_submit())

# Start the Tkinter event loop
window.mainloop()
