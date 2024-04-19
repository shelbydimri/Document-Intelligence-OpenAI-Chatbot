Setting up Environment:
Imports necessary libraries.
Sets OpenAI API key.
Loading PDF and Preprocessing:
Loads a PDF file (Booklet-Guide-for-Using-NBC-2016.pdf).
Extracts text from the PDF.
Splits the text into chunks for processing.
Text Embedding and Vectorization:
Uses OpenAI GPT-2 tokenizer for text embedding.
Splits text into chunks for efficient processing.
Embeds the text chunks using OpenAI Embeddings.
Creates a vector database using FAISS for similarity search.
Question Answering Setup:
Sets up a QA chain to answer questions using a knowledge base.
Creates a conversation chain for chat history management.
Chatbot Interaction Functions:
Defines functions to interact with the chatbot.
Saves chat history as a JSONL file.
User Interface Setup:
Sets up a Tkinter window for the chatbot interface.
Creates a text area for displaying chat history.
Creates a text input box for user input.
Creates a button for submitting user input.
Event Handling:
Defines a function to handle button click events.
Binds the <Return> event to the input box to submit user input.
Training the Chatbot Data:
Loads the dataset for training.
Fine-tunes the chatbot model using the training dataset.
Saves the fine-tuned model to an output directory.
Main Execution:
Starts the Tkinter event loop for the chatbot interface.