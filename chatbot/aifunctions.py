import re
import numpy as np
import json
from dotenv import load_dotenv
from django.db import transaction
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from .models import DocumentEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI
import os
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from .models import DocumentEmbedding
from django.db import transaction
from sklearn.metrics.pairwise import cosine_similarity
from .models import DocumentEmbedding
import pandas as pd
from langchain.schema import HumanMessage, SystemMessage  # Import required message classes
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import fitz

from django.utils.html import format_html

def format_ai_response(content):
    # Replace `\n` with `<br>`
    content = content.replace("\n", "<br>")
    
    # Replace `**text**` with `<strong>text</strong>`
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    
    return format_html(content)

# Ensure API key for OpenAI or other LLM provider is set
api_key=""

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=api_key)


def fetch_all_documents_as_history():
    # Retrieve all document contents from DocumentEmbedding
    documents = DocumentEmbedding.objects.all()
    
    # Concatenate each document's content into a single chat history string
    chat_history = "\n\n".join([doc.content for doc in documents])
    return chat_history

def ask_question_with_chat_history(user_question):
    # Fetch chat history by retrieving all document contents
    chat_history = fetch_all_documents_as_history()
    
    # Format the input as a list of messages, beginning with chat history as context
    messages = [
        SystemMessage(content="This is a summary of previous documents:"),
        HumanMessage(content=chat_history),
        HumanMessage(content=user_question)
    ]

    # Call model.invoke with the formatted list of messages
    response = model.invoke(messages)

    print(response.content)

    # Return the answer directly from the response
    return response.content  # Adjust based on response structure if needed



# Function to create embeddings and store them in SQLite
def create_and_store_embeddings_from_uploads():
    # Define the directory where files are uploaded
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(current_dir, "uploads")
    
    if not os.path.exists(uploads_dir):
        print("Uploads directory does not exist.")
        return
    
    # Initialize embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Process each file in the uploads directory
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Split the content into smaller chunks
        docs = text_splitter.split_text(content)
        
        with transaction.atomic():  # Ensure database operations are atomic
            for doc_text in docs:
                # Generate embedding for the chunk
                embedding_vector = embeddings_model.embed_query(doc_text)
                embedding_binary = np.array(embedding_vector).tobytes()  # Convert to binary

                # Save embedding in the Django model
                DocumentEmbedding.objects.create(
                    content=doc_text,
                    embedding=embedding_binary,
                    source_path=file_path,
                    metadata={"source": filename}
                )





def fetch_and_store_embeddings(user_input):
    # Extract URL from user input
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, user_input)
    
    if not urls:
        print("No URL found in input. Please provide a URL in the input.")
        return
    
    url = urls[0]

    # Step 1: Scrape content from the provided URL
    loader = WebBaseLoader([url])
    documents = loader.load()

    # Step 2: Split the scraped content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1400, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Step 3: Generate embeddings for each document chunk
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    with transaction.atomic():  # Ensure database operations are atomic
        for doc in docs:
            # Generate embedding and validate the size
            embedding_vector = embeddings_model.embed_query(doc.page_content)
            embedding_array = np.array(embedding_vector, dtype=np.float32)
            
            # Check the embedding dimension
            if embedding_array.shape[0] != 1536:
                print(f"Warning: Unexpected embedding dimension {embedding_array.shape}. Expected (1536,).")
                continue  # Skip if the dimension is incorrect

            # Convert embedding to binary format for storage
            embedding_binary = embedding_array.tobytes()
            
            # Save to the SQLite database
            DocumentEmbedding.objects.create(
                content=doc.page_content,
                embedding=embedding_binary,
                source_url=url,
                metadata=json.dumps({"source": url})
            )

def get_relevant_context(question):
    # Embed the question
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    question_embedding = np.array(embeddings_model.embed_query(question))  # Convert to NumPy array
    
    # Retrieve stored embeddings and their corresponding content
    document_embeddings = DocumentEmbedding.objects.all()
    contexts = []
    
    for document in document_embeddings:
        embedding_binary = np.frombuffer(document.embedding, dtype=np.float32)
        content = document.content
        
        # Check dimensions for troubleshooting
        if embedding_binary.shape != question_embedding.shape:
            print(f"Dimension mismatch: question embedding shape {question_embedding.shape} vs document embedding shape {embedding_binary.shape}")
            continue  # Skip mismatched embeddings

        print(question_embedding.shape, embedding_binary.shape)
        contexts.append((content, embedding_binary))
    
    # Calculate similarities
    similarities = [(content, cosine_similarity([question_embedding], [embedding])[0][0]) 
                    for content, embedding in contexts]
    
    # Sort by similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Select the top contexts (you can adjust the number of top contexts as needed)
    top_contexts = [content for content, _ in similarities[:3]]
    
    # Combine the top contexts into a single prompt for the AI model
    context_for_question = "\n".join(top_contexts)
    
    # Send the question and context to the AI model
    response = ask_ai_with_context(question, context_for_question)
    return response



def ask_ai_with_context(question, context):
    # Format the prompt for the AI model
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    # Call your AI model with the prompt
    result = model.invoke(prompt)
    return result


# user_input = "Fetch information from https://www.pavitech.co.ke/ about new product announcements."
# fetch_and_store_embeddings(user_input)
# user_query = "what is in pavitech"
# retrieve_similar_documents(user_query)

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Define a function to generate the response
def generate_feedback_response(feedback):
    result = chain.invoke({"feedback": feedback})
    return result


def process_pdf_embeddings(file_path, source_name, api_key=api_key):
    """
    Process a PDF file by path, generate embeddings, and store them in the database.
    file_path: Path to the PDF file.
    source_name: Name or identifier for the source.
    api_key: OpenAI API key for generating embeddings.
    """
    file_path = file_path.replace("\\", "\\\\")
    # Initialize the loader with a PDF file path
    loader = PyMuPDFLoader(file_path)

    # Load and process the PDF content
    documents = loader.load()


    # Step 2: Split the scraped content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1400, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Step 3: Generate embeddings for each document chunk
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    with transaction.atomic():  # Ensure database operations are atomic
        for doc in docs:
            print("Document Content:", doc.page_content)  # This will print the raw content of the document
            
            # Generate embedding and validate the size
            embedding_vector = embeddings_model.embed_query(doc.page_content)
            embedding_array = np.array(embedding_vector, dtype=np.float32)
            
            # Check the embedding dimension
            if embedding_array.shape[0] != 1536:
                print(f"Warning: Unexpected embedding dimension {embedding_array.shape}. Expected (1536,).")
                continue  # Skip if the dimension is incorrect

            # Convert embedding to binary format for storage
            embedding_binary = embedding_array.tobytes()
            
            # Save to the SQLite database
            DocumentEmbedding.objects.create(
                content=doc.page_content,
                embedding=embedding_binary,
                source_url=file_path,
                metadata=json.dumps({"source": file_path})
            )

def answer_from_pdf_embeddings(question, api_key=api_key):
    """
    Use embeddings from a stored PDF to answer a question.
    question: The user's question as a string.
    api_key: OpenAI API key for generating the question embedding.
    """
    context = get_relevant_context(question=question)
    response = ask_ai_with_context(question=question, context=context)
    return response