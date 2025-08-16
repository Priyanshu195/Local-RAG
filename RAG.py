import torch
import ollama
import os
import json
import argparse
import pickle
from openai import OpenAI
import streamlit as st

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to load metadata from metadata.json
def load_metadata():
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r", encoding="utf-8") as metadata_file:
            return [json.loads(line) for line in metadata_file]
    return []

# Function to get relevant context and metadata from the vault based on user input
def get_relevant_context_and_metadata(rewritten_input, vault_embeddings, vault_content, metadata, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return [], []
    
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Get the corresponding context and metadata from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    ###### DEBUG #################
    # print(top_indices)
    relevant_metadata = [metadata[idx] for idx in top_indices]
    
    return relevant_context, relevant_metadata

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, metadata, ollama_model, conversation_history, conversation_history_with_context,source_list):
    # Get relevant context and metadata from the vault
    relevant_context, relevant_metadata = get_relevant_context_and_metadata(user_input, vault_embeddings_tensor, vault_content, metadata, top_k=3)
    
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        # print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        
        # Print the metadata for each retrieved context
        

            # print(type(meta))
        # for meta in relevant_metadata:
        #     print(YELLOW + f"Metadata: {json.dumps(meta, indent=2)}" + RESET_COLOR)
    # else:
    #     print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    conversation_history_with_context.append({"role": "user", "content": user_input_with_context})
    conversation_history.append({"role": "user", "content": user_input})
    
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history_with_context
    ]
    
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    conversation_history_with_context.append({"role": "assistant", "content": response.choices[0].message.content})
    
    

    print(NEON_GREEN + "Response: \n\n" + response.choices[0].message.content + RESET_COLOR)

    #Printing Sources

    if relevant_context:
        print(YELLOW + "Sources:"+ RESET_COLOR)
        seen_sources = set()  # Create a set to store already printed sources

        for meta in relevant_metadata:
            source_file = meta['source_file']
            if source_file not in seen_sources:  # Check if the source is not already seen
                print(YELLOW + str(source_file) + RESET_COLOR)
                seen_sources.add(source_file)  # Add the source to the set
        source_list.append(seen_sources)
    else:
        print(CYAN + "No relevant source document found." + RESET_COLOR)
    # Return the content of the response from the model
    return response.choices[0].message.content

def generate_embeddings_tensor(vault_content):
    # Generate embeddings for the vault content using Ollama
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    vault_embeddings_tensor = torch.tensor(vault_embeddings) 
    return vault_embeddings_tensor

# Set page configuration
st.set_page_config(page_title="RAG Model Interface", layout="wide")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "conversation_history_with_context" not in st.session_state:
    st.session_state.conversation_history_with_context = []

# Load the vault content
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding="utf-8") as vault_file:
        vault_content = vault_file.readlines()

# Load metadata (assuming `load_metadata` is defined)
metadata = load_metadata()
source_list = []
# Read the flag value
FLAG_FILE_PATH = "flag.txt"
if os.path.exists(FLAG_FILE_PATH):
    with open(FLAG_FILE_PATH, "r") as flag_file:
        flag = flag_file.read().strip()
else:
    flag = "0"

# Update embeddings if necessary
if flag == "1":
    with open("embeddings.pkl", "rb") as file:
        vault_embeddings_tensor = pickle.load(file)
else:
    vault_embeddings_tensor = generate_embeddings_tensor(vault_content)
    with open("embeddings.pkl", "wb") as file:
        pickle.dump(vault_embeddings_tensor, file)
    with open(FLAG_FILE_PATH, "w") as flag_file:
        flag_file.write("1")
    st.sidebar.success("Embeddings Updated")

# Initialize the model client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="llama3"
)

# Sidebar configuration
st.sidebar.title("RAG Model Configuration")
selected_model = st.sidebar.selectbox("Select Model", ["llama3", "other_model"])
system_message = st.sidebar.text_area("System Message", "You are a helpful assistant that is an expert at extracting the most useful information from a given text")

# Main Interface
st.title("RAG Model Interface")
st.write("Ask a question about your documents:")

# User input
user_input = st.text_input("Your Question", placeholder="Type your question here...")
if st.button("Ask"):
    if user_input:
        response = ollama_chat(
            user_input,
            system_message,
            vault_embeddings_tensor,
            vault_content,
            metadata,
            selected_model,
            st.session_state.conversation_history,
            st.session_state.conversation_history_with_context,
            source_list
        )
    else:
        st.warning("Please enter a question.")

# Display conversation history
if st.session_state.conversation_history:
    st.write("### Conversation History")
    # st.write(f"{ st.session_state.conversation_history:}")
    for i,entry in enumerate(st.session_state.conversation_history):
        sources = source_list[i//2]
        if entry['role']=='user':
            st.write(f"**You:** {entry['content']}")
        elif entry['role']=='assistant':
            st.write(f"**Assistant:** {entry['content']}")
            st.write(f"**Sources:** ")
            for source in sources:
                st.write(f"{source}")

        
