import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

TEMP_DIR = "temp_pdfs"

def clean_temp_directory():
    """Removes the temporary directory and all its contents if it exists, then recreates it."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Error deleting temp directory: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

def process_uploaded_files(uploaded_files, api_key):
    """
    Saves uploaded files to disk, parses them using PyPDFLoader, 
    splits the text into chunks, and indexes them in a FAISS vector database.
    
    Returns:
        tuple: (vector_store, total_pages, total_chunks, doc_summaries)
    """
    # 1. Clean and prepare temp directory
    clean_temp_directory()
    
    documents = []
    file_metadata = {}
    
    # 2. Save and load each uploaded file
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        file_docs = loader.load()
        
        # Ensure metadata contains the clean file name as source and page numbers
        for doc in file_docs:
            doc.metadata["source"] = uploaded_file.name
            
        documents.extend(file_docs)
        file_metadata[uploaded_file.name] = {
            "pages": len(file_docs),
            "docs": file_docs
        }
    
    total_pages = len(documents)
    if total_pages == 0:
        return None, 0, 0, {}
        
    # 3. Chunk the documents
    # Using RecursiveCharacterTextSplitter to split documents into 1000 character chunks with 200 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    
    # 4. Create vector store using Gemini Embeddings
    # models/gemini-embedding-001 has been verified to work with the user's API Key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 5. Generate a summary for each document using Gemini
    doc_summaries = {}
    for filename, info in file_metadata.items():
        doc_summaries[filename] = generate_document_summary(info["docs"], api_key)
        
    return vector_store, total_pages, total_chunks, doc_summaries

def generate_document_summary(doc_pages, api_key):
    """
    Extracts text from the first few pages of a document and generates a 3-bullet-point summary using Gemini.
    """
    # Sample up to 10,000 characters from the first few pages
    sample_text = ""
    for page in doc_pages:
        sample_text += page.page_content + "\n"
        if len(sample_text) > 10000:
            break
    sample_text = sample_text[:10000].strip()
    
    if not sample_text:
        return "Empty document or no extractable text found."
        
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        prompt = PromptTemplate.from_template(
            "You are an expert document summarizer. "
            "Write a concise, professional 3-bullet-point summary of the following document content. "
            "Keep it highly informative and capture the core topic and main points.\n\n"
            "Document Text Sample:\n{content}\n\n"
            "Summary (strictly 3 bullet points):"
        )
        
        chain = prompt | llm
        response = chain.invoke({"content": sample_text})
        return response.content.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

def get_relevant_context_and_sources(query, vector_store, k=5):
    """
    Queries the FAISS vector database to retrieve the top k relevant chunks.
    
    Returns:
        tuple: (formatted_context_string, list_of_source_dictionaries)
    """
    docs = vector_store.similarity_search(query, k=k)
    
    context_parts = []
    sources = []
    
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown PDF")
        # pyPDFLoader page is 0-indexed, let's display it 1-indexed to the user
        page_num = doc.metadata.get("page", 0) + 1
        content = doc.page_content
        
        context_parts.append(
            f"[Source: {source_name}, Page: {page_num}]\n{content}"
        )
        sources.append({
            "source": source_name,
            "page": page_num,
            "content": content
        })
        
    formatted_context = "\n\n---\n\n".join(context_parts)
    return formatted_context, sources

def get_rag_response(query, chat_history, context, api_key):
    """
    Generates a response from Gemini using retrieved context and chat history.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.4
        )
        
        # Format chat history to maintain conversational memory
        history_str = ""
        for role, text in chat_history[-6:]:  # limit to last 6 messages to keep context window clean
            speaker = "User" if role == "user" else "Assistant"
            history_str += f"{speaker}: {text}\n"
            
        system_prompt = (
            "You are an expert AI assistant that helps users understand their PDF documents.\n"
            "Answer the user's question using the retrieved document context and conversational history below.\n"
            "Provide a comprehensive, natural, and helpful response. If the information is not found in the "
            "document context, politely state that it's not in the documents, but try to answer generally if possible "
            "while clearly labeling it as general knowledge outside the context.\n"
            "Be professional and clear. The interface will display source citations separately, so you do not need "
            "to cite file names or page numbers in your text unless helpful for context.\n\n"
            f"Conversation History:\n{history_str}\n"
            f"Retrieved Document Context:\n{context}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        
        response = llm.invoke(system_prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
