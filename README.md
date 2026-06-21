# AI-Powered PDF Chatbot using Gemini, LangChain, FAISS, and Streamlit (RAG Architecture)

An intelligent, conversational document assistant that allows users to upload multiple PDF documents and ask questions through a clean, ChatGPT-like interface. Instead of manually scanning large documents, users can query their PDFs naturally and receive accurate responses with page-level citations.

---

## 📌 Problem Statement

Reading and extraction of specific information from large PDFs (e.g., research papers, technical manuals, financial reports, synopses) is time-consuming. This project provides a natural language solution using Retrieval-Augmented Generation (RAG) to find precise facts and summarize documents instantly.

---

## 🛠️ System Workflow

```text
User Uploads PDF(s)
        ↓
PDF Text Extraction (PyPDFLoader)
        ↓
Text Chunking (RecursiveCharacterTextSplitter)
        ↓
Generate Embeddings (models/gemini-embedding-001)
        ↓
Store in FAISS Vector DB
        ↓
User Asks Question
        ↓
Similarity Search (Retrieve Relevant Chunks)
        ↓
Gemini LLM (Context + Query Processing)
        ↓
Generate Final Answer
        ↓
Display Response & Page-Level Citations in Chat UI
```

---

## ✨ Features

*   **Multi-PDF Support:** Upload one or more documents simultaneously.
*   **Automatic Text Extraction:** Parses text structures using LangChain's document loaders.
*   **Semantic Vector Database:** Fast, localized vector search utilizing Facebook AI Similarity Search (FAISS).
*   **Document Summary Panel:** Automatically generates a 3-bullet point overview for each uploaded PDF.
*   **Conversational Memory:** Retains memory of the current chat session for natural multi-turn conversations.
*   **Source Citations:** Displays exact referenced paragraphs along with file names and page numbers.
*   **Premium UI Design:** Responsive, dark/light theme integrations, custom gradients, and styled user/assistant avatars.
*   **Session Exports:** Download your entire chat transcript and citations directly from the sidebar.

---

## 💻 Local Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ai-pdf-chatbot.git
    cd ai-pdf-chatbot
    ```

2.  **Install Dependencies:**
    Make sure you have Python 3.9+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Configuration:**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY="your-gemini-api-key-here"
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The app will open automatically in your browser at `http://localhost:8501` (or `8502`).

---

## 🚀 Deployment to Streamlit Community Cloud

1.  Push the repository to GitHub.
2.  Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Click **New App**, select your repository, branch (`main`), and set the main file path to `app.py`.
4.  Before deploying, open **Advanced settings...**.
5.  In the **Secrets** section, configure your Google API key:
    ```toml
    GOOGLE_API_KEY = "your-actual-gemini-api-key"
    ```
6.  Click **Deploy**!

---

## 📂 Project Structure

```text
ai-pdf-chatbot/
│
├── app.py                 # Streamlit UI & chat state controller
├── rag_helper.py          # Backend RAG loader, chunker, vector database, and QA chain
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignored files (.env, caches)
├── screenshots/           # Application screenshots for GitHub
│   ├── home.png
│   ├── upload.png
│   └── chat.png
└── assets/                # Design assets & icons
```
