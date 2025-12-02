# MULTI-MODAL-RAG

# 🤖 Multimodal RAG PDF Chatbot

A powerful Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents. Unlike standard RAG systems, this application uses **Multimodal AI** to understand and answer questions about **images, charts, and diagrams** inside the PDF, not just the text.


<img width="829" height="599" alt="Screenshot 2025-12-01 205227" src="https://github.com/user-attachments/assets/29f0fe70-3a6a-43e5-84a1-c2097e01686d" />

## 🌟 Key Features

* **Multimodal Understanding:** Uses **OpenAI CLIP** to embed images and **Google Gemini** to interpret them, allowing the AI to "see" charts and figures.
* **Hybrid Search:** Retrieves context from both text chunks and image captions/embeddings.
* **Interactive UI:** Built with **Streamlit** featuring a modern dark-themed chat interface.
* **Vector Search:** Utilizes **FAISS** for efficient similarity search across document embeddings.
* **PDF Processing:** Extracts high-quality text and images using **PyMuPDF**.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini (via `langchain-google-genai`)
* **Embeddings:** OpenAI CLIP (via `transformers`)
* **Vector Store:** FAISS
* **Orchestration:** LangChain
* **PDF Engine:** PyMuPDF (Fitz)

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sindoo24/MULTI-MODAL-RAG.git
    cd MULTI-MODAL-RAG
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    * Create a `.env` file in the root directory.
    * Add your Google Gemini API key:
        ```env
        GEMINI_API_KEY=your_actual_api_key_here
        ```

## 🏃‍♂️ How to Run

Run the Streamlit app with the following command:

```bash
streamlit run app_streamlit.py
