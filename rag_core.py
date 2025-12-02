import fitz
import os
import io
import base64
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import CLIPModel, CLIPProcessor
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv() 

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file or as an environment variable.")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

class CLIPEmbeddings(Embeddings):
    """Custom LangChain Embeddings class for CLIP."""
    def __init__(self, embed_text_func):
        self.embed_text_func = embed_text_func

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text_func(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text_func(text).tolist()

def embed_image(img_data: Any) -> np.ndarray:
    """Embeds a PIL Image object."""
    if isinstance(img_data, str):
        image = Image.open(img_data).convert('RGB')
    else:
        image = img_data
        
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()

def embed_text(text: str) -> np.ndarray:
    """Embeds a text string."""
    inputs = processor(text=text, return_tensors='pt', padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()



def process_pdf_and_create_vector_store(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Processes the PDF, extracts text and images, creates embeddings,
    and returns the FAISS vector store and image data store.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_docs = []
    all_embeddings = []
    image_data_store = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, 'type': 'text'})
            text_chunks = splitter.split_documents([temp_doc])

            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image['image']

                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                image_id = f'page_{i}_img_{img_index}'
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format='PNG')
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = image_base64

                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)

                image_doc = Document(page_content=f'[Image:{image_id}]',
                                     metadata={"page": i, "type": "image", "image_id": image_id})
                all_docs.append(image_doc)
            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue
    doc.close()

    if not all_embeddings:
        return None 

    embeddings_array = np.array(all_embeddings)
    clip_embeddings_instance = CLIPEmbeddings(embed_text)

    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=clip_embeddings_instance,
        metadatas=[doc.metadata for doc in all_docs]
    )
    
    return {
        "vector_store": vector_store,
        "image_data_store": image_data_store,
        "all_docs": all_docs 
    }


def retrieve_multimodal(query: str, vector_store: FAISS, image_data_store: Dict[str, str], k: int = 5) -> List[Document]:
    """Retrieves the top k most relevant text and image documents."""
    query_embedding = embed_text(query)
    
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding, k=k*2 
    )
    
    retrieved_embeddings = []
    for doc in results:
        if doc.metadata.get('type') == 'text':
            retrieved_embeddings.append(embed_text(doc.page_content))
        elif doc.metadata.get('type') == 'image':
            if doc.metadata.get('subtype') == 'text_ref':
                 retrieved_embeddings.append(embed_text(doc.page_content))
            
            else:
                image_base64 = image_data_store.get(doc.metadata['image_id']) 

    similarities = cosine_similarity(query_embedding.reshape(1, -1), np.array(retrieved_embeddings)).flatten()
    for i, doc in enumerate(results):
        doc.metadata['similarity'] = similarities[i]

    results_sorted = sorted(results, key=lambda x: x.metadata['similarity'], reverse=True)
    return results_sorted[:k]

def create_multi_modal_message(query: str, retrieved_docs: List[Document], image_data_store: Dict[str, str]) -> HumanMessage:
    """Formats the retrieved context into a HumanMessage object for the Gemini LLM."""
    content = []
    content.append({"type": "text", "text": f"You are an expert document analyzer. Based ONLY on the provided context (text and images), answer the question. If the information is not present, state that you cannot answer from the context.\n\nQuestion: {query}\n\nContext:\n"})
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]


    if text_docs:
        text_content = "\n\n---\n\n".join([
            f"[page {doc.metadata['page']}, Similarity: {doc.metadata['similarity']:.3f}]:\n{doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f'--- Text Excerpts ---\n\n{text_content}\n\n'
        })

    
    if image_docs:
        content.append({"type": "text", "text": "\n--- Image Excerpts ---\n"})
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"[Image from page {doc.metadata['page']}, Similarity: {doc.metadata['similarity']:.3f}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": f'data:image/png;base64,{image_data_store[image_id]}'
            })
    
    return HumanMessage(content=content)

def multimodal_pipeline_pdf_rag_pipeline(query: str, vector_store: FAISS, image_data_store: Dict[str, str]) -> Dict[str, Any]:
    """Runs the full RAG pipeline and returns the answer and retrieved documents."""
    context_docs = retrieve_multimodal(query, vector_store, image_data_store, k=5)
    message = create_multi_modal_message(query, context_docs, image_data_store)
    response = llm.invoke([message])
    return {
        "answer": response.content,
        "retrieved_docs": context_docs
    }