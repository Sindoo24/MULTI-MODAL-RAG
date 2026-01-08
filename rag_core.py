import fitz
import os
import io
import base64
import numpy as np
import torch
import pickle
from functools import lru_cache
from pathlib import Path
import logging
from PIL import Image
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import CLIPModel, CLIPProcessor
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv() 

if "GEMINI_API_KEY" not in os.environ:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file or as an environment variable.")

# Constants for persistence
DATA_DIR = Path("data")
FAISS_DIR = DATA_DIR / "faiss_indices"
IMAGE_DIR = DATA_DIR / "images"

# Create directories if they don't exist
FAISS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Cache CLIP model loading - loads only once per process
@lru_cache(maxsize=1)
def load_clip_model():
    """
    Load CLIP model and processor with caching.
    This function is called only once per application lifecycle.
    
    Returns:
        Tuple of (model, processor)
    """
    try:
        logger.info("Loading CLIP model (cached)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        logger.info("CLIP model loaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise

# Load model and processor (cached)
model, processor = load_clip_model()

# Initialize Gemini LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.0)
    logger.info("Gemini LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini LLM: {e}")
    raise

class CLIPEmbeddings(Embeddings):
    """Custom LangChain Embeddings class for CLIP."""
    def __init__(self, embed_text_func):
        self.embed_text_func = embed_text_func

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text_func(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text_func(text).tolist()

def embed_image(img_data: Any) -> np.ndarray:
    """
    Embeds a PIL Image object using CLIP.
    
    Args:
        img_data: PIL Image object or path to image file
        
    Returns:
        Normalized embedding vector of shape (512,)
        
    Raises:
        ValueError: If image cannot be processed
    """
    try:
        if isinstance(img_data, str):
            image = Image.open(img_data).convert('RGB')
        else:
            image = img_data
            
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error embedding image: {e}")
        raise ValueError(f"Failed to embed image: {e}")

def embed_text(text: str) -> np.ndarray:
    """
    Embeds a text string using CLIP.
    
    Args:
        text: Input text to embed (max 77 tokens)
        
    Returns:
        Normalized embedding vector of shape (512,)
        
    Raises:
        ValueError: If text cannot be processed
    """
    try:
        inputs = processor(text=text, return_tensors='pt', padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            features = model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise ValueError(f"Failed to embed text: {e}")



def process_pdf_and_create_vector_store(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Processes the PDF, extracts text and images, creates embeddings,
    and returns the FAISS vector store and image data store.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Dictionary containing vector_store, image_data_store, and all_docs,
        or None if processing fails
        
    Raises:
        ValueError: If PDF is invalid or cannot be processed
    """
    try:
        logger.info("Starting PDF processing...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"PDF opened successfully. Pages: {len(doc)}")
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        raise ValueError(f"Invalid PDF file: {e}")

    all_docs = []
    all_embeddings = []
    image_data_store = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    try:
        for i, page in enumerate(doc):
            logger.info(f"Processing page {i+1}/{len(doc)}...")
            
            # Extract text
            text = page.get_text()
            if text.strip():
                temp_doc = Document(page_content=text, metadata={"page": i, 'type': 'text'})
                text_chunks = splitter.split_documents([temp_doc])

                for chunk in text_chunks:
                    try:
                        embedding = embed_text(chunk.page_content)
                        all_embeddings.append(embedding)
                        all_docs.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to embed text chunk on page {i}: {e}")
                        continue
            
            # Extract images
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
                    logger.debug(f"Successfully processed image {img_index} on page {i}")
                except Exception as e:
                    logger.warning(f"Error processing image {img_index} on page {i}: {e}")
                    continue
    finally:
        doc.close()
        logger.info("PDF document closed")

    if not all_embeddings:
        logger.warning("No embeddings created from PDF")
        return None 

    try:
        embeddings_array = np.array(all_embeddings)
        clip_embeddings_instance = CLIPEmbeddings(embed_text)

        vector_store = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
            embedding=clip_embeddings_instance,
            metadatas=[doc.metadata for doc in all_docs]
        )
        
        logger.info(f"Vector store created with {len(all_docs)} documents")
        
        return {
            "vector_store": vector_store,
            "image_data_store": image_data_store,
            "all_docs": all_docs 
        }
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise ValueError(f"Vector store creation failed: {e}")


def save_vector_store(vector_store: FAISS, image_data_store: Dict[str, str], document_id: str) -> None:
    """
    Save FAISS vector store and image data to disk for persistence.
    
    Args:
        vector_store: FAISS vector store to save
        image_data_store: Dictionary of image_id -> base64 encoded images
        document_id: Unique identifier for the document
    """
    try:
        # Save FAISS index
        faiss_path = FAISS_DIR / document_id
        vector_store.save_local(str(faiss_path))
        logger.info(f"FAISS index saved to {faiss_path}")
        
        # Save image data store
        image_store_path = IMAGE_DIR / f"{document_id}.pkl"
        with open(image_store_path, 'wb') as f:
            pickle.dump(image_data_store, f)
        logger.info(f"Image data saved to {image_store_path}")
        
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")
        raise ValueError(f"Vector store save failed: {e}")


def load_vector_store(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Load FAISS vector store and image data from disk.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        Dictionary containing vector_store and image_data_store, or None if not found
    """
    try:
        faiss_path = FAISS_DIR / document_id
        image_store_path = IMAGE_DIR / f"{document_id}.pkl"
        
        # Check if files exist
        if not faiss_path.exists() or not image_store_path.exists():
            logger.warning(f"Vector store not found for document_id: {document_id}")
            return None
        
        # Load FAISS index
        clip_embeddings_instance = CLIPEmbeddings(embed_text)
        vector_store = FAISS.load_local(
            str(faiss_path),
            embeddings=clip_embeddings_instance,
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded from {faiss_path}")
        
        # Load image data store
        with open(image_store_path, 'rb') as f:
            image_data_store = pickle.load(f)
        logger.info(f"Image data loaded from {image_store_path}")
        
        return {
            "vector_store": vector_store,
            "image_data_store": image_data_store
        }
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


def list_saved_documents() -> List[str]:
    """
    List all saved document IDs.
    
    Returns:
        List of document IDs that have been saved
    """
    try:
        saved_docs = [d.name for d in FAISS_DIR.iterdir() if d.is_dir()]
        logger.info(f"Found {len(saved_docs)} saved documents")
        return saved_docs
    except Exception as e:
        logger.error(f"Failed to list saved documents: {e}")
        return []


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
                # FIX: Properly decode and embed the image
                image_id = doc.metadata.get('image_id')
                image_base64 = image_data_store.get(image_id)
                if image_base64:
                    try:
                        image_bytes = base64.b64decode(image_base64)
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        retrieved_embeddings.append(embed_image(pil_image))
                    except Exception as e:
                        print(f"Error embedding image {image_id}: {e}")
                        # Fallback: use text embedding of placeholder
                        retrieved_embeddings.append(embed_text(doc.page_content))
                else:
                    # Fallback if image not found
                    retrieved_embeddings.append(embed_text(doc.page_content))

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def _call_llm_with_retry(message: HumanMessage) -> str:
    """Call Gemini LLM with automatic retry on failures."""
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise

def multimodal_pipeline_pdf_rag_pipeline(query: str, vector_store: FAISS, image_data_store: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline and returns the answer and retrieved documents.
    
    Args:
        query: User question
        vector_store: FAISS vector store
        image_data_store: Dictionary of image_id -> base64 encoded images
        
    Returns:
        Dictionary with 'answer' and 'retrieved_docs'
        
    Raises:
        ValueError: If query processing fails
    """
    try:
        logger.info(f"Processing query: {query[:50]}...")
        context_docs = retrieve_multimodal(query, vector_store, image_data_store, k=5)
        logger.info(f"Retrieved {len(context_docs)} context documents")
        
        message = create_multi_modal_message(query, context_docs, image_data_store)
        answer = _call_llm_with_retry(message)
        
        logger.info("Query processed successfully")
        return {
            "answer": answer,
            "retrieved_docs": context_docs
        }
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise ValueError(f"Failed to process query: {e}")