"""
Simple script to create a test PDF using PyMuPDF
"""
import fitz
from PIL import Image, ImageDraw
import io

# Create a new PDF
doc = fitz.open()

# Page 1 - Introduction
page1 = doc.new_page(width=595, height=842)  # A4 size
text1 = """
MULTIMODAL RAG TEST DOCUMENT

This document is designed to test the RAG system capabilities.

Key Topics:
1. Machine Learning and AI
2. Natural Language Processing  
3. Computer Vision

The main focus is on evaluating multimodal retrieval systems that can
understand both text and visual content. This is crucial for modern
information retrieval applications.
"""
page1.insert_text((50, 50), text1, fontsize=12)

# Add a simple chart image
img = Image.new('RGB', (300, 200), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([20, 20, 280, 180], outline='blue', width=3)
draw.text((80, 85), 'Performance Chart', fill='black')
draw.line([40, 100, 260, 100], fill='red', width=2)
draw.line([40, 120, 200, 80], fill='green', width=2)

img_buffer = io.BytesIO()
img.save(img_buffer, format='PNG')
img_buffer.seek(0)
page1.insert_image(fitz.Rect(50, 300, 350, 500), stream=img_buffer.getvalue())

# Page 2 - Methodology
page2 = doc.new_page(width=595, height=842)
text2 = """
METHODOLOGY

Our approach uses CLIP embeddings for multimodal understanding.
The system combines text and image processing for comprehensive retrieval.

Key Components:
- CLIP model for generating embeddings
- FAISS for efficient vector storage and similarity search
- Gemini LLM for natural language generation
- PyMuPDF for document parsing

The retrieval process involves:
1. Extracting text and images from PDFs
2. Creating embeddings for both modalities
3. Storing in a vector database
4. Retrieving relevant context for queries
5. Generating answers using retrieved context
"""
page2.insert_text((50, 50), text2, fontsize=12)

# Page 3 - Results and Conclusions
page3 = doc.new_page(width=595, height=842)
text3 = """
RESULTS AND CONCLUSIONS

The multimodal RAG system shows promising results across various metrics.

Key Findings:
- High retrieval accuracy for both text and images
- Effective context utilization in answer generation
- Fast query response times (< 3 seconds)
- Good faithfulness to source material

Future Work:
- Improve image understanding capabilities
- Add support for tables and structured data
- Optimize for larger document collections
- Implement advanced ranking algorithms

Conclusion:
The system effectively retrieves both text and visual information,
making it suitable for complex document understanding tasks.
"""
page3.insert_text((50, 50), text3, fontsize=12)

# Save the PDF
doc.save("test_document.pdf")
doc.close()

print("✅ test_document.pdf created successfully with 3 pages")
