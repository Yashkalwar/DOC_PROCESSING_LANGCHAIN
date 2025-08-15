from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Create directory for extracted images
os.makedirs('extracted_images', exist_ok=True)

load_dotenv()

pdf_path = "bray_sample.pdf"
documents = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        for table in page.extract_tables():
            table_str = "\n".join([
                ", ".join([str(cell) if cell is not None else "" for cell in row])
                for row in table if row
            ])
            if table_str.strip():
                documents.append(Document(
                    page_content=f"Extracted table from PDF:\n{table_str}",
                    metadata={"source": pdf_path, "type": "table", "page": page_num}
                ))

doc = fitz.open(pdf_path)
for page_num, page in enumerate(doc):
    # Get regular text
    regular_text = page.get_text().strip()

    # Extract images from the page
    images = page.get_images(full=True)
    print(f"Page {page_num + 1}: Found {len(images)} images")
    
    image_paths = []
    for img_index, img in enumerate(images, 1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        
        # Save the image
        img_filename = os.path.join('extracted_images', f"page_{page_num+1}_img_{img_index}.{base_image['ext']}")
        print(f"  Saving image to: {img_filename}")
        with open(img_filename, "wb") as img_file:
            img_file.write(base_image["image"])
        image_paths.append(img_filename)
    
    # Add image references to the page text
    if image_paths:
        regular_text += "\n\n[Images on this page: " + ", ".join(image_paths) + "]"

    if regular_text.strip():
        documents.append(Document(
            page_content=regular_text,
            metadata={"source": pdf_path, "page": page_num, "images": len(images)}
        ))

def save_extracted_content(documents: List[Document], output_file: str = 'extracted_content.json') -> None:
    """Save extracted content to a JSON file in the specified format."""
    # Initialize the output structure
    output = {
        "document_id": os.path.splitext(os.path.basename(pdf_path))[0],
        "metadata": {
            "source": pdf_path,
            "page_count": len(doc) if 'doc' in locals() else 0,
            "extraction_timestamp": datetime.now().isoformat()
        },
        "pages": []
    }
    
    # Group documents by page
    pages_dict: Dict[int, Dict[str, Any]] = {}
    
    for doc in documents:
        page_num = doc.metadata.get('page', 0) + 1  # Convert to 1-based
        
        if page_num not in pages_dict:
            pages_dict[page_num] = {
                "page_number": page_num,
                "text": "",
                "tables": [],
                "images": []
            }
        
        # Handle different content types
        if doc.metadata.get('type') == 'table':
            # Extract table data (assuming doc.page_content contains the table)
            table_data = [row.split(', ') for row in doc.page_content.split('\n') if row]
            table_id = f"t{len(pages_dict[page_num]['tables']) + 1}"
            pages_dict[page_num]['tables'].append({
                "table_id": table_id,
                "data": table_data
            })
        else:
            # Handle text and images
            if doc.metadata.get('images', 0) > 0:
                # Add image references
                for img_num in range(1, doc.metadata['images'] + 1):
                    img_id = f"i{img_num}"
                    img_path = os.path.join('extracted_images', f"page_{page_num}_img_{img_num}.png")
                    pages_dict[page_num]['images'].append({
                        "image_id": img_id,
                        "path": img_path,
                        "caption": f"Image from page {page_num}"
                    })
            
            # Add text content
            if doc.page_content.strip():
                if pages_dict[page_num]['text']:
                    pages_dict[page_num]['text'] += "\n\n" + doc.page_content
                else:
                    pages_dict[page_num]['text'] = doc.page_content
    
    # Convert the pages dictionary to a list and sort by page number
    output['pages'] = [pages_dict[page_num] for page_num in sorted(pages_dict.keys())]
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted content saved to {output_file}")

# Save extracted content
save_extracted_content(documents)

# Create chunks after combining all documents
chunks = text_splitter.split_documents(documents)
print(f"Total final chunks: {len(chunks)}")

# Create and save embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("bray_faiss_index")

# Load the vector store
vectorstore = FAISS.load_local("bray_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize QA chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

# Example query
query = "what are the different tags for valve identification?"
result = qa.invoke({"query": query})
print(f"\nQuery: {query}")
print(f"Answer: {result['result']}")