from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract, io
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Create directory for extracted images
os.makedirs('extracted_images', exist_ok=True)

load_dotenv()

pdf_path = r"D:\ANAND\BRAY_LANGCHAIN\EN_TB-1005_Act-SelectGuide-Ball-Valves.pdf"
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

print

doc = fitz.open(pdf_path)
for page_num, page in enumerate(doc):
    # Get regular text
    regular_text = page.get_text().strip()

    # OCR any images on the page
    image_texts = []
    images = page.get_images(full=True)
    print(f"Page {page_num + 1}: Found {len(images)} images")
    
    for img_index, img in enumerate(images, 1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Save the image
        img_filename = os.path.join('extracted_images', f"page_{page_num+1}_img_{img_index}.{base_image['ext']}")
        print(f"  Saving image to: {img_filename}")
        with open(img_filename, "wb") as img_file:
            img_file.write(image_bytes)
        
        # OCR the image
        image = Image.open(io.BytesIO(image_bytes))
        image_text = pytesseract.image_to_string(image).strip()
        if image_text:
            image_texts.append(f"[Image {img_index}]: {image_text}")
            image_texts.append(f"[Image saved as: {img_filename}]")

    # Combine text and OCR
    page_text = regular_text
    if image_texts:
        page_text += "\n\n[Extracted from images on page:]\n" + "\n".join(image_texts)

    if page_text.strip():
        documents.append(Document(
            page_content=page_text,
            metadata={"source": pdf_path, "page": page_num}
        ))



# Create chunks after combining all documents
chunks = text_splitter.split_documents(documents)
print(f"Total final chunks: {len(chunks)}")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("bray_faiss_index")

vectorstore = FAISS.load_local("bray_faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = OpenAI(temperature=0)  # or swap for free local LLM
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
query = "What actuator is recommended for 500 in-lbs torque?"
result = qa.invoke({"query": query})
print("Q:", query)
print("A:", result)
sources = result['source_documents']

# print("Sources:")
# for doc in sources:
#     print(f"- Page: {doc.metadata.get('page', 'unknown')}, Source: {doc.metadata.get('source', 'unknown')}")
#     print(f"  Text snippet: {doc.page_content[:200]}...\n")