import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pdfplumber

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None cases

    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, 
        chunk_overlap=500, 
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(text)
    if not text_chunks:
        st.error("No text chunks were generated. Please ensure the document contains valid text.")
        return
    return text_chunks

# Function to extract data from Excel/CSV/TSV files
def get_data_from_file(file):
    try:
        if file.name.endswith(('.csv', '.tsv')):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error(f"Unsupported file format: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error processing file {file.name}: {e}")
        return None

def generate_dataframe_insights(df):
    try:
        # General info and statistics
        insights = []
        insights.append(f"**Number of rows:** {df.shape[0]}")
        insights.append(f"**Number of columns:** {df.shape[1]}")
        insights.append("**Column Names:** " + ", ".join(df.columns))

        # Column-wise statistics (numerical data)
        numeric_cols = df.select_dtypes(include=['number'])
        if not numeric_cols.empty:
            insights.append("**Summary Statistics for Numerical Columns:**")
            insights.append(numeric_cols.describe().to_string())

        # Detect missing values
        missing_values = df.isnull().sum().sum()
        insights.append(f"**Total Missing Values:** {missing_values}")

        return "\n\n".join(insights)
    except Exception as e:
        return f"Error generating insights: {e}"


# Function to create FAISS vector store
def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_chunks = [chunk.replace("\n", " ").strip() for chunk in text_chunks]
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return
    

# Function to load FAISS index and query the data
def get_answer_from_vectorstore(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

    # Build the conversational chain
    prompt_template = """
    You are a knowledgeable data analyst and assistant.
    The software you are running on is developed by Jeff Tumuti.
    Answer the following question based on the provided context.
    If the answer is not in the context, say: "The answer is not available in the provided context.". don't provide the wrong answer\n\n

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Get response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_general_insights(question):
    """
    Generate a general answer for questions not related to the document context.
    """
    # Use Gemini or a similar model to generate a generic response
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = f"""
    You are a financial advisor and assistant. Provide a clear and actionable answer to the following question:
    
    Question: {question}
    Answer:
    """
    response = model.invoke(prompt)
    return response.content

# Main Streamlit application
def main():

    MAX_FILE_SIZE_MB = 5  # Max file size in MB
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

    st.set_page_config(page_title="Financial Insights AI", layout="wide")
    st.title("📊 Financial Insights AI Assistant")

    # Developer Link
    st.markdown("**Developed by [Jeff-Tumuti](https://github.com/jeff-wamugi)**")
    st.markdown("Project Documentation: [Google Docs](https://docs.google.com/document/d/1Ex17OeJcUHN2gJmRUttvJ0nVqVzVFps0gfWx3wp7DPc/edit?usp=sharing)")
    st.write("Upload your financial documents (PDF, Excel, CSV), probe and gain valuable insights!")

    # Sidebar for file upload and instructions
    with st.sidebar:
        st.header("📂 Upload and Process Documents")
        st.info("File Size Limit: 5MB per file")
        uploaded_files = st.file_uploader("Upload PDF/Excel/CSV/TSV files", accept_multiple_files=True, type=["pdf", "csv", "tsv", "xlsx"])
        process_files = st.button("⚙️ Process Documents")

        st.markdown("---")
        st.header("🛠 How to Use")
        st.write("""
        1. Upload financial documents (PDF, Excel, CSV).
        2. Click **Process Documents** to extract insights.
        3. Ask a question related to your documents.
        4. View the AI-generated financial insights.
        """)
        st.info("Supported Formats: PDF, CSV, TSV, Excel")

    # Process uploaded files
    if process_files and uploaded_files:
        all_text = ""
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                # Check file size
                if file.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"The file {file.name} exceeds the size limit of {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
                    continue

                if file.name.endswith(".pdf"):
                    text = get_pdf_text([file])
                    all_text += text + "\n"
                else:
                    df = get_data_from_file(file)
                    if df is not None:
                        all_text += df.to_string() + "\n"  # Add DataFrame as text for FAISS processing

            # Process text into chunks and create vector store
            if all_text:
                text_chunks = get_text_chunks(all_text)
                create_vector_store(text_chunks)
                st.success("✅ Documents successfully processed and vector store created!")
            else:
                st.error("Failed to process uploaded documents. Please check the file formats and size.")

    # Question Answer Section
    st.write("---")
    st.subheader("💬 Ask Questions (Provide as much context as possible)")
    user_question = st.text_input("Enter your question:", key="user_question")
    ask_button = st.button("🔍 Get Insights")

    if ask_button and user_question:
        with st.spinner("Generating insights..."):
            response = get_answer_from_vectorstore(user_question)
            if response:
                if "The answer is not available in the provided context." in response:
                    st.info("The answer is not in the provided document. Here's a general insight instead:")
                    general_response = get_general_insights(user_question)
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f9f9f9; 
                            color: #333; 
                            padding: 10px; 
                            border-radius: 8px; 
                            border: 1px solid #ddd;">
                            {general_response}
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.success("Here’s your answer:")
                    st.write(response)
            else:
                st.warning("No relevant answer found. Try rephrasing your question.")

if __name__ == "__main__":
    main()