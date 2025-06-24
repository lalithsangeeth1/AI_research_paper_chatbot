import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')

# Sidebar for file uploading
st.sidebar.title("Upload Research Papers 📚")
st.sidebar.markdown("📎 **Accepted formats**: `.pdf`, `.txt` only.")
uploaded_files = st.sidebar.file_uploader(
    "Upload your research documents", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

# Optimized document processing function
def process_uploaded_files(uploaded_files):
    documents = []
    with st.spinner("Processing the uploaded papers... 🧐"):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f) for f in uploaded_files]
            for future in futures:
                docs = future.result()
                if docs:
                    documents.extend(docs)
    return documents

# Handle PDF and TXT files
def process_single_file(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            return loader.load()
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
            return loader.load()
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    finally:
        os.unlink(tmp_path)
    return []

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'challenge_answers' not in st.session_state:
    st.session_state.challenge_answers = {}

# Main logic after file upload
if uploaded_files:
    documents = process_uploaded_files(uploaded_files)

    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Generate summary
        with st.spinner("Generating document summary... 📄"):
            from langchain.chains.summarize import load_summarize_chain
            summarize_chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                combine_prompt=PromptTemplate(
                    template="Combine these summaries into one concise summary (≤150 words):\n\n{text}",
                    input_variables=["text"]
                )
            )
            raw_summary = summarize_chain.run(chunks[:10])
            
            # Trim summary to 150 words
            summary_words = raw_summary.split()
            trimmed_summary = " ".join(summary_words[:150])
            if len(summary_words) > 150:
                trimmed_summary += "..."

        st.subheader("📝 Auto Summary (≤ 150 words)")
        st.info(trimmed_summary)

        # Create vector store and QA chain
        embeddings = HuggingFaceEmbeddings()
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Custom prompt with justification requirement
        template = """You are a research assistant. Answer the question using ONLY the provided context. 
        If the answer isn't in the context, say "This information is not in the document."
        ALWAYS provide a justification including the section or page reference.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

# Main UI
st.title("Research Paper Assistant 🤖")
st.write("Upload documents in the sidebar to get started!")

# Interaction Tabs
tab1, tab2 = st.tabs(["💬 Ask Anything", "🎯 Challenge Me"])

with tab1:
    st.subheader("Question Answering Mode")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                with st.expander("Source Reference"):
                    st.text(message["source"])
    
    # Question input
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.qa_chain:
            st.error("Please upload documents first!")
            st.stop()
            
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                
                # Get source references
                sources = []
                for doc in result["source_documents"]:
                    source_info = f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}..."
                    sources.append(source_info)
                
                # Format response with sources
                full_response = f"{answer}\n\n**Sources:**\n" + "\n".join(
                    [f"- {src}" for src in sources[:3]]  # Show top 3 sources
                )
            except Exception as e:
                full_response = f"Error: {str(e)}"
                sources = []
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "source": "\n".join(sources)
        })
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(full_response)
            if sources:
                with st.expander("View Sources"):
                    st.text("\n".join(sources))

with tab2:
    st.subheader("Comprehension Challenge")
    
    if st.button("Generate New Questions"):
        if not st.session_state.vector_store:
            st.error("Please upload documents first!")
            st.stop()
            
        with st.spinner("Creating challenge questions..."):
            # Get document text
            doc_text = "\n\n".join([doc.page_content for doc in st.session_state.vector_store.docstore._dict.values()][:3])
            
            # Generate questions prompt
            prompt = f"""Generate exactly 3 logic-based comprehension questions from this research document.
            For each question, include:
            1. The question
            2. The expected answer
            3. The exact text reference from the document
            Format as:
            Q1: [question]
            A1: [expected answer]
            R1: [reference text]
            ---
            Document excerpt:
            {doc_text[:3000]}"""
            
            response = llm.invoke(prompt)
            questions = []
            
            # Parse response
            try:
                parts = response.content.split("---")
                for part in parts:
                    if "Q1:" in part:
                        q_lines = part.split("\n")
                        questions = [
                            {
                                "question": q_lines[0].replace("Q1:", "").strip(),
                                "answer": q_lines[1].replace("A1:", "").strip(),
                                "reference": q_lines[2].replace("R1:", "").strip()
                            },
                            {
                                "question": q_lines[3].replace("Q2:", "").strip(),
                                "answer": q_lines[4].replace("A2:", "").strip(),
                                "reference": q_lines[5].replace("R2:", "").strip()
                            },
                            {
                                "question": q_lines[6].replace("Q3:", "").strip(),
                                "answer": q_lines[7].replace("A3:", "").strip(),
                                "reference": q_lines[8].replace("R3:", "").strip()
                            }
                        ]
                st.session_state.challenge_questions = questions
                st.session_state.challenge_answers = {}
                st.success("Challenge questions generated!")
            except:
                st.error("Failed to parse questions. Please try again.")
                st.code(response.content)
    
    if st.session_state.challenge_questions:
        st.divider()
        st.markdown("### Answer these comprehension questions:")
        
        for i, question in enumerate(st.session_state.challenge_questions):
            st.markdown(f"**Question {i+1}**: {question['question']}")
            
            # Text input for answer
            user_answer = st.text_input(
                f"Your answer for question {i+1}:",
                value=st.session_state.challenge_answers.get(i, ""),
                key=f"challenge_{i}"
            )
            st.session_state.challenge_answers[i] = user_answer
            
            # Evaluate answer
            if st.button(f"Evaluate Answer {i+1}", key=f"eval_{i}"):
                with st.spinner("Evaluating..."):
                    prompt = f"""Evaluate if the user's answer matches the expected answer. 
                    Be strict but fair. Provide:
                    - Correctness (Correct/Partially Correct/Incorrect)
                    - Explanation
                    - Show the reference from the document
                    
                    Question: {question['question']}
                    Expected Answer: {question['answer']}
                    Document Reference: {question['reference']}
                    User's Answer: {user_answer}"""
                    
                    evaluation = llm.invoke(prompt)
                    st.markdown(f"**Evaluation**:")
                    st.info(evaluation.content)
        
        st.divider()