#pylint: skip-file
import streamlit as st
import cv2
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from forex_python.converter import CurrencyRates
from langchain.vectorstores import Chroma
from streamlit_option_menu import option_menu
from st_btn_select import st_btn_select
import os
from PIL import Image
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import PyPDF2
import docx2txt
import tempfile
import time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
cr = CurrencyRates()
def load_document(file):
    import os
    name ,extention = os.path.splitext(file)
    
    if extention == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print("Loading {file}")
        loader = PyPDFLoader(file)
    elif extention ==".docx":
        from langchain.document_loaders import Docx2txtLoader
        print("Loading {file}")
        loader = Docx2txtLoader(file)
    elif extention == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print("document not supported")
        return None
    data = loader.load()
    return data

def chunk_data(data ,chunk_size = 256 , chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap = chunk_overlap) 
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks , embeddings)
    return vector_store

def ask_and_get_answers(vector_store ,q , k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-3.5-turbo" , temperature = 1)
    retriever = vector_store.as_retriever(search_type = "similarity" , search_kwargs = {"k":k})
    chain = RetrievalQA.from_chain_type(llm =llm , chain_type="stuff" ,retriever = retriever)
    
    answer = chain.run(q)
    return answer
    
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    cost = total_tokens/1000 * 0.004
    inr_cost = cr.convert("USD","INR",cost)
    return total_tokens , inr_cost


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]
        
        
def change_file_to_txt(file):
    import os
    name ,extention = os.path.splitext(file)
    
    if extention == ".pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''

        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            text +=page.extract_text()
    elif extention ==".docx":
        print("Loading {file}")
        text = docx2txt.process(file)
    elif extention == ".txt":
        with open(file=file , encoding="utf-8") as f:
            text = f.read()
    else:
        print("document not supported")
        return None
    return text    

def turn_txt_to_langchain_doc(text):
    docs = [Document(page_content=text)]
    return docs

def stuffing_gpt(file):
                x = change_file_to_txt(file=file)
                docs = turn_txt_to_langchain_doc(x)
                return docs

if __name__ == "__main__":
    # pic2 = "/workspaces/Python-project/front_end_app/Thapar_Institute_of_Engineering_and_Technology_University_logo.ico"
    # img2 = cv2.imread(pic2,1)
    # image2 = np.array([img2])
    im = Image.open("Thapar_Institute_of_Engineering_and_Technology_University_logo.ico")
    st.set_page_config(page_title="UCS751",page_icon=im, layout="centered" , initial_sidebar_state="collapsed")
    # option = st_btn_select(('option1', 'option2', 'option3', 'option4'), index=2)
    # with st.sidebar:
        
    selected = option_menu(
        menu_title = "Page Menu",
        options= ["Home" ,"QnA App" , "Summarizer", "About"],
        icons=["house","question","book","people-fill"],
        menu_icon="cast",
        default_index=2,
        orientation="horizontal"
    )
    if selected =="QnA App":

        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;"> Revision module for exams</p>' 
        st.markdown(new_title, unsafe_allow_html=True)
        pic1 = "ai-generated-7737009_1280.jpg"
        img1 = cv2.imread(pic1, 1)
        image1 = np.array([img1])
        st.image(image=image1,channels = "RGB" , caption = "Cover pic:  Image is submitted to creative commons by Alexandra_Koch on pixabay.com")
        
        
        with st.sidebar:
            api_key = st.text_input("OpenAI API Key: " , type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                
                uploaded_file = st.file_uploader("Upload a file for revision", type=["pdf","docx","txt"])
                chunk_size = st.number_input("Chunk size: " , min_value=100 , max_value=2048 ,value=512 , on_change=clear_history)
                k = st.number_input("k: " , min_value=1 , max_value=20 ,value=3, on_change=clear_history)
                add_data = st.button("Add data" , on_click=clear_history)
                
                
                if uploaded_file and add_data:
                    with st.spinner("Reading , Chunking and EMbedding file...."):
                        bytes_data = uploaded_file.read()
                        filename = os.path.join("./",uploaded_file.name)
                        with open(filename,"wb")as f:
                            f.write(bytes_data)
                            
                        data = load_document(filename)
                        chunks = chunk_data(data , chunk_size=chunk_size) 
                        st.write(f"chunk size: {chunk_size} , chunks: {len(chunks)} ")
                        
                        tokens,embedding_cost = calculate_embedding_cost(chunks)
                        st.write(f"Embedding Cost is Rs{embedding_cost:.4f}")
                        
                        vector_store = create_embeddings(chunks)
                        st.session_state.vs = vector_store
                        st.success("File uploaded , Chunked and Embedded succesfully")
            else:
                st.error("Please enter api key")    
        if api_key:
            q = st.text_input("Ask a question about the file for revision")
            if q:
                if "vs" in st.session_state:
                    vector_store = st.session_state.vs
                    # st.write(f"k:{k}")
                    answer = ask_and_get_answers(vector_store=vector_store ,q=q,k=k )
                    st.text_area("LLM Answer:", value=answer)
                                                        
                    
                    st.divider()
                    
                    if "history" not in st.session_state:
                        st.session_state.history = ""
                    value = f"Q: {q} \nA: {answer}"
                    st.session_state.history = f"{value} \n {'-' * 100} \n {st.session_state.history}"
                    h = st.session_state.history
                    st.text_area(label="Chat History" , value = h , key= "history" , height=400 )
        else:
            q = st.text_input("Ask a question about the file for revision")
            st.warning("Please click arrow button on the top right corner to open sidebar and enter openai api key")
                
    if selected =="Home":
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;"> Welcome to UCS751 Group Project</p>' 
        st.markdown(new_title, unsafe_allow_html=True)

        
        st.write('''In our rapidly evolving digital age, information is more accessible than ever before. 
                     However, the sheer volume of data can be overwhelming, making it challenging to locate and extract valuable insights from sources like PDF documents.
                     The need for efficient knowledge extraction and summarization has led to the development of innovative solutions, one of which is the "PDF Q&A Solver." 
                     This groundbreaking project combines the power of question-answering technology and document summarization to streamline information retrieval and enhance our understanding of complex documents.
            ''')
        st.write('''The PDF Q&A Solver: Bridging the Gap

Imagine you have a lengthy PDF document packed with valuable information, but you don't have the time or patience to read 
through the entire document. This is where the PDF Q&A Solver comes into play. This project combines the following 
components to deliver a revolutionary information retrieval system:

1.Document Parsing: The first step in this process involves the extraction of data from the PDF document. The system parses
the text and understands its structure, converting the contents into a machine-readable format.

2.Question and Answer Repository: Users can provide a set of questions and answers related to the document's content. These
serve as the basis for the system to understand what information is relevant and what needs to be summarized.

3.Summarization: After identifying relevant sections, the system summarizes the content based on the questions and answers 
provided. This summarization process condenses the document into key points, allowing users to quickly access essential information.


 ''')
        st.write('''
                 Benefits of the PDF Q&A Solver:

            1.Time-Saving: Reading through lengthy documents can be time-consuming. With the PDF Q&A Solver, you can swiftly obtain crucial information without having to sift through the entire document.

            2.Enhanced Understanding: The system doesn't just extract information; it also helps you understand how the questions relate to the answers, providing context and clarity.

            3.Efficient Research: Researchers, students, and professionals can significantly expedite their work by quickly finding relevant data within documents.

            4.Customizable: Users have the flexibility to input their own questions and answers, tailoring the system to their specific needs.

                 ''')
        st.write('''
                 Conclusion:

The PDF Q&A Solver represents a significant leap forward in the quest for efficient knowledge extraction from PDF documents. It combines the power of machine learning, NLP, and summarization techniques to
revolutionize the way we interact with information, making it more accessible, comprehensible, and time-efficient. In an era where information is power, tools like the PDF Q&A Solver empower individuals and 
organizations to harness the full potential of the knowledge at their disposal. This innovative project exemplifies the ongoing evolution of technology in making the world's vast store of information more manageable and accessible to all.
                 ''')
    if selected =="About":
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;"> Welcome to UCS751 Group Project</p>' 
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("UCS751 Simulation and Modeling is an advanced academic course that delves into the intricate world of replicating and analyzing real-world systems and processes through mathematical and computational models. Students explore various simulation techniques, including Monte Carlo, discrete event simulations, and agent-based modeling, while gaining hands-on experience in model development, data collection, and optimization. With applications spanning engineering, economics, healthcare, environmental science, and beyond, Simulation and Modeling equips students with the versatile skills needed to address complex challenges and inform data-driven decisions in an ever-evolving, technology-driven world.")
    
    if selected == "Summarizer":
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;"> Document Summarizer </p>' 
        st.markdown(new_title, unsafe_allow_html=True) 
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 19px;"> (Please upload document with max 1 page if you dont have gpt+ subscription as rate and size of data processed is limited to certain amount in free version and the app will crash. ) </p>' 
        st.markdown(new_title, unsafe_allow_html=True)
        st.image("business-5338474_640.jpg",caption="Cover image is taken from copyright free website pixabay and is posted by Altmann/geralt",)
        with st.sidebar:
            api_key = st.text_input("OpenAI API Key: " , type="password")
            uploaded_file = st.file_uploader("Upload a file for summarization", type=["pdf","docx","txt"])
            # stuff_button = st.button("Load chatgpt")
            # if uploaded_file and stuff_button:
              
            # text =  change_file_to_txt(file)
            # docs = turn_txt_to_langchain_doc(text)
                
                
                
                # if uploaded_file and stuff_button:
                #     with st.spinner("Reading , converting and Loading file to chatgpt"):
                #             filename = os.path.join("./",uploaded_file.name)
                #             text = change_file_to_txt(filename)
                #             docs = turn_txt_to_langchain_doc(text)
                #             time.sleep(2)
                #             st.success("File loaded successfully")
                     
                                
        if api_key and uploaded_file:
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(temperature=0, model_name = "gpt-3.5-turbo")  
            temp_dir = tempfile.mkdtemp()
            file = os.path.join(temp_dir, uploaded_file.name)
            st.write(file)
            with open(file, "wb") as f:
                f.write(uploaded_file.getvalue())      
            filename = os.path.join("./",uploaded_file.name)
            text = change_file_to_txt(file)
            docs = turn_txt_to_langchain_doc(text)
            # st.write(docs)   
            def prompting_gpt(text = text):
                template = '''Write a summary of the folowing text. TEXT: {text}  '''
                prompt = PromptTemplate(
                input_variables=["text"],
                template=template)
                return prompt
            
            st.write("Click the button to summarize the loaded document")
            summarize_button = st.button("summarize")
            
            if summarize_button:
                prompt = prompting_gpt(text = text)
            
                def summarization_chains(llm=llm ,prompt = prompt , docs = docs):
                    chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt =prompt,
                    verbose=False)
                    summary = chain.run(docs)
                    return summary
                
                summary = summarization_chains(llm=llm, prompt=prompt,docs = docs)
                
                st.text_area(value= summary , label = "summary",height=500)
                
        else:
            st.warning("Please fill the details in sidebar by clicking in leftmost arrow")            

            
                