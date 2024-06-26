#pylint: skip-file
import sqlite3
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
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join("/workspaces/Python-project", 'db.sqlite3'),
#     }
# }
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
        default_index=0,
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
                        
                        #tokens,embedding_cost = calculate_embedding_cost(chunks)
                        #st.write(f"Embedding Cost is Rs{embedding_cost:.4f}")
                        
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
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;">UCS751 Project - Revision Module</p>' 
        st.markdown(new_title, unsafe_allow_html=True)

        st.image("cover.jpg",caption="Cover image is used for educational purposes only and not for monetary gains")
        st.write('''Welcome to the home of Revision Module (using openai and langchain), your gateway to a revolutionary learning experience! Designed under the expert guidance of Dr. Sanjeev Rao as part of the UCS751 course, this app promises to enhance the way you approach education.
            ''')
        st.write('''At Revision Module, we are on a mission to make learning more accessible, engaging, and tailored to your unique needs. We believe that education should be a lifelong journey, and it should adapt to your pace and preferences. Dr. Sanjeev Rao, a renowned educator in UCS751, has helped shape this vision into a reality. ''')
        st.write("Imagine you have a lengthy PDF document packed with valuable information, but you don't have the time or patience to read through the entire document. This is where our app comes into play. ")
        st.write('''
                 Benefits of our app are:

            1.Reading through lengthy documents can be time-consuming and more frustrating when exams are much nearer than imagined. With the app, you can swiftly obtain crucial information without having to sift through the entire document.

            2.Researchers, students, and professionals can significantly expedite their work by quickly finding relevant data within documents.

            3.Users have the flexibility to input their own questions and answers, tailoring the system to their specific needs.They can even get summary of entire document.

                 ''')
        st.write('''
                 Thank you for visiting Revision Module. Together with Dr. Sanjeev Rao and the UCS751 community, we're here to revolutionize your learning experience.
                 ''')
    if selected =="About":
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;">About</p>' 
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("Welcome to the About page of project. Our app is a labor of love brought to you by a dedicated team of talented individuals, each contributing their unique skills and expertise to create a transformative learning experience. Let's meet the minds that made this possible:")
        name1 = '<p style="font-family:Times-New-Roman; text-align: left; color:WHITE; font-size: 20px;"> Ishanpreet Singh</p>' 
        st.markdown(name1, unsafe_allow_html=True)
        st.markdown("""
                        - Roll Number: 102003641
                        - Role: Lead Developer
                    """)
        name2 = '<p style="font-family:Times-New-Roman; text-align: left; color:WHITE; font-size: 20px;"> Rohit Singla</p>' 
        st.markdown(name2, unsafe_allow_html=True)
        st.markdown("""
                        - Roll Number: 102003254
                        - Role: Developer
                    """)
        name3 = '<p style="font-family:Times-New-Roman; text-align: left; color:WHITE; font-size: 20px;"> Tushar Gupta</p>' 
        st.markdown(name3, unsafe_allow_html=True)
        st.markdown("""
                        - Roll Number: 102003252
                        - Role: UX Designer
                    """)
    if selected == "Summarizer":
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 42px;"> Document Summarizer </p>' 
        st.markdown(new_title, unsafe_allow_html=True) 
        new_title = '<p style="font-family:Times-New-Roman; text-align: center; color:WHITE; font-size: 19px;"> (Please upload document with max 1 page if you dont have gpt+ subscription as rate and size of data processed is limited to certain amount in free version and the app will crash. ) </p>' 
        st.markdown(new_title, unsafe_allow_html=True)
        st.image("business-5338474_640.jpg",caption="Cover image is taken from copyright free website pixabay and is posted by Altmann/geralt")
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
            # st.write(file)
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

            
                