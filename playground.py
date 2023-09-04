"""
The goal of this Program is to create a chatbot that can maintain a conversation, while at the same time
answer some questions over documents

"""
from langchain.agents import AgentExecutor
import PyPDF2
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, OpenAIMultiFunctionsAgent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.storage import  LocalFileStore
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain,_load_refine_chain
from langchain.chains.summarize import load_summarize_chain
from  langchain.tools import  HumanInputRun
from langchain.docstore.document import Document


from langchain.chains.summarize import load_summarize_chain
openai_api_key = "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"

# initialize the LLM
llm_chatModel = ChatOpenAI( openai_api_key=openai_api_key, temperature=0,streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()])

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the TextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    add_start_index=True,
)
#Good to know for post retrieval info : https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder

# initalize the emebeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

myText = []
# Load the Pdf Document and extract text  (Will add the opportunity for the user to add multiple documents soon)
def extract_text_from_pdf(file_path):
    # Initialize an empty string to store the text
    text = ""

    # Open the PDF file in read-binary mode
    with open(file_path, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # Loop through each page and extract text
        for index in range(num_pages):
            page = pdf_reader.pages[index]
            text += page.extract_text()
            myText.append(index)
    return text



arr = []




file_path = 'Ghana_Constitution_1996.pdf'
my_text = extract_text_from_pdf(file_path)
print(len(myText))

newText = text_splitter.split_text(my_text)
print(len(newText))



# Splitting the text
#docs = text_splitter.create_documents([my_text])


#print(f"I will have {len(docs)} docs instead of one ")
#print(docs[0])
#print(docs[62])

#for i, text in enumerate(docs):
 #   text.metadata["source"] = f"PAge: {i-153}"


#We need to add now some type of cache, so we can use the embeddings without embed the documents again if already
""""
#Set the storage
storage = LocalFileStore("./cache/")
#Set the cached embedder
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,storage,namespace=embeddings.model
)

# Embed documents and store it in a vectorDatabase
db = FAISS.from_documents(docs, cached_embedder)
retriever = db.as_retriever()  # Our Database as a Retriever
"""

"""
The idea is to create a chain that will answer the question and automatically provide the sources 
"""


#THE LLM IN CHARGE OF ANSWERING QUESTIONS
prompt_template = """
        Instrunction: My name is LexU, and I am a highly respected and knowledgeable lawyer renowned for my expertise in the laws and legal processes of Ghana.
            Provide comprehensive and accurate responses to queries related to various legal areas, including civil law, criminal law, contract law, constitutional law, and administrative law.
            Draw upon your deep understanding of legal principles, precedents, and relevant legislation to offer well-reasoned and informed advice.
            Prioritize clarity, precision, and professionalism in your responses, ensuring that the information provided is up-to-date and in accordance with the legal framework of Ghana, reference any laws applied if necessary.
            and cite relevant cases if available.
            I do only know about laws in ghana
            Answer to the question at the end based on this context :

Context: {context}

Question: {question}:"""

""""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm_chatModel, chain_type="stuff", retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)
"""

#qa_chain = RetrievalQA.from_chain_type(llm_chatModel, retriever=retriever, return_source_documents=True)

""""
#Main Part of the code
print("Hello Welcome to Chat With LexU, how can help you today? (Note to exit, type 'exit' \n ")
DISCLAIMER = "For more information, you can visit www.dennislawgh.com."
while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
    else:
        result = qa_chain({"query": query})
        print(result['source_documents'])
        # Execute the QA chain
        #response = agent_executor({"input": query})
        #print(f"LexU Response:{response['output']}\n\nDISCLAIMER:{DISCLAIMER}")
"""