"""
The goal of this Program is to create a chatbot that can maintain a conversation, while at the same time
answer some questions over documents

"""

import PyPDF2
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_retriever_tool

openai_api_key = "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"






#initialize the LLM
llm = ChatOpenAI(model_name= "gpt-3.5-turbo",openai_api_key = openai_api_key,temperature=0)

#Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Initialize the TextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 2000,
    chunk_overlap= 20,
)

#initalize the emebeddings
embeddings = OpenAIEmbeddings(openai_api_key =openai_api_key)


#Load the Pdf Document and extract text  (Will add the opportunity for the user to add multiple documents soon)
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

    return text
file_path = 'Ghana_Constitution_1996.pdf'
my_text = extract_text_from_pdf(file_path)


#Splitting the text
docs = text_splitter.create_documents([my_text])


#Embed documents and store it in a vectorDatabase
db = FAISS.from_documents(docs,embeddings)
retriever = db.as_retriever()  #Our Database as a Retriever









#Give a Prompt to the AI to act in a cerain manner
prompt_template = """
Instruction: My name is LexU, and I am a highly respected legal expert specializing in Ghanaian law. My expertise covers various legal areas such as civil law, criminal law, contract law, constitutional law, and administrative law. I am here to provide comprehensive and accurate legal advice based on the documents I have been provided, which are as follows: {context}. I do not consult outside sources.
Your role is to maintain a friendly and conversational tone while providing direct and clear answers to any legal queries. Do not pose questions in your responses. Prioritize clarity, precision, and professionalism. Ensure that the information you provide is up-to-date and in accordance with the legal framework of Ghana. If a question is outside your expertise or not relevant to the documents you have, kindly inform the user in a friendly manner that you are unable to assist with that particular query.
Remember to reference any laws or articles from the documents when providing advice and add a disclaimer at the end of your responses, stating that the advice is based on the documents you have been given.

Question: {question}

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


chain_type_kwargs = {"prompt": PROMPT}
#Chain that is Supposed to use the LLM Correctly
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

print("Hello Welcome to Chat With LexU, how can help you today? (Note to exit, type 'exit' \n ")
DISCLAIMER = "For more information, you can visit www.dennislawgh.com."


#Loop to maintain a convcersation
while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
    else:
        response =qa.run(query)
        print(f"LexU Response:{response}\n\nDISCLAIMER:{DISCLAIMER}")