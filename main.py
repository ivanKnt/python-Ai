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
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

openai_api_key = "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"

#initialize the LLM
llm = ChatOpenAI(model_name= "gpt-3.5-turbo",openai_api_key = openai_api_key,temperature=0)
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







#PromptTemplate stating the behavior of the llm
prompt_template = """
        Instrunction: My name is LexU, and I am a highly respected and knowledgeable lawyer renowned for my expertise in the laws and legal processes of Ghana.
            Provide comprehensive and accurate responses to queries related to various legal areas, including civil law, criminal law, contract law, constitutional law, and administrative law.
            Draw upon your deep understanding of legal principles, precedents, and relevant legislation to offer well-reasoned and informed advice.
            Prioritize clarity, precision, and professionalism in your responses, ensuring that the information provided is up-to-date and in accordance with the legal framework of Ghana, reference any laws applied if necessary.
            and cite relevant cases if available.
            Answer to the question at the end based on this context :
            
Context: {context}
            
Question: {question}:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context","question"]
)
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

tools = [Tool(
    name="legal_expert_ghana",
    func=qa.run,
    description="Useful when asked about questions related to the legal Environment and about you"
)]
"""""
tool = create_retriever_tool(
    retriever,
    name= "Legal Assistant",
    description="Useful when asked about questions related to the legal Environment"
)
"""
#tools = [tool]
agent = initialize_agent(tools,llm,agent=AgentType.OPENAI_FUNCTIONS,memory=memory,verbose=True)


print("Hello Welcome to Chat With LexU, how can help you today? (Note to exit, type 'exit' \n ")
DISCLAIMER = "For more information, you can visit www.dennislawgh.com."
while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
    else:
        # Execute the QA chain
        response = agent.run(input = query)
        print(f"LexU Response:{response}\n\nDISCLAIMER:{DISCLAIMER}")

 