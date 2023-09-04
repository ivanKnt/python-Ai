"""
The goal of this Program is to create a chatbot that can maintain a conversation, while at the same time
answer some questions over documents

"""
from langchain.agents import AgentExecutor
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
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

openai_api_key = "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"

# initialize the LLM
llm = ChatOpenAI( openai_api_key=openai_api_key, temperature=0)

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the TextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
)

# initalize the emebeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


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

    return text


file_path = 'Ghana_Constitution_1996.pdf'
my_text = extract_text_from_pdf(file_path)

# Splitting the text
docs = text_splitter.create_documents([my_text])



# Embed documents and store it in a vectorDatabase
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()  # Our Database as a Retriever


#Define Tool for AI to act as a Legal ASSITANT
tool = create_retriever_tool(
    retriever,
    name="Legal_Assistant",
    description="Useful when asked about questions related to the legal Environment"
)

#Set the tools used by the LLM
tools = [tool]

# This is needed for both the memory and the prompt (need to consult the documentation about this)
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

system_message = SystemMessage(
    content=("""Instruction: My name is LexU, and I am a highly respected legal expert specializing in Ghanaian law. My expertise covers various legal areas such as civil law, criminal law, contract law, constitutional law, and administrative law. I am here to provide comprehensive and accurate legal advice based on the documents I have been provided. I do not consult outside sources.

Your role is to maintain a friendly and conversational tone while providing direct and clear answers to any legal queries. Do not pose questions in your responses. Prioritize clarity, precision, and professionalism. Ensure that the information you provide is up-to-date and in accordance with the legal framework of Ghana. If a question is outside your expertise or not relevant to the documents you have, kindly inform the user in a friendly manner that you are unable to assist with that particular query.

Remember to reference any laws or articles from the documents when providing advice and add a disclaimer at the end of your responses, stating that the advice is based on the documents you have been given.

Use any tools if necessary and reply politely to non relevant questions exceeding your scope.

Question: {question}"""
             )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,return_intermediate_steps=True)


#Main Part of the code
print("Hello Welcome to Chat With LexU, how can help you today? (Note to exit, type 'exit' \n ")
DISCLAIMER = "For more information, you can visit www.dennislawgh.com."
while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
    else:
        # Execute the QA chain
        response = agent_executor({"input": query})
        print(f"LexU Response:{response}\n\nDISCLAIMER:{DISCLAIMER}")
