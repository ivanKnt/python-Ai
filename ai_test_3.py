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
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.storage import  LocalFileStore
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains import create_qa_with_sources_chain



openai_api_key = "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"

# initialize the LLM
llm = ChatOpenAI( openai_api_key=openai_api_key, temperature=0,streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()])

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


for i, text in enumerate(docs):
    text.metadata["source"] = f"{i}-pl"


#We need to add now some type of cache, so we can use the embeddings without embed the documents again if already

#Set the storage
storage = LocalFileStore("./cache/")
#Set the cached embedder
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,storage,namespace=embeddings.model
)

# Embed documents and store it in a vectorDatabase
db = FAISS.from_documents(docs, cached_embedder)
retriever = db.as_retriever()  # Our Database as a Retriever



template = """

Instruction: My name is LexU, and I am a highly respected legal expert specializing in Ghanaian law. My expertise covers various legal areas such as civil law, criminal law, contract law, constitutional law, and administrative law. I am here to provide comprehensive and accurate legal advice based on the documents I have been provided. I do not consult outside sources.

Your role is to maintain a friendly and conversational tone while providing direct and clear answers to any legal queries. Do not pose questions in your responses. Prioritize clarity, precision, and professionalism. Ensure that the information you provide is up-to-date and in accordance with the legal framework of Ghana. If a question is outside your expertise or not relevant to the documents you have, kindly inform the user in a friendly manner that you are unable to assist with that particular query.

Remember to reference any laws or articles from the documents when providing advice and add a disclaimer at the end of your responses, stating that the advice is based on the documents you have been given.

Use any tools if necessary and reply politely to non relevant questions exceeding your scope.

Question: {question}
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = template),
    MessagesPlaceholder(variable_name = "chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

#prompt = PromptTemplate(
 #        input_variables=["chat_history","question"], template=template,
#)

#set the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)
qa_chain = create_qa_with_sources_chain(llm)

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context",
    document_prompt=doc_prompt,
)
retriever_qa = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=final_qa_chain
)



#Define Tool for AI to act as a Legal ASSITANT
tool = create_retriever_tool(
    retriever,
    name="legal_assistant",
    description="Useful when asked about questions related to the legal Environment"
)

#Set the tools used by the LLM
tools = [tool,

         Tool(
             func= retriever_qa.run,
             description= "Useful to Return Source documents",
             name= "source_retriever"
         )
         ]

# This is needed for both the memory and the prompt (need to consult the documentation about this)
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

system_message = SystemMessage(
    content=("""Instruction: My name is LexU, and I am a highly respected legal expert specializing in Ghanaian law. My expertise covers various legal areas such as civil law, criminal law, contract law, constitutional law, and administrative law. I am here to provide comprehensive and accurate legal advice based on the documents I have been provided. I do not consult outside sources.

Your role is to maintain a friendly and conversational tone while providing direct and clear answers to any legal queries. Do not pose questions in your responses. Prioritize clarity, precision, and professionalism. Ensure that the information you provide is up-to-date and in accordance with the legal framework of Ghana. If a question is outside your expertise or not relevant to the documents you have, kindly inform the user in a friendly manner that you are unable to assist with that particular query.

Remember to reference any laws or articles from the documents when providing advice and add a disclaimer at the end of your responses, stating that the advice is based on the documents you have been given.

Use any tools if necessary and reply politely to non relevant questions exceeding your scope.
When answering to legal questions, use the source retriever tool to add the sources to your information at the end
Question: {question}"""
             )
)

prompt_2 = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt_2)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False,return_intermediate_steps=True)


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
        print(f"LexU Response:{response['output']}\n\nDISCLAIMER:{DISCLAIMER}")

