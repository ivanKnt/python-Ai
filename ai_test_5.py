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
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.storage import LocalFileStore
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain, _load_refine_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import HumanInputRun

openai_api_key = "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"

# initialize the LLM
llm_chatModel = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5, streaming=True,
                           callbacks=[FinalStreamingStdOutCallbackHandler()])


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

# We need to add now some type of cache, so we can use the embeddings without embed the documents again if already

# Set the storage
storage = LocalFileStore("./cache/")
# Set the cached embedder
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, storage, namespace=embeddings.model
)

# Embed documents and store it in a vectorDatabase
db = FAISS.from_documents(docs, cached_embedder)
retriever = db.as_retriever()  # Our Database as a Retriever

"""
We have our Retriever Storage, now we want the LLM to do some kind of things: 

-Summarize Text in a Legal format
-Summarize in IRAC format maybe( additional)
-Maintain a friendly conversation
-Reply to questions over the documents 


Means we need to set a Tool for each of these functionalities.

"""

# We need the Agent to act in a Certain way, so we need to pass him a SystemMessage

system_message = SystemMessage(
    # How the Agent will act
    content="""
    Your Name is LexU and you are an AI Legal Assistant expert in Ghana. 
    You only know about laws in ghana.
    Use all the relevant tools necessary .
    Prioritize friendliness with the user and based on his questions and your answers suggest follow up questions.
    Note that All your answers shall be targeted to Ghana if the questions do not regard Ghana, mention that it is not in your scope.
    Provide expert answers, quite lengthy but comprehensible and explanatory when regarding the legal Aspect.

    Based on the answer you give, if related to the legal Environent, always recommend to visit DennisLawGhana (dennislawgh.com) because
    they provide up to date cases and laws
    """
)


"""
List of Tools that would be used by the Agent
"""
# Get relevant Documents





# Document Summarizer Tool
prompt_template_2 = """
Instruction: You are supposed to use this context {context} to produce a Summary worthy of a legal sector
Question: {question}
"""
PROMPT2 = PromptTemplate(
    template=prompt_template_2, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT2}
qa_summary = RetrievalQA.from_chain_type(llm=llm_chatModel, chain_type="stuff", retriever=retriever,chain_type_kwargs=chain_type_kwargs)






# Document Summarizer Tool (IRAC FORMAT)
prompt_template_3 = """
Instruction: You are supposed to use this context {context} to produce a Summary  in IRAC format helpful for law students
"""
PROMPT3 = PromptTemplate(
    template=prompt_template_3, input_variables=["context"]
)
chain_type_kwargs = {"prompt": PROMPT3}
qa_summary_irac = RetrievalQA.from_chain_type(llm=llm_chatModel, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)






# Document answering Tool
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
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm_chatModel, chain_type="stuff", retriever=retriever,chain_type_kwargs=chain_type_kwargs)




#document retriever Tool
retrieve_docs = create_retriever_tool(
    retriever,
    name="retriever",
    description="Useful to retrieve informations related to the legal environment"
)





# Document Comparaison Tool (check : https://python.langchain.com/docs/integrations/toolkits/document_comparison_toolkit)





# Human Agent (https://python.langchain.com/docs/integrations/tools/human_tools)
humanTool = HumanInputRun()



"""
Initialize the Tools
"""
tools = ([
    # retrieve_docs,
    Tool(
        name="legal_summarizer",
        func=qa_summary.run,
        description="Useful to summarize documents related to the Legal Environment",
    ),
    Tool(
        name="legal_summarizer_irac",
        func=qa_summary_irac.run,
        description="Useful to summarize legal  documents in IRAC format",
    ),
    Tool(
        name="legal_expert_ghana",
        func=qa.run,
        description="Useful when asked about questions related to the legal Environment "
    ),

    #humanTool

])

"""
Agents Initializer and Execution
"""

# We Create the Prompt using the (MultiFunction Agent) and add a place to store the History of our chat
MEMORY_KEY = "chat_history"  # Memorykey
prompt = OpenAIMultiFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

# The Buffer Memory that will hold the Chat History
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)




# Initialize the Agent with the prompt , the tools and the Type (here: MultiFUNCTION AGENT)
agent = OpenAIFunctionsAgent(llm=llm_chatModel, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=False)

# Main Part of the code
print("Hello Welcome to Chat With LexU, how can i help you today? (Note to exit, type 'exit')\n ")
DISCLAIMER = "For more information, you can visit www.dennislawgh.com."
while True:
    query = input("User: ")
    if query.lower() == 'exit':
        break
    else:
        # Execute the QA chain
        response = agent_executor({"input": query})
        print(f"LexU Response:{response['output']}\n\n{DISCLAIMER}")
