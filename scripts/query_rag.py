from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from api import openai_api

#open existing DB
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api
)
vector_store = Chroma(
    collection_name = "prompt_engineering_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

#create augmented prompt
prompt = ChatPromptTemplate.from_template(
    """You are helpful assistant that can answer questions about the blog post on prompt engineering. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say "I don't know".
    Question: {question}
    Context: {context}
    Answer:"""
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api
)

# this is a mock of user prompt
#question = "What is prompt engineering?"
#question = "What is Self-Consistency Sampling?"
question = "What is Ferrari?"

# get k=3 most relevant answers for question
retrieved_docs = vector_store.similarity_search(question,k=3)
docs_content = "\n".join([doc.page_content for doc in retrieved_docs])

augmented_prompt = prompt.invoke(
    {"question": question, "context": docs_content}
)

answer = llm.invoke(augmented_prompt)

print(answer.content)

