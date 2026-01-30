import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from api import openai_api


#bs4_strainer is used to download information
bs4_strainer = bs4.SoupStrainer(class_=("post-title","post-header","post-content"))

#download document by url
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

#download info into docs variable
docs = loader.load()

print(f"Total characters: {len(docs[0].page_content)}")

#split data from DB into chunks
#chunk_size should be medium
#chunk_overlap for better work
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print(f"Total splits: {len(all_splits)}")


#create DB. we will use Chroma DB

# set model that will be used for embeddings/ Model will take docs that is splitted into chunks and transform into vectors
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api
)

vector_store = Chroma(
    collection_name = "prompt_engineering_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

ids = vector_store.add_documents(all_splits)
print(f"Persisted: {len(ids)} documents to disk")

