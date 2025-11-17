## langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Vector store
from langchain_community.vectorstores import Chroma

## utility imports
import os
import numpy as np
from typing import List


print("""
    RAG Architecture:
    1. Document Loading from various sources
    2. Document Splitting: Chunking the docs
    3. Embedding Generation: Convert chunk into vector representation
    4. Vector Storage: Store embedding in Chroma DB
    5. Query Processing: query to vector embedding 
    6. Similarity Search: Find relevant chunk from vector store
    7. Context Augmentation: Combine retrieved chunks with query
    8. Response generation: LLM generates answer using context
    
    Benefits of RAG:
    - Reduces hallucinations
    - Provides up-to-date information
    - Allows citing sources
    - Works with domain-specific knowledge
""")

## create sample documents
sample_docs = [
    """
    Machine Learning Fundamentals

    Machine learning is a subset of artificial intelligence that enables systems to learn 
    and improve from experience without being explicitly programmed. There are three main 
    types of machine learning: supervised learning, unsupervised learning, and reinforcement 
    learning. Supervised learning uses labeled data to train models, while unsupervised 
    learning finds patterns in unlabeled data. Reinforcement learning learns through 
    interaction with an environment using rewards and penalties.
    """,

    """
    Deep Learning and Neural Networks

    Deep learning is a subset of machine learning based on artificial neural networks. 
    These networks are inspired by the human brain and consist of layers of interconnected 
    nodes. Deep learning has revolutionized fields like computer vision, natural language 
    processing, and speech recognition. Convolutional Neural Networks (CNNs) are particularly 
    effective for image processing, while Recurrent Neural Networks (RNNs) and Transformers 
    excel at sequential data processing.
    """,

    """
    Natural Language Processing (NLP)

    NLP is a field of AI that focuses on the interaction between computers and human language. 
    Key tasks in NLP include text classification, named entity recognition, sentiment analysis, 
    machine translation, and question answering. Modern NLP heavily relies on transformer 
    architectures like BERT, GPT, and T5. These models use attention mechanisms to understand 
    context and relationships between words in text.
    """
]

## Saving sample docs to files
if not os.path.exists("./temp"):
    os.mkdir("./temp")


for i, doc in enumerate(sample_docs):
    with open(f"./temp/doc_{i}.txt", "w") as f:
        f.write(doc)

## Document Loading
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./temp",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": 'utf-8'}
)
documents = loader.load()

print(len(documents))

## Document splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Maximum size of each chunk
    chunk_overlap=50, # Overlab between chunks to maintain context
    length_function=len,
    separators=[" "] # Hierarchy of separators
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
for c in chunks:
    print(c)

## Embedding model and Storing document chunks in Chroma DB vector store in vector representation
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# Create chroma DB vector store
persistent_directory = "./chroma_db"

# Initialize chromadb with OPEN AI Embeddings
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=persistent_directory,
    collection_name="rag_collection"
)

print(f"Vector store created with {vector_store._collection.count()} vectors")
print(f"Persisted to: {persistent_directory}")


## Test similarity search
query = "What are the types of Machine Learning"

similar_docs = vector_store.similarity_search(query, k=3)
print(similar_docs)

## Advanced similarity search with scores
result_scores = vector_store.similarity_search_with_score(query, k=3)
print(result_scores)


print("""
    Understanding Similarity Scores
    The similarity score represents how closely related a document chunk is to your query. The scoring depends on the distance metric used:
    
    ChromaDB default: Uses L2 distance (Euclidean distance)
    
    - Lower scores = MORE similar (closer in vector space)
    - Score of 0 = identical vectors
    - Typical range: 0 to 2 (but can be higher)
    
    
    Cosine similarity (if configured):
    
    - Higher scores = MORE similar
    - Range: -1 to 1 (1 being identical)
""")


## Initialize LLM, RAG Chain, Prompt Template, Query the RAG system
llm=ChatOpenAI(model="gpt-3.5-turbo")

# Different wa of initializing LLM model
# from langchain.chat_models.base import init_chat_model
# llm=init_chat_model("openai:gpt-3.5-turbo")

## Modern RAG Chain

retriever = vector_store.as_retriever(
    search_kwarg={"k": 3}
)

prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
)


# Format documents helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
# Create chain that returns both answer and docs
# RAG chain
rag_chain = (
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)


# Function to query with sources
def query_with_sources(question):
    # Get retrieved documents
    docs = retriever.invoke(question)

    # Get answer
    answer = rag_chain.invoke(question)

    return {
      "answer": answer,
      "sources": docs,
      "question": question
    }

# Use it
result = query_with_sources("What is Deep learning?")
print(f"Question: {result['question']}")
print(f"\nAnswer: {result['answer']}")
print(f"\nSources ({len(result['sources'])} documents):")
for i, doc in enumerate(result['sources'], 1):
  print(f"\n--- Document {i} ---")
  print(doc.page_content[:200])
