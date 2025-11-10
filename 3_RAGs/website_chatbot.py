import os
import requests
import tempfile
import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import InMemorySaver

# --- Configuration variables ---
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_TOKENS = 15000
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4

# --- Set up OpenAI API key ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- Helper Functions (Scraping and Loading) ---

def fetch_html(url):
    """Fetches the raw HTML content from a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None


def process_website(url):
    """Loads HTML, splits it into LangChain Documents, and returns the chunks."""
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        print("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    os.unlink(temp_file_path)
    print(f"\nNumber of documents loaded: {len(documents)}")

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n\n",
        length_function=len,
        is_separator_regex=False
    )
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks after splitting: {len(texts)}")
    return texts


def print_sample_embeddings(texts, embeddings):
    """Generates and prints a sample embedding for verification."""
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)
        print("\nSample Text:")
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        print("\nSample Embedding (first 10 dimensions):")
        print(np.array(sample_embedding[:10]))
        print(f"\nEmbedding shape: {np.array(sample_embedding).shape}")
    else:
        print("No texts available for embedding sample.")


def format_docs(docs):
    """Formats the list of documents into a single string for the prompt's context variable."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- RAG Pipeline (Simple, Non-LCEL Approach) ---

# Set up OpenAI language model
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# Define the Prompt using ChatPromptTemplate
qa_template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
"""
QA_PROMPT = ChatPromptTemplate.from_template(qa_template)


def rag_pipeline(query, retriever, llm, prompt_template):
    """
    Simple RAG pipeline without LCEL - straightforward function calls.
    """
    # 1. Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # 2. Log the retrieved documents
    print("\nTop 3 most relevant chunks:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"{i}. Content: {doc.page_content}...")
    print("\n" + "=" * 50 + "\n")
    
    # 3. Format the context
    context_content = format_docs(docs)
    
    # 4. Print the full prompt (for logging/debugging) - display raw template text
    print("\nFull Prompt sent to the model:")
    print(f"Context: {context_content}\n")
    print(f"Question: {query}\n")
    print("Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say \"I don't have enough information to answer that question.\"\n")
    print("But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?")
    print("\n" + "=" * 50 + "\n")
    
    # 5. Invoke the LLM
    messages = prompt_template.format_messages(context=context_content, question=query)
    response = llm.invoke(messages)
    
    # 7. Extract and return the answer
    return response.content


if __name__ == "__main__":
    print("Welcome to the Enhanced Web Scraping RAG Pipeline (Simple Version).")

    while True:
        url = input("Please enter the URL of the website you want to query (or 'quit' to exit): ")
        if url.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        try:
            print("Processing website content...")
            texts = process_website(url)

            if not texts:
                print("No content found on the website. Please try a different URL.")
                continue

            print("Creating embeddings and vector store...")
            embeddings = OpenAIEmbeddings()
            print_sample_embeddings(texts, embeddings)
            vectorstore = FAISS.from_documents(texts, embeddings)

            # Get the retriever from the vectorstore, specifying k=3 retrieval
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            print("\nRAG Pipeline initialized. You can now enter your queries.")
            print("Enter 'new' to query a new website or 'quit' to exit the program.")

            while True:
                user_query = input("\nEnter your query: ")
                if user_query.lower() == 'quit':
                    print("Exiting the program. Goodbye!")
                    exit()
                elif user_query.lower() == 'new':
                    break

                # Execute the RAG pipeline
                result = rag_pipeline(user_query, retriever, llm, QA_PROMPT)
                print(f"RAG Response: {result}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try a different URL or check your internet connection.")