# TODO: Better history management for memory optimization.
# TODO: Hash-based PDF file detection.
# TODO: Avoid printing DELETE DATABASE when creating a new one without deleting the old (after manual removal).

import os
import json
import shutil

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_PATH = './data'
DB_METADATA_PATH = './metadata/db_metadata.json'
MODEL_NAME = 'nomic-embed-text'
CHROMA_PATH = './chroma_db'
LLM_MODEL_NAME = 'qwen3:8b'
CONTEXT_WINDOW = 8192

def get_pdf_metadata():
	"""Returns metadata for all PDF files in the data directory."""
	metadata = list()

	for filename in sorted(os.listdir(DATA_PATH)):
		if filename.lower().endswith('.pdf'):
			path = os.path.join(DATA_PATH, filename)
			metadata.append({'filename': filename, 'size': os.path.getsize(path), 'modified': os.path.getmtime(path)})

	return metadata

def save_pdf_metadata():
	"""Saves current PDF metadata to a JSON file."""
	metadata = get_pdf_metadata()

	with open(DB_METADATA_PATH, 'w', encoding='utf-8') as file:
		json.dump(metadata, file, indent=4)

	print(f'Metadata saved to {DB_METADATA_PATH}.')

def load_pdf_metadata():
	"""Loads saved PDF metadata from JSON file."""
	if not os.path.exists(DB_METADATA_PATH):
		return None

	with open(DB_METADATA_PATH, 'r', encoding='utf-8') as file:
		return json.load(file)

def is_database_up_to_date():
	"""Checks if PDF files have changed since last indexing."""
	saved = load_pdf_metadata()
	current = get_pdf_metadata()

	if saved is None:
		return False

	if saved != current:
		print(f'Found changes in PDF files stored within {DATA_PATH} directory.')

	return saved == current

def delete_database():
	"""Method for deleting a database."""
	if os.path.exists(CHROMA_PATH):
		print('Deleting previous database...')
		shutil.rmtree(CHROMA_PATH)
		print('Database deleted.')

def load_documents():
	"""Loads documents from the specified data path."""
	# Load all documents from data directory.
	loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
	documents = loader.load()
	pdf_file_count = len([pdf for pdf in os.listdir(DATA_PATH) if pdf.endswith('.pdf')])
	print(f'Loaded {len(documents)} pages from all [{pdf_file_count}] PDFs.')

	return documents

def split_documents(documents):
	"""Splits documents into smaller chunks."""
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1500, chunk_overlap=300, length_function=len, is_separator_regex=False
	)
	all_splits = text_splitter.split_documents(documents)
	print(f'Split into {len(all_splits)} chunks.\n')

	return all_splits

def get_embedding_function(model_name=MODEL_NAME):
	"""Initializes the Ollama embedding function."""
	# Ensure Ollama server is running ("ollama serve" in terminal).
	embeddings = OllamaEmbeddings(model=model_name)
	print(f'Initialized Ollama embeddings with model: {model_name}.')

	return embeddings

def create_vector_store(embedding_function, persist_directory=CHROMA_PATH):
	"""Creates a new Chroma database."""
	# Load documents.
	documents = load_documents()

	# Split documents.
	chunks = split_documents(documents)
	if len(chunks) == 0:
		raise ValueError('No chunks created. Check PDF loading path and files.')

	print(f'Indexing {len(chunks)} chunks...')
	vector_store = Chroma.from_documents(
		documents=chunks, embedding=embedding_function, persist_directory=persist_directory
	)
	save_pdf_metadata()

	print(f'Indexing complete. Data saved to: {persist_directory}.\n')
	return vector_store

def load_vector_store(embedding_function, persist_directory=CHROMA_PATH):
	"""Loads existing Chroma database."""
	vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
	print(f'Vector store loaded from: {persist_directory}.')

	return vector_store

def initialize_vector_store(embedding_function):
	"""Method for initializing vector_store for RAG chain."""
	print('Checking database state...')

	if is_database_up_to_date() and os.listdir(CHROMA_PATH):
		print('Loading existing database...\n')
		return load_vector_store(embedding_function)

	# Delete previous database if there is one, for data consistency.
	delete_database()

	print('Building new database...\n')
	return create_vector_store(embedding_function)

def create_rag_chain(vector_store, llm_model_name=LLM_MODEL_NAME, context_window=CONTEXT_WINDOW):
	"""Creates the RAG chain."""
	# Initialize the LLM.
	llm = ChatOllama(model=llm_model_name, temperature=0.2, num_ctx=context_window)
	print(f'Initialized ChatOllama with model: {llm_model_name} - Context window: {context_window}.')

	# Create the retriever.
	retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
	print('Retriever initialized.')

	# Define the prompt template.
	template = """
You are aa analyst of the given PDF files.

Use the conversation history and the context to answer.

Conversation history:
{history}

Context:
{context}

Question:
{question}
"""
	prompt = ChatPromptTemplate.from_template(template)
	print('Prompt template created.')

	# Define the RAG chain using LCEL.
	rag_chain = (
			{'context': (lambda x: x["question"]) | retriever, 'question': RunnablePassthrough(),
			 'history': RunnablePassthrough()}
			| prompt | llm
	)
	print('RAG chain created.')

	return rag_chain

def query_rag(chain, question, history):
	"""Queries the RAG chain and prints the response."""
	if question != '/token':
		print('\nAnalyst is thinking...')

	response = chain.invoke({
		'question': question,
		'history': history
	})

	if question != '/token':
		print(f'Question: {question}')
		print('\nResponse:')
		print(response.content)

	return response

# --- Main Execution! ---
if __name__ == '__main__':
	# 1. Get embedding function.
	embedding_function = get_embedding_function()

	# 2. Load or index a database and save metadata.
	vector_store = initialize_vector_store(embedding_function)

	# 3. Create RAG chain.
	rag_chain = create_rag_chain(vector_store)

	# 4. Running program.
	chat_history = list()
	print('\nAI PDF Analyst ready.\nType /exit to quit.\nType /token to see token count.\n')
	while True:
		question = input('(You): ')

		if question.strip().lower() == '/exit':
			print('Exiting...')
			break

		if question.strip().lower() == '/token':
			response = query_rag(rag_chain, question, chat_history)
			total_usage = response.response_metadata['prompt_eval_count'] + response.response_metadata['eval_count']
			remaining = CONTEXT_WINDOW - total_usage
			remaining_percentage = (remaining / CONTEXT_WINDOW) * 100

			print(f'Total usage: {total_usage}')
			print(f'Remaining tokens: {remaining}')
			print(f'Remaining tokens (percentage): {round(remaining_percentage, 2)}%\n')

			continue

		else:
			response = query_rag(rag_chain, question, chat_history)

		# Update chat history.
		chat_history.append({'role': 'user', 'content': question})
		chat_history.append({'role': 'assistant', 'content': response})
