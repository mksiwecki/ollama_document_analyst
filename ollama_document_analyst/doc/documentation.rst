Ollama Document Analyst Documentation
=====================================

Overview
--------
* Ollama Document Analyst is a local RAG-based tool that allows users to ask questions about PDF documents.
* The application reads PDF files from the data directory, creates a vector database, and uses it to answer questions using a local Ollama model.

Usage
-----
1. Place PDF files inside the ``data`` directory.
2. Start the application by running ``main.py``. It can be launched from IDE, by double-clicking or from terminal, using command ``py main.py``.
3. On startup, the application checks whether the existing database matches the current PDF files based on ``db_metadata.json`` file.
4. If no changes are detected, the existing database is loaded.
5. If PDF files were added, removed, or modified, the database is rebuilt automatically.
6. Exit by writing ``/exit``.

Asking Questions
----------------
* After startup, type questions directly into the console. Example:
::

	(You): What is the main topic of given documents?

* The application will search relevant sections of the PDFs and generate an answer.

Database
--------
* The application uses Chroma as a vector database.
* The database is stored in the ``chroma_db`` directory.
* The database is automatically rebuilt when changes to PDF files are detected.

Project Structure
-----------------
::

	.
	├── chroma_db/
	│	└── Generated vector database.
	│
	├── data/
	│	└── PDF files used for analysis.
	│
	├── doc/
	│	├── changelog.rst
	│	└── documentation.rst
	│
	├── metadata/
	│	└── db_metadata.json
	│
	├── main.py
	└── README.md

Troubleshooting
---------------

Cannot connect to Ollama
~~~~~~~~~~~~~~~~~~~~~~~~
* Make sure Ollama is running.

Application cannot find documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Make sure PDF files are placed inside the ``data`` directory.

Database issues
~~~~~~~~~~~~~~~
* Delete the ``chroma_db`` directory content manually and restart the application to force database recreation.
