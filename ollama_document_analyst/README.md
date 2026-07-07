# Ollama Document Analyst - Version 0.6.0
AI Ollama based project for reading PDF files and answering questions based on those files.

## Features
* PDF analysis using local LLMs (Ollama).
* RAG-based question answering.
* Automatic database rebuilding when PDFs change.

## Requirements
* Python3 (created on Python 3.13)
* Ollama
* Models pulled from Ollama:
  * LLM - `qwen3:8b`
  * Embedding - `nomic-embed-text`
* PIP [requirements](doc/requirements.txt)

## Installation
* Download and install [Ollama](https://ollama.com/download)
* In terminal, pull model `qwen3:8b` using command `ollama pull qwen3:8b`.
  * User can use different model, but in `main.py` global variable `LLM_MODEL_NAME` needs to be changed accordingly.
* Again in terminal, pull model `nomic-embed-text` using command `ollama pull nomic-embed-text`.

## Quick start
* Launch Ollama (user can also open terminal and input `ollama serve`).
* Launch the `main.py`.

## User documentation and changelog
* See [documentation](doc/documentation.rst)
* See [changelog](doc/changelog.rst)
