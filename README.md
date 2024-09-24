---
title: AIRiskBot!
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---
## AI Risk Chatbot ##

This repo contains code for a simple chainlit-based chatbot that discusses the risks of the modern AI industry based on information from NIST and Whitehouse.gov. 

Repo contents:
* app.py is a chainlit application 
* ai_risk_bot_rag.ipynb is a notebook that compares different RAG pipelines
* gen_synthetic_data.ipynb generates synthetic test data using RAGAS
* write_chunked_docs.ipynb chunks the documents and serializes them to help make the chainlit app faster
* vars.py contains variables and constants
* utils.py contains supporting functions for the notebook
* fine_tuning_arctic contains code to fine-tune an open-source embedding model

To run the code, you'll need an OpenAI key.
