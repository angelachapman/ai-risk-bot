

# Qdrant constants
LOCATION = ":memory:"
OPENAI_VECTOR_SIZE = 1536
HF_VECTOR_SIZE = 768

# RAG constants
CHUNK_SIZE = 1500
OVERLAP = 150
BASELINE_EMBEDDING_MODEL = "text-embedding-3-small"
BASELINE_CHAT_MODEL = "gpt-4o-mini-2024-07-18"

TE3_LARGE = "text-embedding-3-large"
TE3_VECTOR_LENGTH = 3072
GPT_4O = "gpt-4o-2024-08-06"

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful, kind expert in modern AI, AI safety and risk mitigation. You are factual and technically
accurate, but you always give answers that laypeople can understand.
Answer user questions based only on the context below. Answer in at least a paragraph and provide lots of
details, but don't use too much jargon.

You must follow these rules:
- If you don't know, or if the context is not relevant, apologize and say "I don't know".
- If the user asks questions that are not about AI, offer to discuss AI instead. 
- Do not engage in politically polarized, toxic or inappropriate speech

User question:
{input}

Context:
{context}
"""

SYSTEM_PROMPT_TEMPLATE_CL_APP = """
You are a helpful, kind expert in modern AI, AI safety and risk mitigation. You are factual and technically
accurate, but you always give answers that laypeople can understand.
Answer user questions based only on the context below. Answer in a few paragraphs and provide lots of
details. Be concise and don't use too much jargon. 

You must follow these rules:
- If you don't know, or if the context is not relevant, apologize and say "I don't know".
- If the user asks questions that are not about AI, offer to discuss AI instead. 
- Do not engage in politically polarized, toxic or inappropriate speech
- Don't say "according to the context", "according to the document", etc. 

User question:
{input}

Context:
{context}
"""

# RAGAS constants
RAGAS_CHUNK_SIZE = 750
RAGAS_OVERLAP = 75
GENERATOR_LLM = "gpt-4o-mini-2024-07-18"
CRITIC_LLM = "gpt-4o-2024-08-06"
EVALUATION_MODEL = GENERATOR_LLM
N_EVAL_QUESTIONS = 30 # IRL, we'd want more, and maybe a test and validation set. But set it low to accommodate low rate limits.
TEST_DATASET_FILE = f"test_dataset_{N_EVAL_QUESTIONS}.csv"


# Fine tuning constants
FT_CHUNK_SIZE = 500 # Use smaller chunks so that we have more docs for our train/val/test splits
FT_OVERLAP = 50
FT_TRAIN_DATASET_FILE="ft_training_dataset.jsonl"
FT_VAL_DATASET_FILE="ft_val_dataset.jsonl"
FT_TEST_DATASET_FILE="ft_test_dataset.jsonl"
BATCH_SIZE = 20
EPOCHS = 5
FT_MODEL_NAME = "finetuned_arctic_ai_risk"
HF_USERNAME = "achapman"

# Colab-specific
CONTENT_DIR = "/content"

# Dataset
PDFS = [
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
    "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
]
PDF_DICT = {
    "doc1": {"file_path": "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf", "skip_pages_begin": 4, "skip_pages_end": 10},
    "doc2": {"file_path": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf", "skip_pages_begin": 4, "skip_pages_end": None},
}

# Parent-child chunking parameters
PARENT_CHUNK_SIZE = 3000
PARENT_OVERLAP=0.1*PARENT_CHUNK_SIZE
CHILD_CHUNK_SIZE = 500
CHILD_OVERLAP=0.1*CHILD_CHUNK_SIZE