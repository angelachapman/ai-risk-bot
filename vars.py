# RAG constants
CHUNK_SIZE = 1500
OVERLAP = 150
BASELINE_EMBEDDING_MODEL = "text-embedding-3-small"
BASELINE_CHAT_MODEL = "gpt-4o-mini-2024-07-18"

# RAGAS constants
RAGAS_CHUNK_SIZE = 750
RAGAS_OVERLAP = 75
GENERATOR_LLM = "gpt-4o-mini-2024-07-18"
CRITIC_LLM = "gpt-4o-2024-08-06"
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