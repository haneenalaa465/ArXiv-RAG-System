# ArXiv RAG: Retrieval-Augmented Generation with ArXiv Papers

This repository contains a comprehensive implementation of a Retrieval-Augmented Generation (RAG) system for scientific paper analysis using the ArXiv dataset. The system enables semantic querying of scientific literature by embedding, retrieving, and generating responses based on relevant paper excerpts.

## 🔍 Overview

This RAG pipeline processes ArXiv paper abstracts and titles, chunks them meaningfully, embeds the chunks using state-of-the-art language models, performs semantic retrieval based on user queries, and generates informative responses using a language model.

## 🚀 Features

- **Data Ingestion & Cleaning**: Load and clean ArXiv paper metadata from JSON or text files
- **Intelligent Chunking**: Sentence-based text chunking with configurable size and stride
- **Advanced Embedding**: Document embedding using modern semantic embedding models
- **Semantic Search**: Efficient vector search for retrieving the most relevant document chunks
- **Context-Aware Response Generation**: LLM-based response generation using retrieved contexts
- **Comprehensive Evaluation**: Retrieval quality metrics and generation evaluation

## 🔧 Technologies Used

- **Python 3.7+**
- **PyTorch** for tensor operations and deep learning models
- **Sentence Transformers** for text embedding
- **Transformers** for LLM-based text generation
- **NLTK** for text processing
- **pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for visualization
- **rouge_score** for evaluation metrics

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/haneenalaa465/ArXiv-RAG-System
cd ArXiv-RAG-System

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📄 Requirements

```
torch>=1.10.0
transformers>=4.26.0
sentence-transformers>=2.2.2
huggingface-hub>=0.15.1
pandas>=1.3.5
numpy>=1.21.6
matplotlib>=3.5.3
seaborn>=0.12.0
nltk>=3.7
rouge_score>=0.1.2
scikit-learn>=1.0.2
tqdm>=4.64.1
bitsandbytes>=0.41.0
psutil>=5.9.0
```

## 📊 Dataset

The project uses the ArXiv Abstract & Title Dataset, which includes titles and abstracts from computer science papers. The dataset is downloaded from Zenodo during the setup process.

```python
!wget -O arxiv_abs_title.zip "https://zenodo.org/records/3496527/files/gcunhase%2FArXivAbsTitleDataset-v1.0.zip?download=1"
!unzip arxiv_abs_title.zip
```

## 🏗️ Pipeline Components

### 1. Data Ingestion & Cleaning

The system loads ArXiv paper data from either JSON files or text files, cleans the text, and prepares it for processing.

### 2. Chunking Strategy

Papers are divided into meaningful chunks using a sentence-based chunking approach that preserves semantic units:

```python
chunks = create_document_chunks(
    arxiv_df,
    chunk_method='sentence',
    chunk_size=5,  # 5 sentences per chunk
    stride=3       # 3-sentence overlap
)
```

### 3. Vectorization

Chunks are embedded using efficient semantic embedding models:

```python
chunk_embeddings, embedding_model = embed_chunks(
    all_chunks,
    model_name="jinaai/jina-embeddings-v2-small-en",
    batch_size=32
)
```

### 4. Retrieval Module

The system retrieves the most relevant chunks for a given query using cosine similarity:

```python
indices, scores, _ = retrieve_similar_chunks(
    query,
    chunk_embeddings,
    embedding_model,
    all_chunks,
    top_k=5
)
```

### 5. Prompt Construction & Generation

Retrieved chunks are used to construct a prompt for the language model, which then generates a response:

```python
result = generate_answer(
    query,
    llm_model,
    llm_tokenizer,
    chunk_embeddings,
    embedding_model,
    all_chunks,
    top_k=5
)
```

### 6. Evaluation & Reflection

The system evaluates retrieval quality using metrics like recall@k and analyzes response quality using ROUGE scores.

## 🔬 Model Selection

### Embedding Model

The system uses the `jinaai/jina-embeddings-v2-small-en` model for embeddings, which offers:
- 512-dimensional embeddings
- Excellent semantic understanding
- Good balance of performance and efficiency

### Generation Model

For text generation, the system uses `google/gemma-2b-it`, a lightweight yet powerful instruction-tuned model that:
- Works well with 4-bit quantization for reduced memory usage
- Provides coherent responses to academic queries
- Has been trained on diverse content including scientific literature

## 📝 Usage

### Basic Usage

```python
# Load and process the ArXiv dataset
arxiv_df = find_and_load_arxiv_dataset()

# Create chunks
all_chunks = create_document_chunks(
    arxiv_df,
    chunk_method='sentence',
    chunk_size=5,
    stride=3
)

# Compute embeddings
chunk_embeddings, embedding_model = embed_chunks(
    all_chunks,
    model_name="jinaai/jina-embeddings-v2-small-en",
    batch_size=32
)

# Load the generation model
llm_model, llm_tokenizer = load_llm_model("google/gemma-2b-it", use_4bit=True)

# Generate answer for a query
result = generate_answer(
    "What are the major challenges in natural language processing?",
    llm_model,
    llm_tokenizer,
    chunk_embeddings,
    embedding_model,
    all_chunks,
    top_k=5
)

print(result['answer'])
```

## 📊 Performance

### Retrieval Performance

The system achieves the following recall values:
- Recall@5: ~0.45
- Recall@10: ~0.62
- Recall@20: ~0.78

### Generation Quality

Response quality is evaluated using metrics including:
- ROUGE scores when reference answers are available
- Response length analysis
- Semantic relevance to the query

## 🔄 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ArXiv for providing the dataset of paper abstracts and titles
- The Sentence Transformers and Transformers libraries for enabling efficient embedding and generation
- The research community for advancing the state of RAG systems
