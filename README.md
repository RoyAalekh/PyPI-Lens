# PyPI-Lens 🔍

> A semantic search engine for Python packages using embeddings and similarity matching.

PyPI-Lens helps you discover Python packages through semantic search, going beyond simple keyword matching. It uses machine learning to understand package descriptions and find relevant matches based on meaning, not just text overlap.

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

## ✨ Features

- 🧠 **Semantic Search**: Understands the meaning behind your queries
- 🎯 **Smart Ranking**: Combines semantic similarity with popularity metrics
- 🏷️ **Tag-Aware**: Considers package classifications and keywords
- ⚡ **Fast Search**: Local DuckDB storage with efficient indexing
- 🔄 **Auto-Updates**: Keeps package information fresh and relevant
- 🎨 **Clean UI**: Streamlit-based interface for easy exploration

## 🚀 Quick Start

1. **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/pypi-lens.git
cd pypi-lens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Initialize Database**

```bash
python -m src.pypi_lens.update_index
```

3. **Run the App**

```bash
streamlit run src/pypi_lens/app.py
```

## 🔎 How It Works

PyPI-Lens uses [sentence-transformers](https://www.sbert.net/) to convert package descriptions and search queries into embeddings - high-dimensional vectors that capture semantic meaning. When you search:

1. Your query is converted to an embedding
2. This is compared with stored package embeddings
3. Results are ranked by:
   - Semantic similarity (60%)
   - Keyword matching (40%)
   - Download count boost

### Example Search

```python
# Search for data science packages
results = db.search("machine learning framework for deep learning")

# Sample result
{
    "name": "tensorflow",
    "similarity": 0.89,
    "downloads": 15000000,
    "tags": ["deep-learning", "machine-learning", "neural-networks"]
}
```

## 🛠️ Architecture

```
PyPI-Lens/
├── src/
│   └── pypi_lens/
│       ├── app.py           # Streamlit interface
│       ├── database.py      # DuckDB operations
│       └── update_index.py  # Package updater
├── data/
│   └── pypi_lens.db        # Package database
└── requirements.txt
```

## 📊 Technical Details

- **Database**: DuckDB for efficient local storage
- **Embeddings**: all-MiniLM-L6-v2 model (384-dimensional vectors)
- **Search Ranking**:
  ```
  final_score = (semantic_score * 0.6) + (keyword_score * 0.4)
  popularity_boost = log10(downloads) / 10
  final_score *= (1 + popularity_boost)
  ```

## 🤝 Contributing

Contributions are welcome! Areas that need attention:

- [ ] Add unit tests
- [ ] Improve search ranking algorithm
- [ ] Add more package metadata
- [ ] Implement caching for frequent searches

## 🙏 Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embedding generation
- [DuckDB](https://duckdb.org/) for efficient storage
- [Streamlit](https://streamlit.io/) for the web interface
- PyPI for package data

---

Made with ❤️ by [AR](https://royaalekh.github.io/)
