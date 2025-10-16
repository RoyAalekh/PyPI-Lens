# PyPI-Lens

A semantic search engine for Python packages that uses machine learning embeddings and similarity matching to help developers discover relevant packages beyond simple keyword matching.

## Overview

PyPI-Lens leverages natural language processing to understand package descriptions and find relevant matches based on semantic meaning rather than just text overlap. The system combines semantic similarity with popularity metrics to provide intelligent package recommendations.

## Features

- **Semantic Search**: Natural language understanding for query processing
- **Smart Ranking**: Combines semantic similarity with download popularity
- **Tag Classification**: Considers package keywords and categories
- **Local Storage**: DuckDB-based efficient indexing and retrieval
- **Auto-Updates**: Maintains current package information
- **Web Interface**: Streamlit-based interactive search interface

## Live Demo

[PyPI-Lens Search Tool](https://pypi-lens.streamlit.app/)

## Technical Architecture

### Core Components
- **Backend**: Python with DuckDB for data storage
- **ML Model**: sentence-transformers (all-MiniLM-L6-v2)
- **Frontend**: Streamlit web application
- **Data Source**: PyPI package registry

### Search Algorithm
The ranking algorithm combines multiple factors:

```
final_score = (semantic_score * 0.6) + (keyword_score * 0.4)
popularity_boost = log10(downloads) / 10
final_score *= (1 + popularity_boost)
```

## Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/RoyAalekh/PyPI-Lens.git
cd PyPI-Lens
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize Database**
```bash
python -m src.pypi_lens.update_index
```

5. **Launch Application**
```bash
streamlit run src/pypi_lens/app.py
```

6. **Access Interface**
Open browser to `http://localhost:8501`

## Usage

### Basic Search
Enter natural language queries to find relevant packages:

- "machine learning framework for deep learning"
- "web scraping with async support"
- "data visualization for scientific plots"

### Search Results
Each result includes:
- Package name and version
- Semantic similarity score
- Download statistics
- Package tags and categories
- Description and documentation links

### API Usage
```python
from src.pypi_lens.database import PyPIDatabase

db = PyPIDatabase()
results = db.search("data analysis toolkit")

for package in results[:5]:
    print(f"{package['name']}: {package['similarity']:.3f}")
```

## Project Structure

```
PyPI-Lens/
├── src/
│   └── pypi_lens/
│       ├── app.py           # Streamlit web interface
│       ├── database.py      # DuckDB operations and search logic
│       └── update_index.py  # Package data indexing system
├── data/
│   └── pypi_lens.db        # Local package database
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── .Dockerfile             # Container configuration
```

## Technical Implementation

### Embedding Generation
- Uses sentence-transformers library
- all-MiniLM-L6-v2 model for 384-dimensional vectors
- Processes package descriptions and search queries

### Database Schema
- **Packages**: name, description, version, downloads
- **Embeddings**: vectorized package descriptions
- **Metadata**: tags, categories, update timestamps

### Search Process
1. Query vectorization using sentence-transformers
2. Cosine similarity calculation against stored embeddings
3. Keyword matching for exact term matches
4. Score combination with popularity weighting
5. Result ranking and filtering

## Configuration

### Environment Variables
- `PYPI_API_URL`: PyPI API endpoint (default: https://pypi.org/pypi/)
- `DB_PATH`: Database file location (default: data/pypi_lens.db)
- `UPDATE_INTERVAL`: Index update frequency in hours

### Model Parameters
- Embedding model: all-MiniLM-L6-v2
- Vector dimensions: 384
- Similarity threshold: 0.3
- Results limit: 50

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Building Docker Image
```bash
docker build -f .Dockerfile -t pypi-lens .
docker run -p 8501:8501 pypi-lens
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Create Pull Request

### Development Areas
- Enhanced search ranking algorithms
- Additional package metadata integration
- Performance optimization
- API endpoint development
- Test coverage expansion

## Dependencies

- **streamlit**: Web application framework
- **sentence-transformers**: ML embedding generation
- **requests**: HTTP client for PyPI API
- **numpy**: Numerical computations
- **duckdb**: Local database system

## Performance

- Database size: ~500MB for full PyPI index
- Search latency: <100ms for typical queries
- Memory usage: ~2GB during indexing, ~500MB runtime
- Update frequency: Daily incremental updates

## License

MIT License - see LICENSE file for details

## Acknowledgments

- sentence-transformers team for embedding models
- DuckDB for efficient local storage
- Streamlit for web framework
- PyPI for package data access