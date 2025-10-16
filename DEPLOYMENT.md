# PyPI-Lens Deployment Guide

## Project Setup

PyPI-Lens is a semantic search engine for Python packages that uses machine learning embeddings stored in DuckDB for intelligent package discovery.

## Local Development

### Prerequisites
- Python 3.9 or higher
- UV (for virtual environment management)
- Git

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/RoyAalekh/PyPI-Lens.git
   cd PyPI-Lens
   ```

2. **Create Virtual Environment**
   ```bash
   uv venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   cd src/pypi_lens
   streamlit run app.py
   ```

## Deployment Options

### Streamlit Cloud

1. **Connect Repository**
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Set app file path: `src/pypi_lens/app.py`

2. **Environment Variables**
   - No special environment variables needed
   - Database and embeddings are included

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -f .Dockerfile -t pypi-lens .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 pypi-lens
   ```

### Heroku Deployment

1. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

## Features

- **Semantic Search**: Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
- **Hybrid Scoring**: Combines semantic similarity with keyword matching  
- **4,675 Pre-indexed Packages**: With 384-dimensional embeddings
- **Real-time Search**: Sub-second response times
- **Rich Results**: Package descriptions, tags, versions, and metadata

## Database Information

- **Engine**: DuckDB
- **Embeddings**: 384-dimensional vectors using all-MiniLM-L6-v2
- **Packages**: 4,675 popular Python packages indexed
- **Search Algorithm**: Cosine similarity + keyword matching + popularity boost

## Maintenance

The database includes pre-computed embeddings. To update the package index:

```bash
cd src/pypi_lens
python -m update_index
```

## Architecture

```
User Query -> Embedding Generation -> Similarity Search -> Ranking -> Results
     |              |                       |              |           |
  Natural        Sentence              Cosine          Hybrid     Ranked
  Language      Transformers           Similarity      Scoring    Results
```