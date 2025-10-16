# src/embedding_lens/app.py
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import torch
import umap
from typing import List, Dict

class EmbeddingDemo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

    def create_visualization(self, vector: np.ndarray, title: str = "Embedding Vector") -> go.Figure:
        """Create an interactive visualization of the embedding vector."""
        fig = go.Figure()

        # Add bar plot for vector values
        fig.add_trace(go.Bar(
                x=list(range(len(vector))),
                y=vector,
                name='Dimension Values',
                marker_color='rgb(55, 83, 109)'
        ))

        # Add scatter plot for connecting lines
        fig.add_trace(go.Scatter(
                x=list(range(len(vector))),
                y=vector,
                mode='lines',
                name='Trend',
                line=dict(color='rgba(255, 182, 193, 0.3)')
        ))

        fig.update_layout(
                title=title,
                xaxis_title='Dimension',
                yaxis_title='Value',
                hovermode='x unified',
                height=400
        )

        return fig

    def process_text(self, text: str) -> Dict:
        """Process text and return tokenization details."""
        # Get tokenization
        tokens = self.model.tokenizer.tokenize(text)

        # Get embedding
        embedding = self.model.encode(text)

        return {
            'tokens': tokens,
            'embedding': embedding,
            'dim': len(embedding)
        }

    def compare_texts(self, texts: List[str]) -> np.ndarray:
        """Compare multiple texts and return similarity matrix."""
        embeddings = [self.model.encode(text) for text in texts]

        # Calculate similarity matrix
        similarities = np.zeros((len(texts), len(texts)))
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                similarities[i, j] = similarity

        return similarities

def main():
    st.set_page_config(
            page_title="Embedding Lens",
            layout="wide"
    )

    demo = EmbeddingDemo()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
            "Choose Section",
            ["Introduction", "Embedding Explorer", "Real-world Examples"]
    )

    if mode == "Introduction":
        st.title("Understanding Package Embeddings")
        st.markdown("""
        This tool helps you understand how text descriptions of Python packages
        are converted into numerical vectors (embeddings) that computers can understand
        and compare.
        """)

        st.header("Try it yourself!")
        text_input = st.text_area(
                "Enter any text",
                value="scikit-learn: machine learning library for python",
                height=100
        )

        if text_input:
            result = demo.process_text(text_input)

            # Show steps
            st.subheader("1. Tokenization")
            st.write("The text is split into meaningful pieces (tokens):")
            st.code(result['tokens'])

            st.subheader("2. Encoding")
            st.write(f"The text is converted into a {result['dim']}-dimensional vector:")
            st.plotly_chart(demo.create_visualization(result['embedding']))

            with st.expander("What do these numbers mean?"):
                st.write("""
                Each number represents a different aspect of the text's meaning.
                While individual dimensions aren't directly interpretable, the
                pattern of values captures semantic meaning - similar texts will
                have similar patterns.
                """)

    elif mode == "Embedding Explorer":
        st.title("Embedding Explorer")

        col1, col2 = st.columns(2)

        with col1:
            text1 = st.text_area(
                    "Text 1",
                    value="pandas: data analysis library",
                    height=100
            )

        with col2:
            text2 = st.text_area(
                    "Text 2",
                    value="numpy: numerical computing library",
                    height=100
            )

        if text1 and text2:
            # Process both texts
            result1 = demo.process_text(text1)
            result2 = demo.process_text(text2)

            # Show embeddings side by side
            col1.plotly_chart(demo.create_visualization(
                    result1['embedding'],
                    "Text 1 Embedding"
            ))

            col2.plotly_chart(demo.create_visualization(
                    result2['embedding'],
                    "Text 2 Embedding"
            ))

            # Calculate and show similarity
            similarity = np.dot(result1['embedding'], result2['embedding']) / (
                    np.linalg.norm(result1['embedding']) * np.linalg.norm(result2['embedding'])
            )

            st.metric(
                    "Similarity Score",
                    f"{similarity:.2%}"
            )

            st.info("""
            The similarity score shows how related the texts are in the embedding space.
            Scores closer to 100% indicate more similar meaning.
            """)

    else:  # Real-world Examples
        st.title("Real-world Examples")

        st.header("Package Categories")
        examples = {
            "Data Science": [
                "pandas: data analysis and manipulation",
                "numpy: numerical computing",
                "matplotlib: data visualization"
            ],
            "Web Development": [
                "django: web framework",
                "flask: lightweight web framework",
                "fastapi: modern web APIs"
            ],
            "Machine Learning": [
                "tensorflow: machine learning framework",
                "pytorch: deep learning platform",
                "scikit-learn: machine learning library"
            ]
        }

        category = st.selectbox("Choose a category", list(examples.keys()))

        if category:
            texts = examples[category]
            similarities = demo.compare_texts(texts)

            # Show similarity matrix
            fig = px.imshow(
                    similarities,
                    labels=dict(x="Package", y="Package", color="Similarity"),
                    x=[text.split(":")[0] for text in texts],
                    y=[text.split(":")[0] for text in texts],
                    title=f"Similarity Matrix for {category} Packages"
            )
            st.plotly_chart(fig)

            st.write("""
            The similarity matrix shows how related different packages in the same
            category are to each other. Darker colors indicate higher similarity.
            """)

            # Show individual embeddings
            for text in texts:
                result = demo.process_text(text)
                st.plotly_chart(demo.create_visualization(
                        result['embedding'],
                        f"Embedding for {text.split(':')[0]}"
                ))

if __name__ == "__main__":
    main()
