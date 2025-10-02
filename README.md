🎬 IMBD Sentimental Analysis

A Streamlit-based deep learning web app for sentiment analysis of IMDB movie reviews. Uses a pre-trained RNN to classify reviews as Positive 😊 or Negative 😞. Built on the IMDB dataset.
✨ Features

Interactive text input for real-time sentiment predictions.
Pre-trained RNN model with high accuracy (simple_rnn_imdb.h5).
Clear feedback: ✅ Positive or ❌ Negative with probability scores.
Supports custom movie reviews for testing.
Includes Jupyter notebooks for model training and exploration.

📊 Dataset

Source: IMDB Dataset
Features: Movie review text (variable length); Target: Sentiment (Positive/Negative).
Download the dataset or use the pre-trained model directly.

🛠 Tech Stack



Category
Tools



Backend
Streamlit, TensorFlow, Keras


ML
Recurrent Neural Network (RNN), NumPy


Frontend
Streamlit, HTML/CSS


🚀 Installation & Setup

Clone: git clone https://github.com/akshi-20/IMBD-Sentimental-Analysis.git && cd IMBD-Sentimental-Analysis
Virtual env: python -m venv venv && source venv/bin/activate (Windows: venv\Scripts\activate)
Install: pip install -r requirements.txt (Streamlit, TensorFlow, etc.)
Run: streamlit run main.py → Visit http://localhost:8501

📖 Usage

Home: App overview → Enter review text.
Input: Type or paste a movie review → Click Classify.
Example: "This movie was absolutely fantastic!" → Positive 😊 (~0.92 probability).

🔄 Model Pipeline

Preprocess: Tokenize text, apply word embeddings.
Predict: Load simple_rnn_imdb.h5 for inference.
Train: Use simplernn.ipynb for model training or embedding.ipynb for word embeddings.

🤝 Contributing
Fork → Branch → PR. Issues welcome!
📄 License
MIT - see LICENSE.
🙏 Acknowledgments

IMDB dataset by Stanford AI.
TensorFlow & Keras for model development.
Streamlit for the interactive UI.

👩‍💻 Author
Akshi — Machine Learning & NLP Enthusiast📍 GitHub: akshi-20
⭐ Star if useful! 🚀

<div align="center">
  <p>⭐ Star this repo on GitHub if you find it useful! ⭐</p>
  <p>Built with 💻 and ❤️ by Akshi</p>
</div>
```

