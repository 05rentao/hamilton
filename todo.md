# 🎭 Hamilton Lyrics Analysis Project Roadmap

This project explores character identity and style in *Hamilton* using NLP and machine learning. There are two main parts:

- **Part 1:** Compare and analyze character lyrics (semantic, stylistic, emotional).
- **Part 2:** Build an ML model to predict who said a given line.

---

## ✅ Part 1: Character Lyrics Analysis

### 1. Extract and Clean Character Lines
- [ ] Parse the script using regex to assign lines to characters.
- [ ] Normalize contractions (e.g. “I’m” → “I am”) and remove stage directions.
- [ ] Filter down to major characters (e.g. HAMILTON, BURR, ANGELICA, ELIZA, etc.)

> **Why it's interesting:** Creates a solid foundation for every analysis step. Also fun to debug regex against a real-world script format!

---

### 2. Bag-of-Words + TF-IDF Vectors
- [ ] Use `TfidfVectorizer` to convert character documents into vectors.
- [ ] Compute cosine similarity between characters.
- [ ] Visualize as a similarity matrix or heatmap.

> **Why it's interesting:** Shows how “close” characters are based on word usage — maybe Burr and Jefferson cluster together?

---

### 3. Topic Modeling with LDA
- [ ] Use `LatentDirichletAllocation` to extract topics from character lyrics.
- [ ] Print top words per topic and assign topic distributions to characters.
- [ ] Compare which themes dominate each character’s dialogue.

> **Why it's interesting:** Surfaces hidden thematic structures — e.g., Hamilton might have “legacy/politics” topics vs. Eliza’s “family/love.”

---

### 4. Word Embedding Averages
- [ ] Train Word2Vec or use pretrained GloVe vectors.
- [ ] Average each character’s line embeddings.
- [ ] Plot with PCA or t-SNE to visualize how semantically distinct they are.

> **Why it's interesting:** Goes beyond word frequency — characters who *mean* similar things may show up close even if they use different vocab.

---

### 5. Emotion & Sentiment Analysis
- [ ] Use NRC Emotion Lexicon or VADER to score lines for emotion categories.
- [ ] Aggregate per character (e.g., Hamilton has more anger, Eliza more trust?)
- [ ] Optionally visualize emotions over time or by act/song.

> **Why it's interesting:** You can actually map Hamilton’s emotional arc — and compare it to Burr’s or Angelica’s.

---

### 6. Syntactic & Stylistic Analysis
- [ ] Use `spaCy` to extract POS tag distributions per character.
- [ ] Analyze sentence length, use of exclamations/questions/imperatives.
- [ ] Compare rhetoric: does Hamilton use more first-person pronouns? Does Burr ask more questions?

> **Why it's interesting:** Stylometry insights — helps detect *how* characters speak, not just what they say.

---

## 🤖 Part 2: Machine Learning Classifier (Who Said This Line?)

### 1. Create Dataset
- [x] Split extracted data by line, and create (character/line) pairs
- [x] Store in `df_ml` with cols `character`, `lines_filtered`, `lines_unfiltered`
- [ ] Remove characters with very few lines or group them as "OTHER"
- [ ] Split into training/test sets

---
### bonus: Perceptron classifier
- [ ] Use `TfidfVectorizer`
---

### 2. Baseline Classifier with TF-IDF
- [ ] Use `TfidfVectorizer` + `LogisticRegression`
- [ ] Train model to predict the speaker
- [ ] Evaluate accuracy and confusion matrix

> **Why it's interesting:** This shows how distinguishable character voices are just by word use — are Eliza’s lines harder to separate than Burr’s?

---

### 3. Try Other Classical Models
- [ ] Swap in `MultinomialNB`, `RandomForestClassifier`, `SVM`
- [ ] Compare performance

> **Why it's interesting:** See which models handle sparse, high-dimensional text best — useful ML comparison exercise.

---

### 4. Neural Model (LSTM or BiLSTM)
- [ ] Use `Tokenizer` + `pad_sequences` to prepare input
- [ ] Train an LSTM-based classifier using Keras or PyTorch
- [ ] Track training loss + accuracy

> **Why it's interesting:** Neural models can “learn” writing style and sentence structure — deeper representation of how characters speak.

---

### 5. Transformer-Based Model (e.g. BERT)
- [ ] Use HuggingFace `transformers` to fine-tune `bert-base-uncased`
- [ ] Frame it as a text classification task (line → character)
- [ ] Evaluate results and compare to earlier models

> **Why it's interesting:** You’re using state-of-the-art tools on a creative dataset — it's a strong portfolio piece.

---

### 6. Model Explainability
- [ ] Try `LIME` or `SHAP` to explain why the model predicted a certain character
- [ ] Visualize important words per prediction

> **Why it's interesting:** Makes the model feel less like a black box — great way to show what distinguishes characters linguistically.

---

### 7. (Optional) Interactive Demo
- [ ] Use `Streamlit` or `Gradio` to build a web interface
- [ ] User inputs a line, app predicts speaker + confidence + top keywords
- [ ] Add character stats or emotion radar plots

> **Why it's interesting:** Super fun way to present your work — and lets others play with your model.

---

## 🌟 Bonus / Stretch Ideas

- Compare Hamilton to *In The Heights* or *Les Mis* using the same pipeline
- Cluster lines into emotion or topic types *regardless* of speaker
- Detect sarcasm or rhetorical style
- Analyze rhyme/meter patterns with phoneme tools (`pronouncing`, `textstat`)
- Animate character emotion arcs across songs (timeline style)

---
