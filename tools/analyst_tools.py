"""Sentiment analysis using local fine-tuned BERTweet model with GPU support."""

import json
import os
import sys
import torch
import warnings
from typing import List, Dict

warnings.filterwarnings('ignore')

# Initialize model globally
_model = None
_tokenizer = None
_device = None

print("\n" + "="*70)
print("[INIT] SENTIMENT ANALYZER INITIALIZATION")
print("="*70)

# Check GPU availability
if torch.cuda.is_available():
    _device = "cuda"
    print(f"[OK] CUDA Available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    _device = "cpu"
    print(f"[WARN] GPU NOT available - using CPU (slower)")
    print(f"   PyTorch Version: {torch.__version__}")
    if "+cpu" in torch.__version__:
        print(f"   Your PyTorch is CPU-only build!")
        print(f"   To use GPU:")
        print(f"      1. Run: python tools/check_model_setup.py")
        print(f"      2. Follow the GPU setup instructions")

print("="*70 + "\n")


def _find_model_files(model_path: str) -> tuple:
    """Find model and tokenizer files with flexible naming."""
    
    if not os.path.exists(model_path):
        return None, None
    
    model_file = None
    tokenizer_files = []
    
    # Look for model files
    possible_model_names = ['pytorch_model.bin', 'model.bin', 'model.safetensors']
    for name in possible_model_names:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            model_file = path
            break
    
    # Look for tokenizer files (at least one of these should exist)
    possible_tokenizer_files = ['tokenizer.json', 'tokenizer.model', 'vocab.txt', 'spm.model']
    for name in possible_tokenizer_files:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            tokenizer_files.append(path)
    
    return model_file, tokenizer_files


def initialize_model():
    """
    Initialize the fine-tuned BERTweet model with GPU support.
    Loads from tools/models/bertweet_finetuned directory.
    """
    
    global _model, _tokenizer, _device
    
    if _model is not None:
        return _model, _tokenizer, _device
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Get absolute path to model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "bertweet_finetuned")
        
        print(f"\n{'='*70}")
        print(f"[LOAD] LOADING FINE-TUNED BERTWEET MODEL")
        print(f"{'='*70}")
        print(f"Model Path: {model_path}")

        if not os.path.exists(model_path):
            print(f"\n[ERROR] Model directory not found: {model_path}")
            print(f"\nTo set up your model:")
            print(f"   1. Save your fine-tuned BERTweet model:")
            print(f"      from transformers import AutoModelForSequenceClassification, AutoTokenizer")
            print(f"      model.save_pretrained('{model_path}')")
            print(f"      tokenizer.save_pretrained('{model_path}')")
            print(f"\n   2. Run diagnostic: python tools/check_model_setup.py")
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for required files
        model_file, tokenizer_files = _find_model_files(model_path)
        
        print(f"\n[..] Checking model files:")
        if model_file:
            print(f"   [OK] Model file found: {os.path.basename(model_file)}")
        else:
            print(f"   [ERROR] No model file found (pytorch_model.bin, model.bin, or model.safetensors)")
            raise FileNotFoundError(f"No model file found in {model_path}")

        if tokenizer_files:
            print(f"   [OK] Tokenizer files found: {len(tokenizer_files)} file(s)")
        else:
            print(f"   [ERROR] No tokenizer files found")
            raise FileNotFoundError(f"No tokenizer files found in {model_path}")

        print(f"\n[..] Loading tokenizer...")

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[OK] Tokenizer loaded: {type(_tokenizer).__name__}")

        print(f"[..] Loading model (this may take a moment)...")

        # Load model on CPU first to avoid OOM on GPU
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print(f"[OK] Model loaded")
        print(f"   Model Type: {_model.config.model_type.upper()}")
        print(f"   Architecture: {type(_model).__name__}")
        print(f"   Num Labels: {_model.config.num_labels}")
        print(f"   Hidden Size: {_model.config.hidden_size}")

        # Move model to device (GPU or CPU)
        print(f"\n[..] Moving model to device: {_device.upper()}")

        try:
            _model = _model.to(_device)

            # Set to evaluation mode
            _model.eval()

            # Log device memory for GPU
            if _device == "cuda":
                torch.cuda.synchronize()  # Ensure operations are complete
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   GPU Memory Allocated: {allocated:.2f} GB")
                print(f"   GPU Memory Reserved: {reserved:.2f} GB")

        except RuntimeError as e:
            print(f"[WARN] GPU error occurred: {str(e)}")
            print(f"   Falling back to CPU")
            _device = "cpu"
            _model = _model.to(_device)
            _model.eval()

        print(f"\n[OK] Fine-tuned BERTweet model loaded successfully on {_device.upper()}!")
        print(f"{'='*70}\n")

        return _model, _tokenizer, _device

    except Exception as e:
        print(f"\n[FAILED] FAILED to load model:")
        print(f"   Error: {str(e)}")
        print(f"\nDiagnostics:")
        print(f"   Run: python tools/check_model_setup.py")
        print(f"{'='*70}\n")
        raise


def analyze_sentiment_batch(texts: List[str], batch_size: int = None) -> Dict:
    """
    Analyze sentiment of multiple texts using fine-tuned BERTweet model on GPU/CPU.
    
    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process at once (auto-optimized if None)
    
    Returns:
        Dictionary with sentiment analysis results
    """
    
    if not texts:
        return {
            "sentiments": [],
            "overall_sentiment": "neutral",
            "average_confidence": 0.0
        }
    
    # Initialize model
    model, tokenizer, device = initialize_model()
    
    # Auto-optimize batch size based on device
    if batch_size is None:
        if device == "cuda":
            batch_size = 8
        else:
            batch_size = 32
    
    sentiments = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    try:
        print(f"\n{'='*70}")
        print(f"[ANALYZING] SENTIMENT ({len(texts)} texts)")
        print(f"{'='*70}")
        print(f"Device: {device.upper()}")
        print(f"Batch Size: {batch_size}")
        print(f"Total Batches: {total_batches}")
        
        # Process in batches
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx+batch_size]
            
            try:
                # Tokenize batch
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                
                # Label mapping
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
                
                # Process results - REDUCE TEXT SIZE to save memory
                for j, (text, pred, prob) in enumerate(zip(batch, predictions.cpu().numpy(), probs.cpu().numpy())):
                    pred_idx = int(pred)
                    label = label_map.get(pred_idx, "neutral")
                    confidence = float(prob[pred_idx])
                    
                    sentiments.append({
                        "text": text,
                        "label": label,
                        "confidence": confidence,
                        "scores": {
                            "negative": float(prob[0]),
                            "neutral": float(prob[1]) if len(prob) > 1 else 0.0,
                            "positive": float(prob[2]) if len(prob) > 2 else 0.0
                        }
                    })
                
                # Clear GPU cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Log progress
                current_batch = (batch_idx // batch_size) + 1
                print(f"   Batch {current_batch}/{total_batches} completed ({len(sentiments)}/{len(texts)} texts)")
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] GPU Out of Memory error in batch {batch_idx//batch_size + 1}")
                    print(f"   Reducing batch size and retrying...")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Retry with smaller batch
                    for text_idx in range(0, len(batch), max(1, batch_size // 2)):
                        small_batch = batch[text_idx:text_idx + max(1, batch_size // 2)]
                        
                        inputs = tokenizer(
                            small_batch,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        )
                        
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=-1)
                            predictions = torch.argmax(logits, dim=-1)
                        
                        for j, (text, pred, prob) in enumerate(zip(small_batch, predictions.cpu().numpy(), probs.cpu().numpy())):
                            pred_idx = int(pred)
                            label = label_map.get(pred_idx, "neutral")
                            confidence = float(prob[pred_idx])
                            
                            text_preview = text[:50] if len(text) > 50 else text
                            
                            sentiments.append({
                                "text": text_preview,
                                "label": label,
                                "confidence": confidence,
                                "scores": {
                                    "negative": float(prob[0]),
                                    "neutral": float(prob[1]) if len(prob) > 1 else 0.0,
                                    "positive": float(prob[2]) if len(prob) > 2 else 0.0
                                }
                            })
                        
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    
                    current_batch = (batch_idx // batch_size) + 1
                    print(f"   Batch {current_batch}/{total_batches} completed with reduced size ({len(sentiments)}/{len(texts)} texts)")
                else:
                    raise
    
    except Exception as e:
        print(f"[ERROR] Error in sentiment analysis: {str(e)}")

        # Ensure cleanup happens even on error
        try:
            if device == "cuda":
                cleanup_gpu_memory()
        except Exception as cleanup_error:
            print(f"[WARN] Error during cleanup after failure: {str(cleanup_error)}")

        raise
    
    # Calculate overall sentiment
    if sentiments:
        positive_count = sum(1 for s in sentiments if s["label"] == "positive")
        negative_count = sum(1 for s in sentiments if s["label"] == "negative")
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total = len(sentiments)
        avg_confidence = sum(s["confidence"] for s in sentiments) / total if total > 0 else 0
        
        # Determine overall sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            overall = "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall = "negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            overall = "neutral"
        else:
            overall = "positive" if positive_count >= negative_count else "negative"
        
        print(f"\n{'='*70}")
        print(f"[RESULTS] SENTIMENT ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(f"Total Texts Analyzed: {total}")
        print(f"Positive: {positive_count:,} ({(positive_count/total*100):.1f}%)")
        print(f"Neutral: {neutral_count:,} ({(neutral_count/total*100):.1f}%)")
        print(f"Negative: {negative_count:,} ({(negative_count/total*100):.1f}%)")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print(f"Overall Sentiment: {overall.upper()}")
        print(f"Device Used: {device.upper()}")
        print(f"{'='*70}\n")
    else:
        overall = "neutral"
        avg_confidence = 0.0

    # CLEANUP: Clean up GPU memory before returning results
    try:
        # Add synchronization barriers to ensure all GPU operations complete
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[OK] GPU memory cleaned up")
    except Exception as e:
        print(f"[WARN] Error during GPU cleanup: {str(e)}")

    return {
        "sentiments": sentiments,
        "overall_sentiment": overall,
        "average_confidence": avg_confidence
    }

def discover_aspects_kmeans_ctfidf(texts: List[str], sentiments: List[Dict]) -> Dict:
    """
    Unsupervised ABSA pipeline (ABSA.md specification).

    Phase 1  – Clean text: strip r/, u/, URLs, drop < 4-token entries.
    Phase 2  – Sentence embeddings: all-MiniLM-L6-v2 (384D).
               Falls back to TF-IDF vectors if sentence-transformers missing.
    Phase 3  – PCA dimensionality reduction to 5 components.
    Phase 4  – Silhouette-optimal K-Means, K ∈ {2 … 6}.
    Phase 5  – Cluster class compaction (concatenate texts per cluster).
    Phase 6  – c-TF-IDF on macro-class documents (sklearn TfidfVectorizer).
    Phase 7  – Top-2 noun tokens per cluster → "Token1 & Token2" label.
               Aggregate BERTweet sentiment counts per aspect label.

    Args:
        texts:      Cleaned sentiment texts (parallel to `sentiments`).
        sentiments: List of {"label": "positive"|"neutral"|"negative", ...}

    Returns:
        {"Aspect Label": {
            "positive"      : {"count": N, "percentage": X},
            "neutral"       : {"count": N, "percentage": X},
            "negative"      : {"count": N, "percentage": X},
            "total_mentions": N,
        }, ...}
    """
    import re

    # ── Phase 1: clean + filter ───────────────────────────────────────────────
    def _clean(t: str) -> str:
        t = re.sub(r'r/\w+', '', t)
        t = re.sub(r'u/\w+', '', t)
        t = re.sub(r'https?\S+|www\.\S+', '', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    clean_texts, clean_sents = [], []
    for raw, sent in zip(texts, sentiments):
        c = _clean(raw)
        if len(c.split()) >= 4:
            clean_texts.append(c)
            clean_sents.append(sent)

    MIN_DOCS = 6
    if len(clean_texts) < MIN_DOCS:
        print(f"  [ABSA] Only {len(clean_texts)} usable texts (need {MIN_DOCS}) - skipping")
        return {}

    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.feature_extraction.text import TfidfVectorizer

        # ── Phase 2: sentence embeddings ─────────────────────────────────────
        # Force CPU: BERTweet already occupies GPU VRAM for sentiment analysis;
        # competing for the same device causes OOM and corrupts the CUDA context.
        try:
            from sentence_transformers import SentenceTransformer
            print("  [ABSA] Encoding with all-MiniLM-L6-v2 (CPU) ...")
            embeddings = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(
                clean_texts, show_progress_bar=False, batch_size=32
            )
        except ImportError:
            print("  [ABSA] sentence-transformers not installed - using TF-IDF fallback")
            embeddings = TfidfVectorizer(max_features=384, sublinear_tf=True) \
                             .fit_transform(clean_texts).toarray()

        # ── Phase 3: PCA → 5D ────────────────────────────────────────────────
        n_comp = min(5, len(clean_texts) - 1, embeddings.shape[1])
        reduced = PCA(n_components=n_comp, random_state=42).fit_transform(embeddings)

        # ── Phase 4: silhouette-optimal K-Means ──────────────────────────────
        best_k, best_sil = 2, -1.0
        for k in range(2, min(7, len(clean_texts))):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced)
            if len(set(lbl)) < 2:
                continue
            sil = silhouette_score(reduced, lbl)
            print(f"  [ABSA] K={k}  silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil, best_k = sil, k

        print(f"  [ABSA] Optimal K={best_k} (silhouette={best_sil:.4f})")
        cluster_ids = KMeans(n_clusters=best_k, random_state=42, n_init=10) \
                          .fit_predict(reduced)

        # ── Phase 5: cluster compaction ───────────────────────────────────────
        cluster_docs: Dict = {k: [] for k in range(best_k)}
        for idx, cid in enumerate(cluster_ids):
            cluster_docs[cid].append(clean_texts[idx])
        macro_docs = [" ".join(cluster_docs[k]) for k in range(best_k)]

        # ── Phase 6: c-TF-IDF ────────────────────────────────────────────────
        tfidf_mat  = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 1),
            token_pattern=r"[a-zA-Z]{3,}",
            sublinear_tf=True,
        ).fit_transform(macro_docs)
        feat_names = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 1),
            token_pattern=r"[a-zA-Z]{3,}",
            sublinear_tf=True,
        ).fit(macro_docs).get_feature_names_out()

        # Re-fit once to share the vocabulary
        cv = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 1),
            token_pattern=r"[a-zA-Z]{3,}",
            sublinear_tf=True,
        )
        tfidf_mat  = cv.fit_transform(macro_docs)
        feat_names = cv.get_feature_names_out()

        # ── Phase 7: noun-filtered top-2 tokens → aspect label ───────────────
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                               check=False, capture_output=True)
                _nlp = spacy.load("en_core_web_sm")

            def _prefer_nouns(tokens: list) -> list:
                nouns = [t for t in tokens if _nlp(t)[0].pos_ in ("NOUN", "PROPN")]
                return nouns if nouns else tokens
        except Exception:
            def _prefer_nouns(tokens: list) -> list:
                return tokens

        aspect_analysis: Dict = {}
        for cid in range(best_k):
            row        = tfidf_mat[cid].toarray().flatten()
            candidates = [feat_names[i] for i in row.argsort()[::-1] if row[i] > 0][:20]
            top2       = _prefer_nouns(candidates)[:2]

            label = (f"{top2[0].title()} & {top2[1].title()}" if len(top2) >= 2
                     else top2[0].title() if top2
                     else f"Cluster {cid}")

            indices = [i for i, l in enumerate(cluster_ids) if l == cid]
            pos   = sum(1 for i in indices if clean_sents[i].get("label") == "positive")
            neu   = sum(1 for i in indices if clean_sents[i].get("label") == "neutral")
            neg   = sum(1 for i in indices if clean_sents[i].get("label") == "negative")
            total = len(indices)

            aspect_analysis[label] = {
                "positive"      : {"count": pos, "percentage": (pos / total * 100) if total else 0},
                "neutral"       : {"count": neu, "percentage": (neu / total * 100) if total else 0},
                "negative"      : {"count": neg, "percentage": (neg / total * 100) if total else 0},
                "total_mentions": total,
            }
            print(f"  [ABSA] Cluster {cid} -> '{label}'  ({total} comments)")

        return dict(sorted(aspect_analysis.items(), key=lambda x: -x[1]["total_mentions"]))

    except Exception as exc:
        import traceback
        print(f"[WARN] ABSA clustering failed: {exc}")
        print(traceback.format_exc())
        return {}


def discover_aspects_bertopic(texts: List[str], sentiments: List[Dict]) -> Dict:
    """
    BERTopic ABSA pipeline: all-MiniLM-L6-v2 embeddings → UMAP → HDBSCAN.

    HDBSCAN labels documents that don't fit any cluster as topic -1.  These are
    collected into an "Others" aspect so they remain visible in ABSA visuals.

    Sentiment labels for each comment come from the fine-tuned BERTweet model
    (the `sentiments` parameter is the output of analyze_sentiment_batch()).
    BERTopic is used ONLY for clustering — it never re-predicts sentiment.

    Each aspect in the returned dict includes a private "_top_words" key that
    verify_aspects_with_llm() uses for better LLM prompting.  That key is
    stripped before the dict is stored / displayed.

    Falls back to discover_aspects_kmeans_ctfidf() if bertopic / umap-learn /
    hdbscan are not installed.

    Args:
        texts:      Raw sentiment texts (parallel to `sentiments`).
        sentiments: List of {"label": "positive"|"neutral"|"negative", ...}

    Returns:
        {"Aspect Label": {
            "positive"      : {"count": N, "percentage": X},
            "neutral"       : {"count": N, "percentage": X},
            "negative"      : {"count": N, "percentage": X},
            "total_mentions": N,
            "_top_words"    : [str, ...]   # stripped by verify_aspects_with_llm
        }, ...}
        "Others" is always last (if HDBSCAN produced noise points).
    """
    import re

    # Extra stopwords for social-media / Reddit noise not caught by sklearn's
    # built-in English list (URLs already stripped by regex below).
    _SM_STOP = {
        "httpurl", "http", "https", "www", "com", "org", "amp", "gt", "lt",
        "reddit", "subreddit", "post", "comment", "edit", "upvote", "downvote",
        "like", "just", "get", "got", "would", "could", "should", "also",
        "really", "think", "know", "make", "good", "one", "even", "still",
        "thing", "things", "way", "use", "used", "lot", "much", "many",
        "something", "someone", "anyone", "anything", "everything",
        "people", "person", "everyone", "well", "need", "want", "see",
        "new", "look", "going", "come", "time", "year", "day", "back",
        "right", "say", "said", "dont", "doesnt", "didnt", "cant", "wont",
        "ive", "im", "youre", "theyre", "thats", "heres", "whats", "yeah",
        "yes", "no", "ok", "okay", "oh", "ah", "uh", "lol", "lmao",
    }

    def _is_content_keyword(w: str) -> bool:
        """Return True for a meaningful unigram or bigram BERTopic keyword."""
        parts = w.split()
        if not parts:
            return False
        # All parts must be alphabetic and not in our stop sets
        return (
            all(p.isalpha() and len(p) > 2 and p.lower() not in _SM_STOP
                for p in parts)
        )

    # ── Phase 1: clean + filter ───────────────────────────────────────────────
    def _clean(t: str) -> str:
        t = re.sub(r'r/\w+', '', t)
        t = re.sub(r'u/\w+', '', t)
        t = re.sub(r'https?\S+|www\.\S+', '', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        # Remove standalone digits (version numbers kept via context)
        t = re.sub(r'\b\d+\b', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    clean_texts, clean_sents = [], []
    for raw, sent in zip(texts, sentiments):
        c = _clean(raw)
        if len(c.split()) >= 4:
            clean_texts.append(c)
            clean_sents.append(sent)

    MIN_DOCS = 10
    if len(clean_texts) < MIN_DOCS:
        print(f"  [ABSA] Only {len(clean_texts)} usable texts (need {MIN_DOCS}) - skipping")
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from collections import defaultdict

        # ── Phase 2: embed with all-MiniLM-L6-v2 (CPU) ──────────────────────
        # Force CPU: BERTweet already occupies GPU VRAM; sharing it causes OOM
        # which corrupts the CUDA context and breaks the K-Means fallback too.
        print(f"  [ABSA-BERTopic] Encoding {len(clean_texts)} docs with all-MiniLM-L6-v2 (CPU)...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = embedding_model.encode(
            clean_texts, show_progress_bar=False, batch_size=32
        )

        # ── Phase 3+4: UMAP → HDBSCAN via BERTopic ───────────────────────────
        n = len(clean_texts)
        min_cluster_size = max(5, min(20, n // 20))

        umap_model = UMAP(
            n_neighbors=min(15, n - 1),
            n_components=5,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # CountVectorizer with English stopwords prevents function/stopwords
        # from dominating the c-TF-IDF topic representations.
        # ngram_range=(1,2) allows bigrams like "battery life", "camera quality".
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2),
            max_features=5000,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=False,
        )

        print(f"  [ABSA-BERTopic] Fitting (min_cluster_size={min_cluster_size})...")
        topics, _ = topic_model.fit_transform(clean_texts, embeddings=embeddings)

        # ── Phase 5: build raw labels with top-keyword metadata ───────────────
        cluster_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, tid in enumerate(topics):
            cluster_indices[tid].append(idx)

        # Build mapping: topic_id → (label_string, top_words_list)
        topic_labels: Dict[int, tuple] = {}
        for tid in sorted(cluster_indices.keys()):
            if tid == -1:
                topic_labels[-1] = ("Others", [])
                continue
            word_score_pairs = topic_model.get_topic(tid) or []
            # Filter to meaningful content words / bigrams only
            words = [
                w for w, _ in word_score_pairs
                if isinstance(w, str) and _is_content_keyword(w)
            ][:8]
            # Use best keyword as the raw label (LLM will rename it anyway)
            label = words[0].title() if words else f"Topic {tid}"
            topic_labels[tid] = (label, words)

        # ── Phase 6: aggregate BERTweet sentiment counts per cluster ──────────
        aspect_analysis: Dict = {}
        for tid, indices in cluster_indices.items():
            label, top_words = topic_labels.get(tid, (f"Topic {tid}", []))
            pos   = sum(1 for i in indices if clean_sents[i].get("label") == "positive")
            neu   = sum(1 for i in indices if clean_sents[i].get("label") == "neutral")
            neg   = sum(1 for i in indices if clean_sents[i].get("label") == "negative")
            total = len(indices)
            if total == 0:
                continue
            aspect_analysis[label] = {
                "positive"      : {"count": pos, "percentage": pos / total * 100},
                "neutral"       : {"count": neu, "percentage": neu / total * 100},
                "negative"      : {"count": neg, "percentage": neg / total * 100},
                "total_mentions": total,
                "_top_words"    : top_words,
            }
            print(f"  [ABSA-BERTopic] Cluster {tid} -> '{label}'  "
                  f"({total} docs, top: {top_words[:4]})")

        # Sort named aspects by mention count; keep Others at the end
        others_data = aspect_analysis.pop("Others", None)
        sorted_aspects = dict(
            sorted(aspect_analysis.items(), key=lambda x: -x[1]["total_mentions"])
        )
        if others_data:
            sorted_aspects["Others"] = others_data

        n_named = len(sorted_aspects) - (1 if others_data else 0)
        print(f"  [ABSA-BERTopic] Done: {n_named} topic clusters + "
              f"{'Others (' + str(others_data['total_mentions']) + ' docs)' if others_data else 'no noise'}")
        return sorted_aspects

    except ImportError as ie:
        print(f"  [ABSA-BERTopic] Missing package: {ie}")
        print(f"  [ABSA-BERTopic] Install with: pip install bertopic umap-learn hdbscan")
        print(f"  [ABSA-BERTopic] Falling back to K-Means + c-TF-IDF...")
        return discover_aspects_kmeans_ctfidf(texts, sentiments)

    except Exception as exc:
        import traceback
        print(f"  [ABSA-BERTopic] Failed: {exc}")
        print(traceback.format_exc())
        # Free any GPU memory held by the failed BERTopic run before falling back
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  [ABSA-BERTopic] Falling back to K-Means + c-TF-IDF...")
        return discover_aspects_kmeans_ctfidf(texts, sentiments)


def verify_aspects_with_llm(
    raw_aspects     : Dict,
    topic           : str,
    llm,
    category        : str = "",
    category_detail : str = "",
) -> Dict:
    """
    LLM verification layer for ABSA aspect labels.

    Works with both BERTopic output (which includes "_top_words" per aspect)
    and the legacy K-Means output (no "_top_words").

    Behaviour:
      - "Others" is always preserved as-is and is never sent to the LLM for
        renaming, so uncategorised HDBSCAN noise is always visible in visuals.
      - For each remaining label the LLM either polishes it into a concise
        1-3 word Title Case phrase (no "&") or marks it for removal (null).
        When BERTopic top-keywords are present they are included in the prompt
        so the LLM has richer context.
      - Duplicate polished labels are merged (sentiment counts are summed).
      - "_top_words" is stripped from every entry before returning.

    Args:
        raw_aspects: Dict returned by discover_aspects_bertopic() or
                     discover_aspects_kmeans_ctfidf().
        topic:       The original user topic string (e.g. "iPhone 17 Pro Max").
        llm:         Initialised LangChain LLM instance.

    Returns:
        Cleaned aspect dict.  Falls back to raw_aspects (minus "_top_words")
        if the LLM call fails.  "Others" is always last.
    """
    if not raw_aspects or not topic:
        return _strip_top_words(raw_aspects)

    # ── Separate "Others" — never sent to LLM ────────────────────────────────
    others_data = raw_aspects.pop("Others", None)

    if not raw_aspects:
        result = {}
        if others_data:
            others_clean = {k: v for k, v in others_data.items() if k != "_top_words"}
            result["Others"] = others_clean
        return result

    raw_labels = list(raw_aspects.keys())

    # ── Build label list with top-keywords for the LLM prompt ────────────────
    label_lines = []
    for label in raw_labels:
        words = raw_aspects[label].get("_top_words", [])
        if words:
            label_lines.append(f'  "{label}"  (keywords: {", ".join(words[:6])})')
        else:
            label_lines.append(f'  "{label}"')

    # Build optional topic-type context line for the prompt
    _type_line = ""
    if category:
        _type_line = f"\nTopic type: {category}"
        if category_detail:
            _type_line += f" ({category_detail})"

    prompt = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA).

Topic / Product being analysed: "{topic}"{_type_line}

The following aspect category labels were produced automatically by BERTopic
(UMAP + HDBSCAN) applied to Reddit comments about this topic.
Each label is followed by its top representative keywords from the cluster:

{chr(10).join(label_lines)}

Your job — for each label decide ONE of two actions:

A) KEEP & POLISH — the label represents a real, meaningful aspect of "{topic}".
   Use the keywords AND the topic type to identify the single most specific,
   relevant concept users are discussing.  Rewrite as ONE short phrase of 1-3
   words in Title Case — NO ampersand (&), NO slash, NO comma.
   Choose aspect names that make sense for the topic type.  Examples:
     • Consumer Electronics / Phone  → "Battery Life", "Camera Quality", "Display", "Performance", "Pricing", "Build Quality"
     • Software / App / Game         → "User Interface", "Performance", "Bugs", "Pricing", "Content", "Updates"
     • Person / Celebrity            → "Music", "Live Shows", "Personality", "Controversies", "Collaborations"
     • Movie / TV Show               → "Plot", "Acting", "Visual Effects", "Characters", "Soundtrack", "Pacing"
     • Company / Brand               → "Customer Service", "Product Quality", "Pricing", "Brand Image", "Shipping"

B) REMOVE — the cluster is noise: grammatical fragments, off-topic chatter,
   or too generic to be a useful product/service aspect.  Return null.

Return ONLY a valid JSON object mapping every original label to its polished
string or null:
{{
  "Original Label 1": "Polished Label",
  "Original Label 2": null,
  ...
}}

Rules:
- Map EVERY label from the input list — do not skip any.
- Two different labels may map to the same polished label (they are merged).
- Be conservative: keep aspects even if uncertain; only remove clear noise.
- Output ONLY the JSON object — no explanation, no markdown."""

    try:
        print(f"  [ABSA-LLM] Verifying {len(raw_labels)} labels for '{topic}'...")
        response = llm.invoke(prompt)
        text     = response.content if hasattr(response, "content") else str(response)

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            print("  [ABSA-LLM] Could not parse JSON - keeping raw aspects")
            result = _strip_top_words(raw_aspects)
            if others_data:
                result["Others"] = {k: v for k, v in others_data.items() if k != "_top_words"}
            return result

        mapping: dict = json.loads(text[start:end])

        # Rebuild dict with polished labels; merge duplicates
        verified: Dict = {}
        for old_label, new_label in mapping.items():
            if old_label not in raw_aspects or new_label is None:
                continue
            data = raw_aspects[old_label]
            if new_label in verified:
                for sentiment in ("positive", "neutral", "negative"):
                    verified[new_label][sentiment]["count"] += data[sentiment]["count"]
                verified[new_label]["total_mentions"] += data["total_mentions"]
            else:
                verified[new_label] = {
                    "positive"      : {"count": data["positive"]["count"],      "percentage": 0.0},
                    "neutral"       : {"count": data["neutral"]["count"],       "percentage": 0.0},
                    "negative"      : {"count": data["negative"]["count"],      "percentage": 0.0},
                    "total_mentions": data["total_mentions"],
                }

        # Recalculate percentages after merges
        for _, data in verified.items():
            total = data["total_mentions"]
            if total > 0:
                for sentiment in ("positive", "neutral", "negative"):
                    data[sentiment]["percentage"] = data[sentiment]["count"] / total * 100

        # Sort named aspects, then re-add Others at the end
        result = dict(sorted(verified.items(), key=lambda x: -x[1]["total_mentions"]))
        if others_data:
            result["Others"] = {k: v for k, v in others_data.items() if k != "_top_words"}

        print(f"  [ABSA-LLM] {len(raw_labels)} raw -> {len(result)} final aspects: "
              f"{list(result.keys())}")
        return result

    except Exception as exc:
        print(f"  [ABSA-LLM] Verification failed ({str(exc)[:100]}) - keeping raw aspects")
        result = _strip_top_words(raw_aspects)
        if others_data:
            result["Others"] = {k: v for k, v in others_data.items() if k != "_top_words"}
        return result


def _strip_top_words(aspects: Dict) -> Dict:
    """Remove the private '_top_words' key from every aspect entry."""
    return {
        label: {k: v for k, v in data.items() if k != "_top_words"}
        for label, data in aspects.items()
    }


def prepare_for_aspect_analysis(texts: List[str]) -> Dict:
    """Prepare texts for aspect-level sentiment analysis using spaCy."""
    
    try:
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[..] Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         check=False, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
        
        aspects_data = {}
        
        for text in texts:
            if not text:
                continue
            
            doc = nlp(text)
            
            for token in doc:
                if token.pos_ == "NOUN":
                    aspect = token.text.lower()
                    if aspect not in aspects_data:
                        aspects_data[aspect] = []
                    aspects_data[aspect].append(text)
        
        return {
            "aspects": list(aspects_data.keys()),
            "aspect_texts": aspects_data,
            "total_aspects": len(aspects_data)
        }
    
    except Exception as e:
        print(f"[WARN] Could not prepare aspect analysis: {str(e)}")
        return {
            "aspects": [],
            "aspect_texts": {},
            "total_aspects": 0
        }


def get_device_info() -> Dict[str, str]:
    """Get information about the current device being used."""
    
    info = {
        "device": _device.upper() if _device else "NOT INITIALIZED",
        "pytorch_version": torch.__version__
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    
    return info

def cleanup_gpu_memory():
    """Clean up GPU memory after analysis."""
    global _model, _tokenizer, _device

    if _device == "cuda":
        try:
            # Ensure all GPU operations are complete
            torch.cuda.synchronize()

            # Clear GPU cache multiple times to ensure complete cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            # Final synchronization
            torch.cuda.synchronize()

            print("[OK] GPU memory cleaned up")
        except Exception as e:
            print(f"[WARN] Error during GPU cleanup: {str(e)}")


def reset_model():
    """
    Completely reset and unload the model from GPU/CPU.
    Call this after sentiment analysis is complete to free all resources.
    """
    global _model, _tokenizer, _device

    try:
        if _model is not None:
            # Move model back to CPU if on GPU
            if _device == "cuda":
                try:
                    _model = _model.cpu()
                except Exception as e:
                    print(f"[WARN] Could not move model to CPU: {str(e)}")

            # Delete model reference
            del _model
            _model = None

        if _tokenizer is not None:
            del _tokenizer
            _tokenizer = None

        # Clean up GPU memory
        if _device == "cuda":
            cleanup_gpu_memory()

        print("[OK] Model and tokenizer unloaded from memory")

    except Exception as e:
        print(f"[WARN] Error during model reset: {str(e)}")
        _model = None
        _tokenizer = None
        print(f"[WARN] Error cleaning GPU: {str(e)}")