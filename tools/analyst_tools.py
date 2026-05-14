"""Sentiment analysis using local fine-tuned BERTweet model with GPU support."""

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
print("🔧 SENTIMENT ANALYZER INITIALIZATION")
print("="*70)

# Check GPU availability
if torch.cuda.is_available():
    _device = "cuda"
    print(f"✅ CUDA Available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    _device = "cpu"
    print(f"⚠️  GPU NOT available - using CPU (slower)")
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
        print(f"📦 LOADING FINE-TUNED BERTWEET MODEL")
        print(f"{'='*70}")
        print(f"Model Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"\n❌ Model directory not found: {model_path}")
            print(f"\nTo set up your model:")
            print(f"   1. Save your fine-tuned BERTweet model:")
            print(f"      from transformers import AutoModelForSequenceClassification, AutoTokenizer")
            print(f"      model.save_pretrained('{model_path}')")
            print(f"      tokenizer.save_pretrained('{model_path}')")
            print(f"\n   2. Run diagnostic: python tools/check_model_setup.py")
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for required files
        model_file, tokenizer_files = _find_model_files(model_path)
        
        print(f"\n✓ Checking model files:")
        if model_file:
            print(f"   ✓ Model file found: {os.path.basename(model_file)}")
        else:
            print(f"   ❌ No model file found (pytorch_model.bin, model.bin, or model.safetensors)")
            raise FileNotFoundError(f"No model file found in {model_path}")
        
        if tokenizer_files:
            print(f"   ✓ Tokenizer files found: {len(tokenizer_files)} file(s)")
        else:
            print(f"   ❌ No tokenizer files found")
            raise FileNotFoundError(f"No tokenizer files found in {model_path}")
        
        print(f"\n✓ Loading tokenizer...")
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded: {type(_tokenizer).__name__}")
        
        print(f"✓ Loading model (this may take a moment)...")
        
        # Load model on CPU first to avoid OOM on GPU
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print(f"✓ Model loaded")
        print(f"   Model Type: {_model.config.model_type.upper()}")
        print(f"   Architecture: {type(_model).__name__}")
        print(f"   Num Labels: {_model.config.num_labels}")
        print(f"   Hidden Size: {_model.config.hidden_size}")
        
        # Move model to device (GPU or CPU)
        print(f"\n✓ Moving model to device: {_device.upper()}")
        
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
            print(f"⚠️  GPU error occurred: {str(e)}")
            print(f"   Falling back to CPU")
            _device = "cpu"
            _model = _model.to(_device)
            _model.eval()
        
        print(f"\n✅ Fine-tuned BERTweet model loaded successfully on {_device.upper()}!")
        print(f"{'='*70}\n")
        
        return _model, _tokenizer, _device
    
    except Exception as e:
        print(f"\n❌ FAILED to load model:")
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
        print(f"🔍 ANALYZING SENTIMENT ({len(texts)} texts)")
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
                    
                    # OPTIMIZATION: Store only first 50 chars of text to reduce payload
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
                
                # Clear GPU cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Log progress
                current_batch = (batch_idx // batch_size) + 1
                print(f"   Batch {current_batch}/{total_batches} completed ({len(sentiments)}/{len(texts)} texts)")
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  GPU Out of Memory error in batch {batch_idx//batch_size + 1}")
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
        print(f"❌ Error in sentiment analysis: {str(e)}")

        # Ensure cleanup happens even on error
        try:
            if device == "cuda":
                cleanup_gpu_memory()
        except Exception as cleanup_error:
            print(f"⚠️  Error during cleanup after failure: {str(cleanup_error)}")

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
        
        print(f"\n{'─'*70}")
        print(f"📊 SENTIMENT ANALYSIS RESULTS")
        print(f"{'─'*70}")
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

        print("✅ GPU memory cleaned up")
    except Exception as e:
        print(f"⚠️  Error during GPU cleanup: {str(e)}")

    return {
        "sentiments": sentiments,
        "overall_sentiment": overall,
        "average_confidence": avg_confidence
    }

def prepare_for_aspect_analysis(texts: List[str]) -> Dict:
    """Prepare texts for aspect-level sentiment analysis using spaCy."""
    
    try:
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Installing spaCy model...")
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
        print(f"⚠️ Could not prepare aspect analysis: {str(e)}")
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

            print("✅ GPU memory cleaned up")
        except Exception as e:
            print(f"⚠️  Error during GPU cleanup: {str(e)}")


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
                    print(f"⚠️  Could not move model to CPU: {str(e)}")

            # Delete model reference
            del _model
            _model = None

        if _tokenizer is not None:
            del _tokenizer
            _tokenizer = None

        # Clean up GPU memory
        if _device == "cuda":
            cleanup_gpu_memory()

        print("✅ Model and tokenizer unloaded from memory")

    except Exception as e:
        print(f"⚠️  Error during model reset: {str(e)}")
        _model = None
        _tokenizer = None
        print(f"⚠️  Error cleaning GPU: {str(e)}")