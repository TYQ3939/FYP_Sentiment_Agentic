"""Data processing tools for cleaning and preparing data."""

import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                   check=False, capture_output=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        print("⚠️ Could not load spaCy model, will use fallback")

# Extended stopwords list for better filtering
EXTENDED_STOPWORDS = set(stopwords.words('english')).union({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'i', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
    'to', 'was', 'will', 'with', 'lot', 'both', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'ten', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'do', 'does', 'did', 'have', 'had',
    'this', 'that', 'these', 'those', 'such', 'very', 'just', 'more', 'most',
    'also', 'so', 'all', 'each', 'every', 'any', 'some', 'no', 'not', 'only',
    'own', 'same', 'then', 'than', 'too', 'up', 'out', 'now', 'down', 'off',
    'over', 'under', 'through', 'during', 'before', 'after', 'above', 'below',
    'me', 'him', 'her', 'us', 'them', 'you', 'my', 'your', 'our', 'their'
})

# Technology-related aspects for ABSA
TECH_ASPECTS = {
    'camera', 'photo', 'picture', 'video', 'display', 'screen', 'battery',
    'charge', 'charging', 'processor', 'cpu', 'gpu', 'performance', 'speed',
    'memory', 'ram', 'storage', 'design', 'build', 'material', 'quality',
    'price', 'cost', 'value', 'warranty', 'support', 'software', 'app',
    'interface', 'ui', 'ux', 'feature', 'network', 'connectivity',
    '5g', 'wifi', 'bluetooth', 'speaker', 'audio', 'sound', 'microphone',
    'mic', 'keyboard', 'trackpad', 'mouse', 'touchpad', 'resolution',
    'brightness', 'color', 'refresh', 'hz', 'fps', 'frame',
    'weight', 'size', 'dimension', 'durability', 'water', 'resistant',
    'fingerprint', 'face', 'recognition', 'security', 'encryption', 'privacy',
    'life', 'longevity', 'reliability', 'issue', 'problem', 'bug',
    'crash', 'lag', 'delay', 'response', 'heating', 'heat', 'temperature',
    'fan', 'cooling', 'ventilation', 'noise', 'quiet', 'silent', 'hinge',
    'fold', 'flex', 'scratch', 'drop', 'break', 'repair',
    'maintenance', 'upgrade', 'customization', 'expansion', 'port', 'connector',
    'cable', 'charger', 'adapter', 'accessory', 'availability', 'stock',
    'shipping', 'delivery', 'customer', 'service', 'experience'
}


def generate_topic_stopwords_with_llm(topic: str, llm) -> set:
    """
    Use LLM to generate topic-specific stopwords that are semantically related to the topic.
    
    Args:
        topic: The main topic being analyzed (e.g., "iphone 17 pro max")
        llm: The LLM instance to use
    
    Returns:
        Set of topic-specific stopwords
    """
    
    try:
        prompt = f"""You are analyzing sentiment about: "{topic}"

For wordcloud visualization, we need to exclude words that are too closely related to the main topic itself.

Generate a JSON list of words that should be excluded because they're semantically close to the main topic.

Example: for "iPhone 17 Pro Max", exclude: ["iphone", "phone", "apple", "pro", "max", "model", "device"]

Return ONLY a JSON array of lowercase words:
["word1", "word2", "word3", ...]"""
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON array
        import json
        
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            stopwords_list = json.loads(json_match.group())
            return set(w.lower() for w in stopwords_list if isinstance(w, str))
    
    except Exception as e:
        print(f"⚠️ LLM stopword generation failed: {str(e)[:100]}")
    
    return set()


def preprocess_wordcloud_with_pos_tagging(texts: List[str], topic: str = "", llm=None) -> List[str]:
    """
    Advanced preprocessing for wordcloud using POS tagging and lemmatization.
    
    Steps:
    1. Keep only NOUNS, PROPER NOUNS, and ADJECTIVES
    2. Remove VERBS, PRONOUNS, ADVERBS, etc.
    3. Apply lemmatization (plurals → singular)
    4. Use LLM-generated topic-specific stopwords
    5. Filter short words
    
    Args:
        texts: List of raw text strings
        topic: Topic for topic-specific stopword generation
        llm: Optional LLM instance for better stopword generation
    
    Returns:
        List of preprocessed texts for wordcloud
    """
    
    # If spaCy not available, use fallback
    if nlp is None:
        return _fallback_wordcloud_preprocessing(texts, topic)
    
    # Get topic-specific stopwords
    topic_stopwords = set()
    if llm and topic:
        try:
            topic_stopwords = generate_topic_stopwords_with_llm(topic, llm)
            if topic_stopwords:
                print(f"Generated {len(topic_stopwords)} topic-specific stopwords")
        except:
            pass
    
    # Base stopwords
    base_stopwords = set(stopwords.words('english'))
    all_stopwords = base_stopwords | topic_stopwords
    
    processed = []
    
    for text in texts:
        if not text or len(text.strip()) < 5:
            continue
        
        try:
            # Process with spaCy
            doc = nlp(text.lower())
            
            # Keep only meaningful POS tags
            # NOUN = regular noun, PROPN = proper noun, ADJ = adjective
            meaningful_tokens = []
            
            for token in doc:
                # Skip if stopword
                if token.text in all_stopwords:
                    continue
                
                # Skip if too short
                if len(token.text) <= 2:
                    continue
                
                # Keep only NOUN, PROPN, ADJ
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    # Use lemma (singular form)
                    lemma = token.lemma_
                    if lemma and len(lemma) > 2 and lemma not in all_stopwords:
                        meaningful_tokens.append(lemma)
            
            # Reconstruct text from meaningful tokens
            if meaningful_tokens:
                processed_text = " ".join(meaningful_tokens)
                if len(processed_text.split()) >= 2:  # At least 2 meaningful words
                    processed.append(processed_text)
        
        except Exception as e:
            print(f"⚠️ Error processing text with spaCy: {str(e)[:50]}")
            continue
    
    return processed


def _fallback_wordcloud_preprocessing(texts: List[str], topic: str) -> List[str]:
    """Fallback preprocessing without spaCy (basic approach)."""
    
    base_stopwords = set(stopwords.words('english'))
    
    # Add topic words as stopwords
    topic_words = set(topic.lower().split())
    all_stopwords = base_stopwords | topic_words
    
    processed = []
    
    for text in texts:
        if not text or len(text.strip()) < 5:
            continue
        
        # Clean and lowercase
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split and filter
        words = text.split()
        filtered = [
            w for w in words
            if w not in all_stopwords and len(w) > 2 and w.isalpha()
        ]
        
        if len(filtered) >= 2:
            processed.append(" ".join(filtered))
    
    return processed


def clean_text(text: str) -> str:
    """Clean text by removing URLs, mentions, and special characters."""
    
    if not text or text in ["[removed]", "[deleted]", ""]:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the words)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text: str, custom_stopwords: set = None) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Text to process
        custom_stopwords: Optional custom stopwords set (overrides default)
    
    Returns:
        Text without stopwords
    """
    
    if custom_stopwords is None:
        custom_stopwords = EXTENDED_STOPWORDS
    
    words = text.split()
    filtered_words = [w for w in words if w not in custom_stopwords and len(w) > 2]
    return ' '.join(filtered_words)


def filter_meaningful_words(text: str) -> str:
    """Filter out meaningless/short words for wordcloud."""
    
    words = text.split()
    meaningful_words = [
        w for w in words 
        if len(w) > 2 and not w.isdigit() and w.isalpha()
    ]
    return ' '.join(meaningful_words)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def normalize_text(text: str) -> str:
    """Normalize text (remove duplicates, standardize spacing)."""
    
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    return text.strip()


def preprocess_wordcloud_with_pos_tagging(texts: List[str], topic: str = "", llm=None) -> List[str]:
    """
    Advanced preprocessing for wordcloud using POS tagging and lemmatization.
    
    Steps:
    1. Keep only NOUNS, PROPER NOUNS, and ADJECTIVES
    2. Remove VERBS, PRONOUNS, ADVERBS, etc.
    3. Apply lemmatization (plurals → singular)
    4. Use LLM-generated topic-specific stopwords
    5. Filter short words and noise like "s", "ve"
    
    Args:
        texts: List of raw text strings
        topic: Topic for topic-specific stopword generation
        llm: Optional LLM instance for better stopword generation
    
    Returns:
        List of preprocessed texts for wordcloud
    """
    
    # If spaCy not available, use fallback
    if nlp is None:
        return _fallback_wordcloud_preprocessing(texts, topic)
    
    # Get topic-specific stopwords from LLM
    topic_stopwords = set()
    if llm and topic:
        try:
            topic_stopwords = generate_topic_stopwords_with_llm(topic, llm)
            if topic_stopwords:
                print(f"Generated {len(topic_stopwords)} topic-specific stopwords from LLM")
        except Exception as e:
            print(f"⚠️ LLM stopword generation failed: {str(e)[:80]}, using fallback")
            # Fallback: use basic topic words
            topic_stopwords = set(topic.lower().split())
    else:
        # If no LLM, at least use topic words as stopwords
        if topic:
            topic_stopwords = set(topic.lower().split())
    
    # Base stopwords
    base_stopwords = set(stopwords.words('english'))
    
    # Add noise words that slip through (contractions, fragments)
    noise_words = {
        's', 've', 'd', 'll', 're', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'shall', 'ml', 'gb', 'gb', 'mb'  # Common abbreviations
    }
    
    all_stopwords = base_stopwords | topic_stopwords | noise_words
    
    processed = []
    
    for text in texts:
        if not text or len(text.strip()) < 5:
            continue
        
        try:
            # Process with spaCy
            doc = nlp(text.lower())
            
            # Keep only meaningful POS tags
            # NOUN = regular noun, PROPN = proper noun, ADJ = adjective
            meaningful_tokens = []
            
            for token in doc:
                # Skip if stopword
                if token.text in all_stopwords:
                    continue
                
                # Skip if too short (must be > 2 chars)
                if len(token.text) <= 2:
                    continue
                
                # Skip if only punctuation or numbers
                if not token.is_alpha:
                    continue
                
                # Keep only NOUN, PROPN, ADJ
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    # Use lemma (singular form)
                    lemma = token.lemma_
                    if lemma and len(lemma) > 2 and lemma not in all_stopwords:
                        meaningful_tokens.append(lemma)
            
            # Reconstruct text from meaningful tokens
            if meaningful_tokens:
                processed_text = " ".join(meaningful_tokens)
                if len(processed_text.split()) >= 2:  # At least 2 meaningful words
                    processed.append(processed_text)
        
        except Exception as e:
            print(f"⚠️ Error processing text with spaCy: {str(e)[:50]}")
            continue
    
    return processed

def preprocess_for_sentiment_bertweet(texts: List[str]) -> List[str]:
    """
    Preprocess texts specifically for BERTweet sentiment analysis.
    
    IMPORTANT: BERTweet requires exact preprocessing:
    a. URL Normalization: Replace URLs with HTTPURL token
    b. Emoji Conversion: Convert emojis to text descriptors
    c. Whitespace Cleaning: Single spaces only
    d. PRESERVE: Capitalization, punctuation, grammar, stopwords
    
    Args:
        texts: List of raw text strings
    
    Returns:
        List of preprocessed texts ready for BERTweet
    """
    
    import re
    
    processed = []
    
    for text in texts:
        if not text or text in ["[removed]", "[deleted]"]:
            continue
        
        if len(text.strip()) < 5:
            continue
        
        # Step 1: URL Normalization - Replace URLs with HTTPURL token
        text = re.sub(r'http\S+|www\S+|https\S+', 'HTTPURL', text)
        
        # Step 2: Emoji Conversion - Convert emojis to text descriptors
        try:
            import emoji
            text = emoji.replace_emoji(text, replace=" ")
        except:
            # Fallback: just keep emojis as-is if emoji library not available
            pass
        
        # Step 3: Username/Mention Normalization - Replace @mentions with @USER
        text = re.sub(r'@\w+', '@USER', text)
        
        # Step 4: Whitespace Cleaning - Replace multiple spaces, tabs, newlines with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # PRESERVE: Do NOT lowercase, do NOT remove punctuation, do NOT remove stopwords
        
        if text:
            processed.append(text)
    
    return processed

def preprocess_for_sentiment(texts: List[str]) -> List[str]:
    """Preprocess texts specifically for sentiment analysis."""
    
    processed = []
    
    for text in texts:
        if not text or text in ["[removed]", "[deleted]"]:
            continue
        
        if len(text.strip()) < 5:
            continue
        
        text = clean_text(text)
        text = normalize_text(text)
        
        if text:
            processed.append(text)
    
    return processed


def remove_duplicates(items: List[Dict]) -> List[Dict]:
    """Remove duplicate items based on text content."""
    
    seen = set()
    unique_items = []
    
    for item in items:
        text = item.get("text", "")
        if text and text not in seen:
            seen.add(text)
            unique_items.append(item)
    
    return unique_items