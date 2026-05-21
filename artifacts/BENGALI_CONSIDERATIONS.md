# Bengali Language Considerations for LLM Library

## 1. BENGALI SCRIPT FUNDAMENTALS

### 1.1 Script Characteristics
- **Script Type**: Abugida (consonant-vowel ligature system)
- **Direction**: Left-to-right
- **Base Characters**: 50 base consonants + 10 vowels
- **Complexity**: High due to conjunct consonants and diacritics

### 1.2 Character Set

#### Vowels (স্বরবর্ণ)
```
অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ
a  ā  i  ī  u  ū  ri e  ai o  au
```

#### Consonants (ব্যঞ্জনবর্ণ)
```
ক খ গ ঘ ঙ
চ ছ জ ঝ ঞ
ট ঠ ড ঢ ণ
ত থ দ ধ ন
প ফ ব ভ ম
য র ল শ ষ স হ ড় ঢ় য়
```

#### Vowel Marks (কার)
```
া (া-kar)    - /aː/
ি (ি-kar)    - /i/
ী (ী-kar)    - /iː/
ু (ু-kar)    - /u/
ূ (ূ-kar)    - /uː/
ৃ (ৃ-kar)    - /ri/
ে (ে-kar)    - /e/
ৈ (ৈ-kar)    - /ai/
ো (ো-kar)    - /o/
ৌ (ৌ-kar)    - /au/
```

---

## 2. KEY LINGUISTIC CHALLENGES

### 2.1 Conjunct Consonants (যুক্তব্যঞ্জন)

**What are they?**
Two or more consonants combine into a single ligature:

```python
# Examples:
"ক্ষ" = ক + ্ + ষ (kṣa sound)
"জ্ঞ" = জ + ্ + ঞ (gyã sound)
"ষ্ট" = ষ + ্ + ট (ṣṭa sound)
"দ্র" = দ + ্ + র (dra sound)
"ত্র" = ত + ্ + র (tra sound)
"ন্ট" = ন + ্ + ট (nṭa sound)
"ম্প" = ম + ্ + প (mpa sound)

# Common Bengali conjuncts (340+ possible combinations)
```

**Tokenization Strategy:**
```python
# Option 1: Treat as single unit (RECOMMENDED for Bengali)
tokens = ["ক্ষ", "জ্ঞ"]  # Better semantics

# Option 2: Split into components
tokens = ["ক", "্", "ষ", "জ", "্", "ঞ"]  # Loses structure

# Option 3: Unicode Normalization (NFC)
# Preserves combining marks while normalizing representation
```

### 2.2 Halant/Virama (হসন্ত)

The "্" character that marks consonant clusters:

```python
# Examples:
"ক্ত" = ক (k) + ্ (halant) + ত (ta)
"ক্ষ" = ক (k) + ্ (halant) + ষ (ṣa)

# Challenges:
# 1. Can appear with different combining marks
# 2. May be invisible in some fonts
# 3. Critical for parsing

HALANT = '\u09BC'  # Unicode: U+09BC
```

### 2.3 Script Variants

**Normalization Issues:**
```python
# Two equivalent representations:
"আমি" vs "ा॥ि"  # Different Unicode points, same meaning

# Solution: Unicode Normalization (NFC)
import unicodedata

def normalize_bengali(text):
    # NFC normalizes to composed form
    return unicodedata.normalize('NFC', text)

# Handles:
# ✓ Composed vs decomposed vowel marks
# ✓ Different ways to represent same conjunct
# ✓ Compatibility characters
```

### 2.4 Zero-Width Joiner (ZWJ) and Zero-Width Non-Joiner (ZWNJ)

```python
# ZWJ affects how conjuncts display
ZWJ = '\u200D'
ZWNJ = '\u200C'

# Example: Influencing conjunct formation
"ক" + ZWJ + "্" + ZWJ + "ষ"  # Explicit conjunct formation
"ক" + ZWNJ + "্" + ZWNJ + "ষ"  # Prevents conjunct formation
```

**Tokenizer Handling:**
```python
def handle_zero_width_chars(text):
    """Normalize zero-width characters"""
    ZWJ = '\u200D'
    ZWNJ = '\u200C'
    
    # Remove or normalize ZWJ/ZWNJ based on strategy
    # Strategy 1: Remove entirely
    text = text.replace(ZWJ, '').replace(ZWNJ, '')
    
    # Strategy 2: Mark them as special tokens
    # [for later tokenization phase]
    
    return text
```

---

## 3. BENGALI-SPECIFIC PREPROCESSING

### 3.1 Text Normalization Pipeline

```python
# src/bllmc/data/preprocessing/bengali_normalizer.py

import unicodedata
import re

class BengaliNormalizer:
    """Handle Bengali-specific text normalization"""
    
    # Unicode points for Bengali
    BENGALI_SCRIPT_START = 0x0980
    BENGALI_SCRIPT_END = 0x09FF
    
    # Special characters
    HALANT = '\u09BC'
    NUKTA = '\u09BC'
    CHANDRABINDU = '\u0981'
    ANUSVARA = '\u0982'
    VISARGA = '\u0983'
    ZWJ = '\u200D'
    ZWNJ = '\u200C'
    
    @staticmethod
    def is_bengali_char(char):
        """Check if character is Bengali"""
        code = ord(char)
        return BengaliNormalizer.BENGALI_SCRIPT_START <= code <= BengaliNormalizer.BENGALI_SCRIPT_END
    
    @staticmethod
    def unicode_normalize(text):
        """Apply Unicode NFC normalization"""
        return unicodedata.normalize('NFC', text)
    
    @staticmethod
    def remove_extra_spaces(text):
        """Remove multiple spaces, tabs, newlines"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def normalize_numerals(text):
        """Convert Arabic numerals to Bengali numerals (or vice versa)"""
        # Option 1: Normalize to Arabic (0-9)
        bengali_digits = '০১২৩৪৫৬৭৮৯'
        arabic_digits = '0123456789'
        
        trans_table = str.maketrans(bengali_digits, arabic_digits)
        return text.translate(trans_table)
    
    @staticmethod
    def normalize_quotes(text):
        """Normalize different quote types to standard ASCII quotes"""
        replacements = {
            '"': '"',   # Right double quotation mark
            '"': '"',   # Left double quotation mark
            "'": "'",   # Right single quotation mark
            "'": "'",   # Left single quotation mark
            '`': "'",   # Backtick
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    @staticmethod
    def remove_diacritics(text, keep_nukta=True):
        """Remove diacritical marks (optional)"""
        # Vowel marks
        vowel_marks = '\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C7\u09C8\u09CB\u09CC'
        
        for mark in vowel_marks:
            text = text.replace(mark, '')
        
        # Keep Nukta (diacritic dot) for pronunciation distinction
        if not keep_nukta:
            text = text.replace('\u09BC', '')
        
        return text
    
    @staticmethod
    def remove_chandrabindu(text):
        """Remove Chandrabindu (ঁ)"""
        return text.replace('\u0981', '')
    
    @staticmethod
    def normalize_all(text):
        """Apply all normalization steps"""
        text = BengaliNormalizer.unicode_normalize(text)
        text = BengaliNormalizer.remove_extra_spaces(text)
        text = BengaliNormalizer.normalize_quotes(text)
        text = BengaliNormalizer.remove_diacritics(text, keep_nukta=True)
        return text


# Example usage:
bengali_text = "আমি  একটি   বাক্য লিখছি।"
normalized = BengaliNormalizer.normalize_all(bengali_text)
# Result: "আমি একটি বাক্য লিখছি।"
```

### 3.2 Sentence Segmentation

```python
# src/bllmc/data/preprocessing/bengali_segmenter.py

class BengaliSentenceSegmenter:
    """Segment Bengali text into sentences"""
    
    # Bengali sentence terminators
    TERMINAL_MARKS = {
        '।': 'devanagari_danda',      # Single danda
        '॥': 'devanagari_double_danda', # Double danda
        '।': 'bengali_danda',
        '!': 'exclamation',
        '?': 'question',
        '।।': 'double_danda',
    }
    
    @staticmethod
    def segment(text):
        """Split text into sentences"""
        # Use sentence terminators
        sentences = re.split(r'[।\!\?]+', text)
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


# Example:
text = "আমি স্কুলে যাই। আমার বন্ধুরাও যায়।"
sentences = BengaliSentenceSegmenter.segment(text)
# Result: ["আমি স্কুলে যাই", "আমার বন্ধুরাও যায়"]
```

---

## 4. BENGALI-SPECIFIC TOKENIZATION

### 4.1 Morphological Tokenization

```python
# src/bllmc/tokenizers/bengali_tokenizer.py

import re
from typing import List

class BengaliTokenizer:
    """
    Bengali-aware tokenizer that respects morphological structure
    """
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.build_vocab()
    
    def build_vocab(self):
        """Build vocabulary including conjunct consonants"""
        # Step 1: Add base characters
        base_consonants = 'ককখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়'
        vowels = 'অআইঈউঊঋএঐওঔ'
        vowel_marks = 'ািীুূৃেৈোৌ'
        
        # Step 2: Add special tokens
        special_tokens = [
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', 
            '[MASK]', '[BOS]', '[EOS]'
        ]
        
        idx = 0
        for token in special_tokens:
            self.vocab[token] = idx
            idx += 1
        
        # Step 3: Add individual characters
        for char in (base_consonants + vowels + vowel_marks):
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1
        
        # Step 4: Add common conjuncts
        common_conjuncts = [
            'ক্ষ', 'জ্ঞ', 'ষ্ট', 'দ্র', 'ত্র', 'ন্ট', 'ম্প',
            'ক্ত', 'ক্ল', 'গ্ল', 'প্ত', 'প্ল', 'প্র',
            'ব্র', 'য়্ম', 'ল্ড', 'ল্ত', 'ল্প', 'ল্ব',
            'শ্ছ', 'শ্চ', 'শ্ট', 'শ্ন', 'ষ্ণ', 'স্ত',
            'ন্ধ', 'ন্ত', 'ঞ্জ', 'ঙ্গ', 'ঙ্ক', 'ড়্ঢ়', 'হ্ম'
        ]
        
        for conjunct in common_conjuncts:
            if conjunct not in self.vocab:
                self.vocab[conjunct] = idx
                idx += 1
    
    def is_conjunct(self, char_seq):
        """Check if sequence is a conjunct"""
        return char_seq in self.vocab
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Bengali text respecting conjuncts
        
        Strategy:
        1. Try to match longest conjunct
        2. Fall back to individual characters
        3. Use vocabulary for OOV handling
        """
        tokens = []
        i = 0
        
        while i < len(text):
            # Try 3-character conjunct
            if i + 3 <= len(text):
                seq3 = text[i:i+3]
                if self.is_conjunct(seq3):
                    tokens.append(seq3)
                    i += 3
                    continue
            
            # Try 2-character conjunct
            if i + 2 <= len(text):
                seq2 = text[i:i+2]
                if self.is_conjunct(seq2):
                    tokens.append(seq2)
                    i += 2
                    continue
            
            # Single character
            char = text[i]
            if char not in self.vocab:
                tokens.append('[UNK]')
            else:
                tokens.append(char)
            i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token[idx] for idx in token_ids]
        return ''.join(tokens)


# Example usage:
tokenizer = BengaliTokenizer()
text = "আমি ক্ষেত্রে যাই"
tokens = tokenizer.tokenize(text)
# Result: ['আ', 'ম', 'ি', ' ', 'ক্ষ', 'ে', 'ত্র', 'ে', ' ', 'য', 'া', 'ই']

token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)
```

### 4.2 WordPiece for Bengali

```python
# src/bllmc/tokenizers/bengali_wordpiece.py

class BengaliWordPiece:
    """WordPiece tokenization adapted for Bengali"""
    
    def __init__(self, vocab, lower_case=True):
        self.vocab = vocab
        self.lower_case = lower_case
    
    def tokenize(self, word):
        """
        Break word into subword units
        
        Strategy:
        1. Check word in vocab
        2. Split from start using longest matching prefix
        3. Continue with "##" prefix for subwords
        """
        if word in self.vocab:
            return [word]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = None
            
            # Find longest matching subword
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                
                if substr in self.vocab:
                    found = substr
                    break
                end -= 1
            
            if found is None:
                # Character not in vocab
                tokens.append('[UNK]')
                start += 1
            else:
                tokens.append(found)
                start = end
        
        return tokens
```

---

## 5. SPECIAL TOKEN HANDLING FOR BENGALI

```python
# src/bllmc/tokenizers/special_tokens.py

class BengaliSpecialTokens:
    """Special tokens for Bengali models"""
    
    # Core special tokens
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    
    # For generation
    BOS_TOKEN = '[BOS]'  # Beginning of sequence
    EOS_TOKEN = '[EOS]'  # End of sequence
    
    # Bengali-specific
    NUMERAL_MASK = '[NUM]'  # Mask for numbers
    PROPER_MASK = '[PROPER]'  # Mask for proper nouns
    VERB_TOKEN = '[VERB]'  # Mark for verbs
    NOUN_TOKEN = '[NOUN]'  # Mark for nouns
    
    # Script variants
    MIXED_SCRIPT = '[MIXED]'  # Mixed script text
    
    SPECIAL_TOKENS_LIST = [
        PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, 
        MASK_TOKEN, BOS_TOKEN, EOS_TOKEN,
        NUMERAL_MASK, PROPER_MASK, VERB_TOKEN, NOUN_TOKEN,
        MIXED_SCRIPT
    ]
    
    @classmethod
    def get_special_tokens_dict(cls):
        """Get dictionary of special tokens"""
        return {
            'pad_token': cls.PAD_TOKEN,
            'unk_token': cls.UNK_TOKEN,
            'cls_token': cls.CLS_TOKEN,
            'sep_token': cls.SEP_TOKEN,
            'mask_token': cls.MASK_TOKEN,
            'bos_token': cls.BOS_TOKEN,
            'eos_token': cls.EOS_TOKEN,
        }
```

---

## 6. BENGALI NUMERAL HANDLING

```python
# src/bllmc/adapters/bengali_numeral.py

class BengaliNumeralAdapter:
    """Handle Bengali numerals (০-৯) and mixed numerals"""
    
    BENGALI_DIGITS = '০১२৩৪৫৬৭৮৯'
    ARABIC_DIGITS = '0123456789'
    
    @staticmethod
    def to_arabic(text):
        """Convert Bengali numerals to Arabic"""
        trans = str.maketrans(BengaliNumeralAdapter.BENGALI_DIGITS, 
                             BengaliNumeralAdapter.ARABIC_DIGITS)
        return text.translate(trans)
    
    @staticmethod
    def to_bengali(text):
        """Convert Arabic numerals to Bengali"""
        trans = str.maketrans(BengaliNumeralAdapter.ARABIC_DIGITS, 
                             BengaliNumeralAdapter.BENGALI_DIGITS)
        return text.translate(trans)
    
    @staticmethod
    def is_mixed_numeral(text):
        """Check if text contains both Arabic and Bengali numerals"""
        has_arabic = any(c in BengaliNumeralAdapter.ARABIC_DIGITS for c in text)
        has_bengali = any(c in BengaliNumeralAdapter.BENGALI_DIGITS for c in text)
        return has_arabic and has_bengali
    
    @staticmethod
    def normalize_numerals(text, target='arabic'):
        """Normalize to single numeral type"""
        if target == 'arabic':
            return BengaliNumeralAdapter.to_arabic(text)
        else:
            return BengaliNumeralAdapter.to_bengali(text)
```

---

## 7. DATASET PREPARATION FOR BENGALI

### 7.1 Data Collection Strategies

```
1. **Corpus Sources**:
   - News articles (প্রথম আলো, ঢাকা ট্রিবিউন)
   - Wikipedia (Bengali)
   - Common Crawl (Bengali subset)
   - Literature and books
   - Web scraped content

2. **Data Cleaning Priorities**:
   - Remove non-Bengali text
   - Handle mixed scripts (Bengali + English)
   - Normalize formatting
   - Remove HTML/XML
   - Check encoding consistency

3. **Quality Metrics**:
   - Language identification
   - Script validation
   - Alphabet coverage
   - Vocabulary diversity
```

### 7.2 Preprocessing Pipeline

```python
# src/bllmc/data/preprocessing/pipeline.py

class BengaliDataPreprocessor:
    """End-to-end Bengali data preprocessing"""
    
    def __init__(self):
        self.normalizer = BengaliNormalizer()
        self.tokenizer = BengaliTokenizer()
    
    def process(self, text):
        """Apply full preprocessing pipeline"""
        # Step 1: Normalize
        text = self.normalizer.normalize_all(text)
        
        # Step 2: Check if Bengali
        bengali_chars = sum(1 for c in text if self.normalizer.is_bengali_char(c))
        if bengali_chars / len(text) < 0.8:  # <80% Bengali
            return None
        
        # Step 3: Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Step 4: Filter if too short or too long
        if len(tokens) < 5 or len(tokens) > 512:
            return None
        
        return tokens
```

---

## 8. EVALUATION METRICS FOR BENGALI

```python
# src/bllmc/metrics/bengali_metrics.py

class BengaliMetrics:
    """Bengali-specific evaluation metrics"""
    
    @staticmethod
    def character_error_rate(predicted, reference):
        """CER for Bengali text"""
        # Character-level Levenshtein distance
        pass
    
    @staticmethod
    def word_error_rate(predicted, reference):
        """WER for Bengali text"""
        pass
    
    @staticmethod
    def bleu_score(predicted, reference):
        """BLEU score adapted for Bengali"""
        pass
    
    @staticmethod
    def perplexity(model, dataset):
        """Measure model perplexity on Bengali text"""
        pass
```

---

## 9. TESTING BENGALI FUNCTIONALITY

```python
# tests/test_bengali_components.py

import unittest
from bllmc.tokenizers import BengaliTokenizer
from bllmc.data.preprocessing import BengaliNormalizer

class TestBengaliComponentsTest(unittest.TestCase):
    
    def test_conjunct_tokenization(self):
        tokenizer = BengaliTokenizer()
        text = "ক্ষ"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["ক্ষ"])
    
    def test_numeral_normalization(self):
        text = "২০২৪ সালে ৫টি বই পড়েছি"
        normalizer = BengaliNormalizer()
        normalized = normalizer.normalize_numerals(text)
        self.assertNotIn('২', normalized)
    
    def test_unicode_normalization(self):
        """Test Unicode NFC normalization"""
        text1 = "ো" # Precomposed
        text2 = "ে" + "ৗ"  # Decomposed
        
        normalizer = BengaliNormalizer()
        norm1 = normalizer.unicode_normalize(text1)
        norm2 = normalizer.unicode_normalize(text2)
        
        self.assertEqual(norm1, norm2)
```

---

## 10. RECOMMENDATIONS FOR BENGALI LLM

### 10.1 Tokenizer Choice
**Recommendation**: Custom Bengali WordPiece
- Respects conjunct consonants
- Handles morphological structure
- Better than generic SentencePiece for Bengali

### 10.2 Vocabulary Size
- **Base**: 30,000 tokens
- **Standard**: 50,000 tokens (recommended)
- **Large**: 100,000 tokens

### 10.3 Special Handling
1. **Always normalize Unicode** (NFC)
2. **Preserve morphological structure** (conjuncts)
3. **Handle numerals** (normalize to Arabic)
4. **Script validation** (ensure Bengali text)

### 10.4 Pre-training Tasks
1. **Masked Language Modeling** (MLM) - Primary
2. **Next Sentence Prediction** (NSP) - Secondary
3. **Morphological awareness** - Task-specific

---

## RESOURCES

- [Unicode Bengali Block](https://en.wikipedia.org/wiki/Bengali_(Unicode_block))
- [ISCII Standard](https://en.wikipedia.org/wiki/ISCII)
- [Bengali Morphology](https://en.wiktionary.org/wiki/Category:Bengali_language)

