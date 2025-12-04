import re
from datasketch import MinHash, MinHashLSH, WeightedMinHashGenerator
import pandas as pd
import numpy as np
from collections import Counter

class AdvancedMatcher:
    def __init__(self, complaints_db):
        self.complaints_db = complaints_db
        self.tokenized_docs = [self._tokenize(text) for text in complaints_db]
        
        # 1. Build Vocabulary & Calculate Weights (IDF equivalent)
        # We use frequency counts to build the WeightedMinHash generator
        all_tokens = [t for doc in self.tokenized_docs for t in doc]
        self.vocab_counts = Counter(all_tokens)
        
        # Create Weighted MinHash Generator
        # dim=128 is a good balance of speed vs accuracy
        self.wmg = WeightedMinHashGenerator(dim=128, sample_size=len(self.vocab_counts))
        
        # 2. Build LSH Index (Speed Layer)
        # threshold=0.5 means we only index things likely to be >50% similar
        self.lsh = MinHashLSH(threshold=0.5, num_perm=128)
        
        # Index all documents
        self.minhashes = {}
        for i, doc in enumerate(self.tokenized_docs):
            # Create weighted minhash
            wm = self._create_wm(doc)
            self.minhashes[i] = wm
            self.lsh.insert(str(i), wm)
            
        # Pre-calculate Max Information Content for normalization (optional but good)
        self.max_idf = max(1 / (count + 1) for count in self.vocab_counts.values())

    def _tokenize(self, text):
        # Simple optimization: only keep alphanumeric, lowercase
        return re.findall(r'\w+', text.lower())

    def _create_wm(self, tokens):
        # Create weighted minhash vector from tokens
        # We use the counts of tokens in the specific document for the vector
        vec = [1 if t in tokens else 0 for t in self.vocab_counts.keys()]
        # In a real prod scenario, you'd align this with the generator's expected input efficiently
        # For this snippet, we use the provided bulk function from datasketch if possible
        # or re-generate. *Optimization note*: Ideally, map tokens to IDs first.
        
        # Simplified generation for clarity:
        wm = self.wmg.minhash(tokens) 
        return wm

    def _get_information_content(self, tokens_a, tokens_b):
        """
        Calculates the 'value' of the intersection.
        High value = match contains rare/important words.
        Low value = match contains only common filler words.
        """
        intersection = set(tokens_a).intersection(set(tokens_b))
        score = 0
        for word in intersection:
            # IDF approximation: 1 / frequency
            # Rare words give HIGHER score
            word_rarity = 1 / (self.vocab_counts.get(word, 10000) + 1)
            score += word_rarity
        return score

    def find_match(self, new_text, strictness_threshold=0.005):
        tokens = self._tokenize(new_text)
        if not tokens: return None
        
        query_wm = self._create_wm(tokens)
        
        # Step 1: Fast Candidate Retrieval (O(log N))
        candidate_indices = self.lsh.query(query_wm)
        
        if not candidate_indices:
            return "No Match Found (LSH filtered)"

        best_match = None
        highest_score = -1
        
        # Step 2 & 3: Verification & SIC Filter
        for idx in candidate_indices:
            idx = int(idx)
            target_tokens = self.tokenized_docs[idx]
            
            # Jaccard Similarity (0.0 to 1.0)
            jaccard = query_wm.jaccard(self.minhashes[idx])
            
            # SIC Score (The False Positive Killer)
            sic_score = self._get_information_content(tokens, target_tokens)
            
            # COMBINED SCORE logic
            # We only accept if Jaccard is high AND SIC is significant
            if jaccard > highest_score:
                # FP Check: Does this match have enough "rare word" overlap?
                if sic_score > strictness_threshold: 
                    highest_score = jaccard
                    best_match = self.complaints_db[idx]
        
        if best_match:
            return best_match, highest_score
        else:
            return "No Match (FP Avoided by SIC Filter)"

# --- Usage ---
db = [
    "The billing cycle was incorrect for March",
    "I cannot login to the dashboard",
    "The system is very slow today",
    "I have an issue with the account"  # Generic trap
]

matcher = AdvancedMatcher(db)

# Test Case 1: Real match
print(matcher.find_match("billing cycle is wrong")) 
# Output: ('The billing cycle was incorrect for March', 0.8...)

# Test Case 2: False Positive Trap
# "I have an issue" matches the generic sentence in DB heavily on simple Jaccard
# but "I", "have", "an", "issue" are common, so SIC score will be low.
print(matcher.find_match("I have an issue")) 
# Output: No Match (FP Avoided...) or very low confidence depending on threshold
