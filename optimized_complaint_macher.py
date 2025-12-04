import re
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer

class OptimizedComplaintMatcher:
    def __init__(self, data, threshold=0.5, num_perm=128):
        self.data = data
        self.threshold = threshold
        self.num_perm = num_perm
        
        # 1. Calculate Word Importance (IDF) once globally
        # This helps us know that "billing" > "is"
        print("Computing global word weights...")
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        self.vectorizer.fit(data)
        # Create a fast lookup for word weights
        self.word_weights = dict(zip(
            self.vectorizer.get_feature_names_out(), 
            self.vectorizer.idf_
        ))
        
        # 2. Build the LSH Index (The Speed Layer)
        print("Building LSH Index...")
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}
        
        for i, text in enumerate(data):
            mh = self._text_to_minhash(text)
            self.minhashes[i] = mh
            self.lsh.insert(str(i), mh)
            
    def _tokenize(self, text):
        # Simple tokenizer: lowercase and keep only words
        return set(re.findall(r'\w+', text.lower()))

    def _text_to_minhash(self, text):
        # Create a MinHash signature for the text
        tokens = self._tokenize(text)
        m = MinHash(num_perm=self.num_perm)
        for t in tokens:
            m.update(t.encode('utf8'))
        return m

    def _weighted_score(self, tokens_a, tokens_b):
        """
        The False Positive Killer.
        Calculates similarity based on the *value* of shared words, not just count.
        """
        intersection = tokens_a.intersection(tokens_b)
        
        if not intersection:
            return 0.0
        
        # Sum of IDF weights for shared words
        shared_weight = sum(self.word_weights.get(w, 0) for w in intersection)
        
        # Normalize by the total weight of the query to get a 0-1 score
        # (How much of the IMPORTANT meaning in Query A is found in Target B?)
        query_weight = sum(self.word_weights.get(w, 0) for w in tokens_a)
        
        if query_weight == 0: return 0.0
        
        return shared_weight / query_weight

    def find_similar(self, new_complaint, min_weighted_score=0.3):
        query_mh = self._text_to_minhash(new_complaint)
        query_tokens = self._tokenize(new_complaint)
        
        # Step 1: LSH Retrieval (Fast reduction from 1M -> 10 candidates)
        candidate_ids = self.lsh.query(query_mh)
        
        if not candidate_ids:
            return None
        
        best_match = None
        best_score = -1
        
        # Step 2: Precise Verification
        for i in candidate_ids:
            idx = int(i)
            target_text = self.data[idx]
            target_tokens = self._tokenize(target_text)
            
            # Calculate our custom Weighted Score
            score = self._weighted_score(query_tokens, target_tokens)
            
            # Logic: Keep the best score
            if score > best_score:
                best_score = score
                best_match = target_text
        
        # Step 3: Final Threshold Check
        # If the weighted score is too low, it's a False Positive
        if best_score >= min_weighted_score:
            return {"match": best_match, "score": round(best_score, 4)}
        else:
            return None

# --- TEST ---
dataset = [
    "The billing cycle for March was calculated incorrectly.",
    "I cannot login to my account dashboard.",
    "My internet connection is dropping frequently.",
    "I have an issue."  # The trap
]

matcher = OptimizedComplaintMatcher(dataset)

# Case 1: Real match (Should match "billing cycle...")
# "billing" and "incorrectly" are rare/heavy words, so score will be high.
print(f"Query 1: {matcher.find_similar('billing is wrong')}")

# Case 2: False Positive Trap (Should return None)
# "I have an issue" shares words with the trap sentence, 
# but "I", "have", "issue" are common/low-weight.
# LSH might propose it, but _weighted_score will kill it.
print(f"Query 2: {matcher.find_similar('I have an issue')}")
