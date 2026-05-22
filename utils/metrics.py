import re

def compute_levenshtein_distance(s1, s2):
    """
    Computes Levenshtein edit distance between two strings s1 and s2.
    Implemented locally to guarantee zero dependencies.
    """
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def normalize_text(text):
    """Removes articles, punctuation, duplicate whitespaces, and lowercases text."""
    text = str(text).lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove duplicate whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_anls(predicted, ground_truths, threshold=0.5):
    """
    Calculates Average Normalized Levenshtein Similarity (ANLS) for a single sample.
    DocVQA accepts multiple valid ground-truth strings per question. We choose the best matching one.
    """
    if not ground_truths:
        return 0.0
        
    best_anls = 0.0
    pred_norm = normalize_text(predicted)
    
    for gt in ground_truths:
        gt_norm = normalize_text(gt)
        
        max_len = max(len(pred_norm), len(gt_norm))
        if max_len == 0:
            best_anls = max(best_anls, 1.0)
            continue
            
        edit_dist = compute_levenshtein_distance(pred_norm, gt_norm)
        norm_dist = edit_dist / max_len
        
        # If edit distance ratio is above threshold, similarity is set to 0
        similarity = 1.0 - norm_dist if norm_dist < threshold else 0.0
        best_anls = max(best_anls, similarity)
        
    return best_anls

def calculate_exact_match(predicted, ground_truths):
    """Calculates Exact Match (EM) metric."""
    pred_norm = normalize_text(predicted)
    for gt in ground_truths:
        if pred_norm == normalize_text(gt):
            return 1.0
    return 0.0

def calculate_f1(predicted, ground_truths):
    """Calculates token-level precision, recall, and F1 score."""
    pred_tokens = normalize_text(predicted).split()
    if not pred_tokens:
        return 0.0
        
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_text(gt).split()
        if not gt_tokens:
            continue
            
        # Count token intersections
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        
        if num_same == 0:
            continue
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
        
    return best_f1
