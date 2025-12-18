import torch
import itertools

def compatibility_score(embeddings):
    pairs = list(itertools.combinations(range(len(embeddings)), 2))

    if not pairs:
        return 50.0  # fallback if only 1 item

    scores = []
    for i, j in pairs:
        sim = torch.dot(embeddings[i], embeddings[j]).item()
        scores.append(sim)

    mean_sim = sum(scores) / len(scores)

    # Convert cosine sim (-1 → 1) → (0 → 100)
    final_score = (mean_sim + 1) / 2 * 100
    return round(final_score, 2)
