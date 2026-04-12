# evaluation/metrics.py

def avg_retrieval_score(scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)


def verdict_stats(results):
    total = len(results)

    counts = {
        "CORRECT": 0,
        "AMBIGUOUS": 0,
        "INCORRECT": 0
    }

    for r in results:
        counts[r["verdict"]] += 1

    return {k: round(v / total, 2) for k, v in counts.items()}