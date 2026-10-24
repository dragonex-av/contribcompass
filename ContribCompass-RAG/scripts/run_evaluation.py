"""Reproduce the retrieval/gating benchmark and experiment matrix without API calls."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag_engine import HybridRAG
from src.evaluation import evaluate


def save(name, rows):
    destination = ROOT / 'reports' / name
    destination.parent.mkdir(exist_ok=True)
    with destination.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    engine = HybridRAG.from_folder(ROOT / 'documents', use_embeddings='--keyword-only' not in sys.argv)
    rows = evaluate(engine, 5, .24)
    label = 'hybrid' if engine.embedder is not None else 'keyword'
    save(f'evaluation-{label}.csv', rows)
    matrix = []
    for top_k in (3, 5, 7):
        for threshold in (.15, .24, .30):
            results = evaluate(engine, top_k, threshold)
            matrix.append({'retriever': label, 'top_k': top_k, 'threshold': threshold,
                           'passed': sum(r['Outcome'] == 'Pass' for r in results), 'total': len(results)})
    save(f'experiments-{label}.csv', matrix)
    print(engine.embedding_status)
    print(f"Retrieval/gate rubric: {sum(r['Outcome']=='Pass' for r in rows)}/{len(rows)}. Not generated-answer accuracy.")
    for row in rows:
        if row['Outcome'] != 'Pass':
            print('Review:', row['Question'], '-', row['Reason'])


if __name__ == '__main__':
    main()
