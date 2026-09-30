from src.ingestion.pipeline import IngestionPipeline
from src.common.console import ensure_utf8_streams

ensure_utf8_streams()

ingestion = IngestionPipeline(corpus_dir='corpus', storage_backend='parquet')
chunk_store, _, pdf_paths = ingestion.run(force_rebuild=False)
texts = chunk_store.get_texts()

print('Finding correct chunk indices for all docs in EVAL_DATASET')
docs_to_find = [
    '2605.23950v1.pdf',
    '2604.00073v3.pdf',
    '2605.10223v1.pdf',
    '2509.19590v2.pdf',
    '2509.01063v1.pdf',
    '1604.08127v1.pdf',
    '2604.13107v1.pdf',
    '2605.18747v1.pdf',
    '2609.11801v1.pdf',
    '2609.11310v1.pdf',
    '2609.09153v1.pdf',
    '2604.22750v2.pdf',
    '2608.18613v1.pdf',
    '2608.07796v1.pdf',
    '2609.10630v1.pdf',
    '2609.14858v1.pdf'
]

for doc in docs_to_find:
    for i, text in enumerate(texts):
        if doc.lower() in text.lower():
            print(f'{doc}: chunk {i}')
            break
