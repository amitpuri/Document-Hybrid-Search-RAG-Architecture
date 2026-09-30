import arxiv
import requests
import time

start = time.time()
paper = next(arxiv.Client().results(arxiv.Search(id_list=['2609.13406'])))
response = requests.get(paper.pdf_url)
file_path = 'C:/repositories/amitpuri-repos/research-hybrid-search/corpus/2609.13406.pdf'
with open(file_path, 'wb') as f:
    f.write(response.content)
end = time.time()
print(f'Downloaded to: {file_path}')
print(f'Total time: {end - start:.2f} seconds')
print(f'Paper: {paper.title}')
