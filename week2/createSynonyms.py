import fasttext
from pathlib import Path

similarity_threshold = 0.74
model_path = Path('~/workspace/corise/datasets/fasttext/title_model.bin').expanduser()
output_path = Path('~/workspace/corise/datasets/fasttext/synonyms.csv').expanduser()

model = fasttext.load_model(str(model_path))

synonyms = []
top_words_path = Path('~/workspace/corise/datasets/fasttext/top_words.txt').expanduser()
with open(top_words_path, 'r') as f:    
    for top_word in f:
        nearest_neighbours = model.get_nearest_neighbors(top_word)
        filtered_neighbours = [(word) for similarity,word in nearest_neighbours if similarity > similarity_threshold]  
        if len(filtered_neighbours) > 1:
            comma_separated = ",".join(filtered_neighbours)
            synonyms.append(comma_separated)      

with open(output_path, 'w') as f:
    for line in synonyms:
        f.write(line)
        f.write('\n')