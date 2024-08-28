import json
import pandas as pd

from transformers import pipeline, AutoTokenizer

# Fungsi untuk memotong teks menggunakan tokenizer
def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

def classify_sdgs(text):
    truncated_text = truncate_text(text, tokenizer)
    results = classifier(truncated_text)
    labels = [result['label'] for result in results[0] if result['score'] > 0.5]
    return labels if labels else None

# Initialize the pipeline
classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)

tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")

# Path to your JSON file
sinta = f'contoh_sinta.json'
oplib = f"preprocessedOplib_4-8-2024_5-8-2024.json"

df1 = pd.read_json(sinta)
df2 = pd.read_json(oplib)

# df2 = df2[["Judul","Penulis","Tahun","Abstrak"]]

df = pd.concat([df1, df2])

df['Sdgs'] = df['Abstrak'].apply(classify_sdgs)

df.to_json('hasil akhir.json', orient='records')