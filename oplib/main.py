#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
from oplib.oplib2 import *
from oplib.preprocessOplib import PreprocessLibrary
from requests.exceptions import ConnectionError
from tenacity import retry, stop_after_attempt, wait_fixed
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

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def fetch_content(url, session, **search_options):
    response = session.post(url, data=search_options)
    response.raise_for_status()
    return response.text

def run_oplib_main():
    oplib = OpenLibrary()
    preprocess = PreprocessLibrary()

    current_date = datetime.date.today()
    current_date = current_date - datetime.timedelta(days=8)
    one_month_ago = current_date - datetime.timedelta(days=1)

    start_day, start_month, start_year = one_month_ago.day, one_month_ago.month, one_month_ago.year
    end_day, end_month, end_year = current_date.day, current_date.month, current_date.year

    print(end_day, end_month, end_year)
    print(start_day, start_month, start_year)

    publication_types = [
        AdvancedSearchType.SKRIPSI,
        AdvancedSearchType.TA,
        AdvancedSearchType.THESIS
    ]

    combined_df = pd.DataFrame({
        'title': [], 'classification': [], 'type_publication': [], 'subject': [], 'abstract': [],
        'keywords': [], 'author': [], 'lecturer': [], 'publisher': [], 'publish_year': []
    })

    for publication_type in publication_types:
        search_options = {
            'search[type]': publication_type,
            'search[number]': '',
            'search[title]': '',
            'search[author]': '',
            'search[publisher]': '',
            'search[editor]': '',
            'search[subject]': '',
            'search[classification]': '',
            'search[location]': '',
            'search[entrance][from][day]': start_day,
            'search[entrance][from][month]': start_month,
            'search[entrance][from][year]': start_year,
            'search[entrance][to][day]': end_day,
            'search[entrance][to][month]': end_month,
            'search[entrance][to][year]': end_year,
        }

        file_type = {AdvancedSearchType.SKRIPSI: 'SKRIPSI', AdvancedSearchType.TA: 'TA', AdvancedSearchType.THESIS: 'THESIS'}
        file = file_type.get(publication_type, 'Unknown')

        print(f'Scraping {file} from {start_day} {start_month} {start_year} to {end_day} {end_month} {end_year} ......')

        try:
            content = fetch_content(f'{oplib.base_url}/home/catalog.html', oplib.session, **search_options)
            results = oplib.parse_results(content)

            for index, totals, data in results:
                if data:
                    combined_df = pd.concat([combined_df, pd.DataFrame([data])], ignore_index=True)
                print(f"[{index}/{totals}]: {data['title'] if data else 'Error parsing data'}")

        except ConnectionError as e:
            print(f"Connection error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Scraping done, performing preprocessing....")
    try:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        combined_df = combined_df.rename(columns={
            'title': 'Judul', 'author': 'Penulis1', 'lecturer': 'Penulis2',
            'publish_year': 'Tahun', 'abstract': 'Abstrak'
        })

        combined_df = combined_df[["Judul", "Penulis1", "Penulis2", "Tahun", "Abstrak"]]
        combined_df = combined_df.dropna()

        # combined_df.to_json("tahap1.json", orient='records')

        combined_df['Abstrak'] = combined_df['Abstrak'].apply(preprocess.cleaningAbstrak)
        combined_df['Judul'] = combined_df['Judul'].apply(preprocess.cleaningJudul)
        combined_df["Penulis1"] = combined_df["Penulis1"].apply(preprocess.cleaningPenulis)
        combined_df["Penulis"] = combined_df["Penulis1"] + ", " + combined_df["Penulis2"]
        combined_df = combined_df.drop(["Penulis1", "Penulis2"], axis=1)
        combined_df = combined_df[["Judul", "Tahun", "Abstrak", "Penulis"]]
        combined_df["Tahun"] = combined_df["Tahun"].astype(int)
        # combined_df.to_json("tahap2.json", orient='records')

        combined_df['Abstrak'] = combined_df['Abstrak'].replace('', np.nan)
        combined_df['Judul'] = combined_df['Judul'].replace('', np.nan)
        # combined_df.to_json("tahap3.json", orient='records')
        combined_df = combined_df.dropna()
        combined_df.to_json("testing.json")

        # Membaca data dosen dari CSV
        df1 = pd.read_csv('./dataDosen.csv', sep=';')
        df1['NAMA LENGKAP'] = df1['NAMA LENGKAP'].str.title()
        dosen_list = df1['NAMA LENGKAP'].tolist()

       # Menerapkan fungsi untuk setiap baris di kolom 'Penulis' dan membuat DataFrame baru
        df2 = combined_df[combined_df['Penulis'].apply(lambda penulis: preprocess.penulisGaAda(penulis, dosen_list))]
        print(df2.info())

        # Menyimpan hasil ke file JSON
        df2.to_json("Penulisgaada.json", orient='records', lines=True)

        benarkan_nama = {}
        with open('./benarkanNama.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    salah, benar = parts
                    benarkan_nama[salah.strip()] = benar.strip()

       # Menerapkan fungsi ganti_nama ke df2["Penulis"]
        df2["Penulis"] = df2["Penulis"].apply(lambda penulis: preprocess.gantiNama(penulis, benarkan_nama))
        # df2.to_json("1.json", orient='records', lines=True)
        df2['Penulis'] = df2['Penulis'].astype(str).apply(lambda x: ', '.join([item.strip() for item in x.split(',') if item.strip() != '0']))
        # df2.to_json("2.json", orient='records', lines=True)
        # Mengubah string kosong dan "nan" menjadi NaN
        df2['Penulis'].replace(['', 'nan'], np.nan, inplace=True)
        # df2.to_json("3.json", orient='records', lines=True)
        # Menghapus baris yang mengandung NaN
        df2.dropna(subset=['Penulis'], inplace=True)
        # df2.to_json("Ending.json", orient='records', lines=True)

        df_final = combined_df[combined_df['Penulis'].apply(lambda penulis: preprocess.penulisAda(penulis, dosen_list))]
        df_final = pd.concat([df_final,df2])
        # df_final.to_json("Done.json", orient='records', lines=True)

        preprocessed_file_name = f'preprocessedOplib_{start_day}-{start_month}-{start_year}_{end_day}-{end_month}-{end_year}.json'
        print("Preprocessing done!")
        print(f'Finished preprocessing all data!\nFile name: {preprocessed_file_name}\n\n')
        df_final["Source"] = "Oplib"
        df_final.to_json(preprocessed_file_name, orient='records')
        
        df = pd.read_json("./hasil_akhir.json")
        df_final = pd.read_json(preprocessed_file_name)
        
        # Menerapkan fungsi klasifikasi ke setiap baris di kolom Abstrak
        # Initialize the pipeline
        classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)

        tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")
        df_final['Sdgs'] = df_final['Abstrak'].apply(classify_sdgs)
        df_final["Source"] = "Oplib"
        print(df_final.info())
        print(df.info())
        df = pd.concat([df, df_final])
        df = df.drop_duplicates(subset=['Judul'])
        df.to_json("./hasil_akhir.json",orient='records')
        print(df.info())

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
