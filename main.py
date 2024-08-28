from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
import numpy as np
import json
import fitz
import PyPDF2
import os
import datetime
from sinta.preprocessSinta import SintaPreprocessor
from transformers import pipeline, AutoTokenizer

from oplib.main import run_oplib_main
from oplib.oplib2 import OpenLibrary
from oplib.preprocessOplib import PreprocessLibrary

app = Flask(__name__)
CORS(app)

def clean_phone_number(phone):
    if pd.isna(phone):
        return ''
    cleaned_phone = re.sub(r'\D', '', phone)
    return cleaned_phone[:10] if len(cleaned_phone) > 10 else cleaned_phone

def select_single_email(email):
    if pd.isna(email):
        return None
    return re.split(r'[;,#]', email)[0].strip()

def read_csv_to_json():
    try:
        df = pd.read_csv('DataDosen.csv', delimiter=';', on_bad_lines='skip')
        df.columns = ['no', 'nip', 'fronttitle', 'nama_lengkap', 'backtitle', 'jenis_kelamin', 'kode_dosen',
                      'nidn', 'jafung', 'lokasi_kerja', 'jabatan_struktural', 'status_pegawai',
                      'email', 'no_hp', 'lokasi_kerja_sotk']
        df['no_hp'] = df['no_hp'].apply(clean_phone_number)
        df['email'] = df['email'].apply(select_single_email)

        # Gabungkan 'fronttitle' dan 'backtitle' ke 'nama_lengkap'
        df['nama_lengkap'] = df['fronttitle'] + ' ' + df['nama_lengkap'] + ' ' + df['backtitle']

        # Hapus kolom 'fronttitle' dan 'backtitle'
        df = df.drop(columns=['fronttitle', 'backtitle'])

        # Mengganti 'N/A' di kolom 'jabatan_struktural' dengan string kosong
        df['jabatan_struktural'] = df['jabatan_struktural'].replace('N/A', '')

        # Mengganti NaN dengan string kosong di seluruh DataFrame
        df = df.fillna('')

        # Membaca data publikasi dari hasil_akhir.json
        with open('hasil_akhir.json', 'r') as f:
            publications = json.load(f)

        # Menambahkan publikasi ke setiap dosen
        def get_publications_for_dosen(nama_lengkap):
            dosen_publications = [
                pub for pub in publications if nama_lengkap.lower() in pub['Penulis'].lower()
            ]
            return dosen_publications

        # Menambahkan kolom 'publications' dan 'jumlah_publikasi' ke setiap dosen
        df['publications'] = df['nama_lengkap'].apply(get_publications_for_dosen)
        df['jumlah_publikasi'] = df['publications'].apply(len)

        json_data = df.to_dict(orient='records')
        return json_data
    except pd.errors.ParserError as e:
        return {'error': 'Error reading CSV file', 'details': str(e)}
    except KeyError as e:
        return {'error': 'Column not found', 'details': str(e)}
    except FileNotFoundError:
        return {'error': 'hasil_akhir.json file not found'}

# Endpoint untuk mendapatkan data dosen beserta publikasi mereka
@app.route('/data_dosen', methods=['GET'])
def get_data():
    try:
        # Membaca data dari file CSV
        data = read_csv_to_json()

        # Periksa jika data berisi error
        if isinstance(data, dict) and 'error' in data:
            return jsonify(data), 400

        # Response JSON tanpa pagination
        response = {
            'total': len(data),
            'data_dosen': data
        }

        return jsonify(response)
    except ValueError as e:
        return jsonify({'error': 'Invalid parameters', 'details': str(e)}), 400

@app.route('/data_dosen/<int:no>', methods=['GET'])
def get_data_by_no(no):
    try:
        data = read_csv_to_json()

        # Periksa jika data berisi error
        if isinstance(data, dict) and 'error' in data:
            return jsonify(data), 400

        # Cari dosen berdasarkan 'no'
        dosen = next((d for d in data if d['no'] == no), None)

        if dosen:
            return jsonify(dosen)
        else:
            return jsonify({'error': 'Dosen not found'}), 404

    except ValueError as e:
        return jsonify({'error': 'Invalid no parameter', 'details': str(e)}), 400

@app.route('/get-data-oplib', methods=['GET'])
def get_data_oplib():
    # Create an instance of the OpenLibrary class
    ol = OpenLibrary()

    # Define the search options based on the request or default values
    search_options = {
        'type': request.args.get('type', '4'),  # Default to SKRIPSI if not provided
        'start_date': request.args.get('start_date', '2022-01-01'),
        'end_date': request.args.get('end_date', '2022-12-31'),
        # Add other search options as needed
    }

    # Get the data from the Open Library
    content = ol.get_all_data_from_range_date(**search_options)

    # Parse the results
    parsed_results = list(ol.parse_results(content))

    # Return the results as a JSON response
    return jsonify(parsed_results)

@app.route('/post-data-sinta', methods = ['post'])
def upload_file_sinta():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Simpan file dan lakukan ekstraksi
    current_date = datetime.date.today()
    end_day, end_month, end_year = current_date.day, current_date.month, current_date.year
    file_path = f"./sinta/storage/result/scrappingSinta/crawleddSinta{end_day}-{end_month}-{end_year}.csv"
    file.save(file_path)
    preprocessor = SintaPreprocessor(file_path)
    processed_df = preprocessor.preprocess()
    file_result = f'preProcessSinta{end_day}-{end_month}-{end_year}'
    preprocessor.save_result_main(file_result)
        
     ## Klasifikasi
    def truncate_text(text, tokenizer, max_length=512):
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
        return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

    def classify_sdgs(text):
        truncated_text = truncate_text(text, tokenizer)
        results = classifier(truncated_text)
        labels = [result['label'] for result in results[0] if result['score'] > 0.5]
        return labels if labels else None
    
    df_final = pd.read_json(f'./sinta/storage/result/preprocessSinta/{file_result}.json')
    print(df_final.info())
    df = pd.read_json("./hasil_akhir.json")
    classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)

    tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")
    df_final['Sdgs'] = df_final['Abstrak'].apply(classify_sdgs)
    df_final["Source"] = "Sinta"
    print(df_final.info())
    print(df.info())
    df = pd.concat([df, df_final])
    df = df.drop_duplicates(subset=['Judul'])
    df.to_json("./hasil_akhir.json",orient='records')
    print(df.info())
    return "Data Sinta added"

@app.route('/post-data-oplib', methods=['POST'])
def post_data_oplib():
    try:
        run_oplib_main()
        return jsonify({'message': 'Oplib processing completed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Simpan file dan lakukan ekstraksi
    file_path = f"./upload/{file.filename}"
    file.save(file_path)
    extracted_data = extract_pdf_data_pymupdf(file_path)
    df = pd.DataFrame([extracted_data])
    file_result = file_path[:-4] +".csv"
    file_end = file.filename[:-4]
    df.to_csv(file_result)
    preprocessor = SintaPreprocessor(file_result)
    processed_df = preprocessor.preprocess()
    preprocessor.save_result2(file_end)

    def truncate_text(text, tokenizer, max_length=512):
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
        return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

    def classify_sdgs(text):
        truncated_text = truncate_text(text, tokenizer)
        results = classifier(truncated_text)
        labels = [result['label'] for result in results[0] if result['score'] > 0.5]
        return labels if labels else None
    
    #klasifikasi
    df_final = pd.read_json(f'./upload/{file_end}.json')
    print(df_final.info())
    df = pd.read_json("./hasil_akhir.json")
    classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)

    tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")
    df_final['Sdgs'] = df_final['Abstrak'].apply(classify_sdgs)
    df_final["Source"] = "Upload"
    print(df_final.info())
    print(df.info())
    df = pd.concat([df, df_final])
    df = df.drop_duplicates(subset=['Judul'])
    df.to_json("./hasil_akhir.json",orient='records')
    print(df.info())
    
    return jsonify(extracted_data)


@app.route('/get-oplib', methods=['GET'])
def get_oplib():
    try:
        # Load dataset JSON
        combined_df = pd.read_json("testingcrawler.json")

        # Preprocess the data
        preprocess = PreprocessLibrary()

        combined_df = combined_df.rename(columns={
            'title': 'Judul', 'author': 'Penulis1', 'lecturer': 'Penulis2',
            'publish_year': 'Tahun', 'abstract': 'Abstrak'
        })

        combined_df = combined_df[["Judul", "Penulis1", "Penulis2", "Tahun", "Abstrak"]]
        combined_df = combined_df.dropna()

        combined_df['Abstrak'] = combined_df['Abstrak'].apply(preprocess.cleaningAbstrak)
        combined_df['Judul'] = combined_df['Judul'].apply(preprocess.cleaningJudul)
        combined_df["Penulis1"] = combined_df["Penulis1"].apply(preprocess.cleaningPenulis)
        combined_df["Penulis"] = combined_df["Penulis1"] + ", " + combined_df["Penulis2"]
        combined_df = combined_df.drop(["Penulis1", "Penulis2"], axis=1)
        combined_df = combined_df[["Judul", "Tahun", "Abstrak", "Penulis"]]
        combined_df["Tahun"] = combined_df["Tahun"].astype(int)

        combined_df['Abstrak'] = combined_df['Abstrak'].replace('', np.nan)
        combined_df['Judul'] = combined_df['Judul'].replace('', np.nan)
        combined_df = combined_df.dropna()

        # Load dosen list
        df1 = pd.read_csv('DataDosen.csv', sep=';')
        df1['NAMA LENGKAP'] = df1['NAMA LENGKAP'].str.title()
        dosen_list = df1['NAMA LENGKAP'].tolist()

        # Filter data without authors in dosen_list
        df_without_authors = combined_df[combined_df['Penulis'].apply(lambda penulis: preprocess.penulisGaAda(penulis, dosen_list))]

        # Correct names in df_without_authors
        benarkan_nama = {}
        with open('benarkanNama.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    salah, benar = parts
                    benarkan_nama[salah.strip()] = benar.strip()

        df_without_authors["Penulis"] = df_without_authors["Penulis"].apply(lambda penulis: preprocess.gantiNama(penulis, benarkan_nama))
        df_without_authors['Penulis'] = df_without_authors['Penulis'].astype(str).apply(lambda x: ', '.join([item.strip() for item in x.split(',') if item.strip() != '0']))
        df_without_authors['Penulis'].replace(['', 'nan'], np.nan, inplace=True)
        df_without_authors.dropna(subset=['Penulis'], inplace=True)

        # Filter data with authors in dosen_list
        df_with_authors = combined_df[combined_df['Penulis'].apply(lambda penulis: preprocess.penulisAda(penulis, dosen_list))]

        # Combine both datasets without saving to file
        final_df = pd.concat([df_with_authors, df_without_authors])

        # Convert final dataframe to JSON format
        final_data = final_df.to_dict(orient='records')

        return jsonify(final_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-hasil-akhir', methods=['GET'])
def get_hasil_akhir():
    # Path to your JSON file
    hasil_akhir_path = 'hasil akhir.json'

    # Read the JSON data
    with open(hasil_akhir_path, 'r') as file:
        data = json.load(file)

    # Calculate the length of the data
    length = len(data)

    # Add the length to the response
    response = {
        'data': data,
        'length': length
    }

    # Return the data and length as a JSON response
    return jsonify(response)
@app.route('/get-sdgs-count', methods=['GET'])
def get_sdgs_count():
    # Path to your JSON file
    hasil_akhir_path = 'hasil akhir.json'

    # Read the JSON data
    with open(hasil_akhir_path, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to count occurrences of each SDG
    sdgs_count = {f'SDGS{i}': 0 for i in range(1, 18)}

    # Iterate through the data and count SDGs
    for item in data:
        if 'Sdgs' in item and isinstance(item['Sdgs'], list):
            for sdg in item['Sdgs']:
                if sdg in sdgs_count:
                    sdgs_count[sdg] += 1

    # Return the SDGs count as a JSON response
    return jsonify(sdgs_count)

UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_pdf_data_pymupdf(pdf_path):
    # Membuka file PDF

    doc = fitz.open(pdf_path)
    text = ""
    
    # Ekstrak teks dari semua halaman
    for page in doc:
        text += page.get_text()

    # Ekstrak teks dari halaman pertama saja
    first_page_text = doc[0].get_text()

    # Regex untuk menangkap judul setelah "homepage: www.GrowingScience.com/ijds"
    title_match = re.search(r'homepage: www\.GrowingScience\.com/ijds\s+(.+?)\s+\n', first_page_text, re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        # Menghapus karakter newline yang berlebihan
        title = re.sub(r'\n+', ' ', title).strip()
    else:
        title = "Not found"

    # Regex untuk menangkap abstrak
    abstract_match = re.search(r'A B S T R A C T(.+)', text, re.DOTALL)

    if abstract_match:
        abstract = abstract_match.group(1).strip()
        
        # Ambil semua teks setelah "Accepted:"
        accepted_match = re.search(r'Accepted:.*?\n(.*)', abstract, re.DOTALL)
        if accepted_match:
            # Ambil teks setelah "Accepted:"
            abstract_cleaned = accepted_match.group(1).strip()
            
            # Hapus semua teks setelah "\n©"
            abstract_cleaned = re.sub(r'\n©.*', '', abstract_cleaned, flags=re.DOTALL).strip()
            
            # Hapus semua teks sebelum "\n\n" (dua baris kosong)
            abstract_cleaned = re.sub(r'^.*?\n \n', '', abstract_cleaned, flags=re.DOTALL).strip()
        else:
            abstract_cleaned = "Not found"
    else:
        abstract_cleaned = "Not found"

    # Menghapus semua karakter newline (\n)
    abstract_cleaned = re.sub(r'\n+', ' ', abstract_cleaned).strip()

    accepted_date_match = re.search(r'Accepted: .*?(\d{4})', text)
    if accepted_date_match:
        accepted_date = int(accepted_date_match.group(1))
    else:
        accepted_date = "Not found"

    authors_match = re.search(r'www\.GrowingScience\.com/ijds\s+\n(.+?)\na', first_page_text, re.DOTALL)
    if authors_match:
        print("1")
        authors = authors_match.group(1).strip()
        
        # Hapus bagian "www.GrowingScience.com/ijds \n \n \n \n \n \n \n"
        authors = re.sub(r'www\.GrowingScience\.com/ijds\s', '', authors, flags=re.DOTALL).strip()

        # Hapus semua teks sebelum "\n \n \n \n"
        authors = re.sub(r'^.*?\n \n', '', authors, flags=re.DOTALL).strip()

        # Hapus semua karakter "*"
        authors = authors.replace('*', '').strip()

        # Hapus satu karakter sebelum setiap koma
        authors = re.sub(r'\s,', ',', authors)
    else:
        authors = "Not found"

    authors = re.sub(r'\n+', ' ', authors).strip()

    # Ubah "and" menjadi ","
    authors = authors.replace(" and ", ", ")

    # Hapus satu karakter sebelum setiap koma
    authors = re.sub(r'.(?=,)', '', authors)

    # Hapus satu karakter terakhir
    authors = authors[:-1]

    data = {
        "Title": title,
        "Abstract": abstract_cleaned,
        "Year": accepted_date if isinstance(accepted_date, int) else "Not found",
        "Authors": authors,
        # "first_page_text": first_page_text.strip()  # Teks dari halaman pertama saja
    }

    return data
if __name__ == "__main__":
    app.run(port=8004, debug=True)
