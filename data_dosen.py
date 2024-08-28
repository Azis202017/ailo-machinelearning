import pandas as pd
import re
def clean_phone_number(phone):
    if pd.isna(phone):
        return ''
    # Menghapus semua karakter non-digit dan mengambil 10 digit pertama
    cleaned_phone = re.sub(r'\D', '', phone)
    return cleaned_phone[:10] if len(cleaned_phone) > 10 else cleaned_phone


def select_single_email(email):
    if pd.isna(email):
        return None
    # Memilih email pertama jika ada lebih dari satu, dipisahkan dengan '#', ';' atau ','
    return re.split(r'[;,#]', email)[0].strip()


def read_csv_to_json():
    try:
        # Membaca file CSV dengan delimiter ';'
        df = pd.read_csv('DataDosen.csv', delimiter=';', on_bad_lines='skip')
        # Mengganti nama kolom sesuai dengan format yang diinginkan
        df.columns = ['no', 'nip', 'fronttitle', 'nama_lengkap', 'backtitle', 'jenis_kelamin', 'kode_dosen',
                      'nidn', 'jafung', 'lokasi_kerja', 'jabatan_struktural', 'status_pegawai',
                      'email', 'no_hp', 'lokasi_kerja_sotk']

        # Membersihkan nomor HP dan email
        df['no_hp'] = df['no_hp'].apply(clean_phone_number)
        df['email'] = df['email'].apply(select_single_email)

        # Mengonversi DataFrame ke format JSON
        json_data = df.to_dict(orient='records')
        return json_data
    except pd.errors.ParserError as e:
        return {'error': 'Error reading CSV file', 'details': str(e)}
    except KeyError as e:
        return {'error': 'Column not found', 'details': str(e)}
