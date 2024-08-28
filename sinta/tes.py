import os
import datetime

def create_empty_file():
    # Tentukan tahun ini dan tahun lalu
    tahun_ini = datetime.datetime.now().year
    tahun_lalu = tahun_ini - 1

    # Tentukan nama file
    filename = f"{tahun_lalu}_{tahun_ini}_sinta_scrapping.csv"

    # Direktori penyimpanan
    storage_dir = os.path.join(os.getcwd(), "sinta/storage/result/scrappingSinta")

    # Pastikan direktori storage ada
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Buat file kosong
    file_path = os.path.join(storage_dir, filename)
    with open(file_path, 'w') as file:
        pass  # Membuat file kosong tanpa konten

    return file_path

# Contoh penggunaan
file_path = create_empty_file()
print(f"File kosong telah dibuat di: {file_path}")
