import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

class PreprocessLibrary:

    def cleaningAbstrak(self, text):
        text = str(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.lower()
        text = re.sub(r'http\\S+', '', text)
        text = re.sub('(@\\w+|#\\w+)', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub("\\n", " ", text)
        text = re.sub('(s{2,})', ' ', text)
        return text

    def cleaningJudul(self, text):
        text = str(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.title()
        text = re.sub("\\n", " ", text)
        text = re.sub('(s{2,})', ' ', text)
        return text

    def cleaningPenulis(self, text):
        text = text.title()

        text = re.sub("\n"," ", text)
        # Daftar gelar yang akan dihapus
        gelar = ["Dr.", "Drs.", "Ir.", "S.T.,", "Prof.", "Prof. Dr.", "S.H.", "S.Kom.", "S.T.", "S.Sos.", "S.Pd.", "S.E.", "M.Si.", "M.Kom.", "M.T.", "M.Pd.", "M.A.", "M.Sc.", "M.Eng.", "Ph.D.", "Ed.D.", "S.Psi.", "M.Psi.", "Drs.", "Dra.", "Dra. Hj.", "Drs. H.", "Dra. Hj.", "Ir. H.", "Ir. Hj.", "Prof. Dr. H.", "Prof. Dr. Hj.", "St, Mt"]

        # Menghapus gelar
        for g in gelar:
            text = re.sub(r'\b{}\b'.format(re.escape(g)), '', text)

        text = re.sub("(s{2,})"," ", text)

        # Pola regex untuk menghapus gelar
        gelar_pattern = r',?\s*(S\.T\.|M\.T\.|Dr|Prof|Ir|Sp|M\.Sc|Ph\.D|S\.Si|M\.Kom|S\.Kom|\.Amd|\.Sp\.|\.K\.)'
        
        # Menghilangkan gelar dari string penulis
        text = re.sub(gelar_pattern, '', text)
        
        # Menghapus spasi berlebihan dan tanda koma yang tidak diperlukan
        text = re.sub(r'\s*,\s*', ', ', text)
        text = text.strip(', ')

        return text
    
    def penulisGaAda(self, penulis, dosen_list):
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        if set(penulis_list) & set(dosen_list):
            return False
        return True
    
    def gantiNama(self, penulis, benarkan_nama):
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        penulis_list_baru = [benarkan_nama.get(nama.strip(), np.nan) for nama in penulis_list]
        # Jika ada yang tidak ditemukan (nan), tetap masukkan nan
        return ', '.join([nama if pd.notna(nama) else '0' for nama in penulis_list_baru])
    
    def penulisAda(self, penulis, dosen_list):
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        if set(penulis_list) & set(dosen_list):
            return True
        return False