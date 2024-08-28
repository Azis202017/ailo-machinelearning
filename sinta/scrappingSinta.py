from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import datetime
import random
import undetected_chromedriver as uc
import os

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

class WebScraper:
    def __init__(self):
        self.driver = self.create_driver()

    def random_delay(self, min_seconds=2, max_seconds=5):
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_random_user_agent(self):
        return random.choice(user_agents)

    def create_driver(self):
        download_dir = os.path.abspath("storage/result/scrappingSinta")  # Gunakan path absolut
        
        # Pastikan direktori storage ada
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Setel preferensi untuk Chrome
        options = uc.ChromeOptions()
        prefs = {
            "download.default_directory": download_dir,  # Tentukan lokasi unduhan
            "download.prompt_for_download": False,       # Nonaktifkan prompt unduhan
            "directory_upgrade": True,                   # Perbarui direktori unduhan
            "safebrowsing.enabled": True                 # Aktifkan safe browsing
        }
        
        # Menambahkan preferensi ke opsi Chrome
        options.add_experimental_option("prefs", prefs)
        
        # Menambahkan opsi lain ke Chrome
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(f'user-agent={self.get_random_user_agent()}')

        # Membuat driver dengan opsi yang sudah disiapkan
        driver = uc.Chrome(options=options)
        return driver

    def perform_human_interaction(self):
        actions = webdriver.ActionChains(self.driver)
        actions.move_by_offset(random.randint(0, 100), random.randint(0, 100))
        actions.perform()   
        self.random_delay()
        self.driver.execute_script("window.scrollBy(0, {});".format(random.randint(200, 800)))
        self.random_delay()

    def login_sinta(self, username, password):
        self.driver.get("https://sinta.kemdikbud.go.id/logins")
        self.driver.find_element(By.NAME, "username").send_keys(username)
        self.driver.find_element(By.NAME, "password").send_keys(password)
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    def login_elsevier(self, username, password):
        self.driver.get("https://id.elsevier.com/as/authorization.oauth2?platSite=SC%2Fscopus&ui_locales=en-US&scope=openid+profile+email+els_auth_info+els_analytics_info+urn%3Acom%3Aelsevier%3Aidp%3Apolicy%3Aproduct%3Aindv_identity&els_policy=idp_policy_indv_identity_plus&response_type=code&redirect_uri=https%3A%2F%2Fwww.scopus.com%2Fauthredirect.uri%3FtxGid%3De5949ec1f7f8942be40f031fec9c4705&state=userLogin%7CtxId%3DBFEEEC06342ACB062CC06964CAAFD770.i-091fb6f4d2a483d2a%3A5&authType=SINGLE_SIGN_IN&prompt=login&client_id=SCOPUS")
        WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))).click()
        # self.random_delay()
        time.sleep(2)
        self.driver.find_element(By.ID, "bdd-email").send_keys(username)
        # self.random_delay()
        self.driver.find_element(By.CSS_SELECTOR, "button[value='emailContinue']").click()
        
        WebDriverWait(self.driver, 20).until(EC.visibility_of_element_located((By.ID, "bdd-password")))
        # self.random_delay()
        time.sleep(1)
        self.driver.find_element(By.ID, "bdd-password").send_keys(password)
        
        WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[value='signin']")))
        # self.random_delay()
        self.driver.find_element(By.CSS_SELECTOR, "button[value='signin']").click()

    def get_article_links(self, url, num_pages):
        current_year = datetime.datetime.now().year
        target_year = current_year - 1
        done = False

        self.driver.get(url)
        WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "ar-title")))
        
        all_article_links = []
        all_years = []

        for _ in range(num_pages):
            x = 0
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            years = []
            ar_years = soup.find_all('a', class_='ar-year')
            for ar_year in ar_years:
                x += 1
                year = int(ar_year.text.strip())
                if year <= target_year:
                    x -= 1
                    done = True
                    break
                years.append(year)
            
            all_years.extend(years)
            article_links = []
            ar_titles = soup.find_all('div', class_='ar-title')
            for i in range(x):
                link = ar_titles[i].find('a')['href']
                article_links.append(link)
            
            all_article_links.extend(article_links)
            if done:
                break
            if any(year <= target_year for year in years):
                all_years.extend(years)
                break
            try:
                self.driver.find_element(By.XPATH, "//a[contains(@class,'page-link') and contains(text(),'Next')]").click()
            except:
                break
        return all_article_links, all_years

    def scrape_article(self, article_links, article_years):
        result = {
            "judul": [],
            "penulis": [],
            "tahun": [],
            "sdgs": [],
            "abstrak": []
        }

        judul = []
        penulis = []
        abstrak = []
        sdgs = []

        for link in article_links:
            self.driver.get(link)
            self.random_delay()
            self.perform_human_interaction()
            penulisBanyak = []
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            judul.append([h2.get_text(strip=True) for h2 in soup.find_all('h2', class_='Typography-module__lVnit Typography-module__o9yMJ Typography-module__JqXS9 Typography-module__ETlt8')])
            abstrak.append([p.get_text(strip=True) for p in soup.find_all('p', class_='Typography-module__lVnit Typography-module__ETlt8 Typography-module__GK8Sg')])
            sdgs.append([div.get_text(strip=True) for div in soup.find_all('div', class_='Col-module__hwM1N')])
            div_elements = soup.find_all('ul', class_='DocumentHeader-module__LpsWx')
            for div_element in div_elements:
                spans = div_element.find_all('span', class_='Typography-module__lVnit Typography-module__Nfgvc Button-module__Imdmt')
                for span in spans:
                    penulisBanyak.append(span.text)
            penulis.append(";".join(penulisBanyak))
        result["judul"] = judul
        result["penulis"] = penulis
        result["tahun"] = article_years
        result["abstrak"] = abstrak
        result["sdgs"] = sdgs
        return result

    def save_to_json(self, result):
        df = pd.DataFrame(result)
        file = f'{datetime.datetime.now().strftime("%Y-%m-%d")}_scrappingSinta.json'
        if not os.path.exists('./storage/result/scrappingSinta/'):
            os.makedirs('./storage/result/scrappingSinta/')
            
        df.to_json(f'./storage/result/scrappingSinta/{file}', orient='records')
        print("Crawl Success")
    
    def download(self, yearFrom):
        self.driver.get("https://www.scopus.com/results/results.uri?sort=plf-f&src=s&sid=6e7daf710aa80d4e278bd93e88ffc986&sot=aff&sdt=a&sl=34&s=AF-ID%2860103730%29+OR+AF-ID%2860114126%29&origin=AffiliationProfile&editSaveSearch=&txGid=38d11def69c7093f01c787438abcf607&sessionSearchId=6e7daf710aa80d4e278bd93e88ffc986&limit=10")
        time.sleep(5)
        self.driver.find_element(By.CSS_SELECTOR, "input[data-testid='input-range-from']").send_keys(yearFrom)
        self.driver.find_element(By.CSS_SELECTOR, "label[data-testid='radio-button']").click()
        time.sleep(1)
        self.driver.find_element(By.CSS_SELECTOR, "button[data-testid='apply-facet-range']").click()
        time.sleep(3)
        # self.driver.find_element(By.ID, "bulkSelectDocument-primary-document-search-results-toolbar").click() # 10 artikel or no
        self.driver.find_element(By.CSS_SELECTOR, "div.export-dropdown").click()
        time.sleep(1)
        # Mengklik tombol Export
        self.driver.find_element(By.CSS_SELECTOR, "button[data-testid='export-to-csv']").click()
        time.sleep(1)
        # Mengklik label berdasarkan nilai aria-controls
        self.driver.find_element(By.XPATH, "//label[@aria-controls='field_group_authors field_group_titles field_group_year field_group_eid field_group_sourceTitle field_group_volumeIssuePages field_group_citedBy field_group_sourceDocumentType field_group_publicationStage field_group_doi field_group_openAccess']").click()
        self.driver.find_element(By.ID, "field_group_authors").click()
        # time.sleep(1)
        self.driver.find_element(By.ID, "field_group_titles").click()
        # time.sleep(1)
        self.driver.find_element(By.ID, "field_group_year").click()
        # time.sleep(1)
        self.driver.find_element(By.ID, "field_group_abstact").click()
        time.sleep(1)
        # Mengklik tombol berdasarkan data-testid
        self.driver.find_element(By.CSS_SELECTOR, "button[data-testid='submit-export-button']").click()
        time.sleep(10)
        download_dir = os.path.abspath("storage/result/scrappingSinta")
        self.wait_for_download(download_dir)
        # Panggil fungsi rename untuk mengganti nama file
        current_date = datetime.date.today()
        end_day, end_month, end_year = current_date.day, current_date.month, current_date.year
        new_filename = f'crawleddSinta{end_day}-{end_month}-{end_year}.csv'
        self.rename_downloaded_file("storage/result/scrappingSinta", new_filename)
        
    def wait_for_download(self, download_dir):
        seconds = 0
        dl_wait = True
        while dl_wait and seconds < 180:  # Mengatur batas waktu tunggu selama 60 detik
            time.sleep(1)
            dl_wait = False
            for fname in os.listdir(download_dir):
                if fname.endswith('.crdownload'):  # Jika file .crdownload masih ada, maka unduhan belum selesai
                    dl_wait = True
            seconds += 1
        return seconds < 60
    
    def rename_downloaded_file(self, download_dir, new_filename):
        # Monitor direktori unduhan untuk file baru
        # time.sleep(5)  # Berikan waktu untuk memastikan file sudah diunduh
        files = os.listdir(download_dir)
        files = [f for f in files if not f.endswith('.crdownload')]  # Abaikan file yang belum selesai diunduh

        if files:
            latest_file = max([os.path.join(download_dir, f) for f in files], key=os.path.getctime)
            new_file_path = os.path.join(download_dir, new_filename)
            os.rename(latest_file, new_file_path)
            print(f"File berhasil diubah namanya menjadi: {new_file_path}")
        else:
            print("Tidak ada file yang ditemukan di direktori unduhan.")

    def run(self, sinta_credentials, elsevier_credentials, sinta_url, num_pages):
        # username_sinta, password_sinta = sinta_credentials
        username_elsevier, password_elsevier = elsevier_credentials
        
        # self.login_sinta(username_sinta, password_sinta)
        # article_links, article_years = self.get_article_links(sinta_url, num_pages)
        # print(article_links)
        # # print(article_years)
        
        # self.driver.quit()  # Quit and create a new driver for Elsevier login

        self.driver = self.create_driver()
        self.login_elsevier(username_elsevier, password_elsevier)
        current_date = datetime.date.today()
        yearFrom = current_date.year - 1

        self.download(yearFrom)

        # result = self.scrape_article(article_links, article_years)
        # self.save_to_json(result)
        # subprocess.run(["python", "preprocessSinta/preprocessSinta.py"])
        self.driver.quit()
