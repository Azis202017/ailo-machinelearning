from dataclasses import dataclass
from bs4 import BeautifulSoup
import requests
import re
import time

@dataclass
class AdvancedSearchType:
    SKRIPSI: int = 4 #S1
    TA: int = 6  #D3
    THESIS: int = 5 #S2

class OpenLibrary:
    base_url: str = "https://openlibrary.telkomuniversity.ac.id"
    
    def __init__(self):
        self.session = requests.Session()

    def get_all_data_from_range_date(self, **search_options) -> str:
        print(f'{self.base_url}/home/catalog.html')
        response = self.session.post(f'{self.base_url}/home/catalog.html', data=search_options)
        response.raise_for_status()

        return response.text
    
    def get_pagination(self, content: str) -> list:
        parsed = BeautifulSoup(content, "html.parser")
        
        paginations = parsed \
            .find("div", class_="pagination-imtelkom") \
            .find_all("li")
        
        # find last-page
        result = []
        url_last_page = paginations[-1].find('a').get('href')
        number_last_page = int(re.search(r'(\d+)\.html', url_last_page).group(1))

        for i in range(1 , number_last_page+1):
            url = self.base_url + f'/home/catalog/page/{i}.html'
            result.append(url)

        print('=' * 32)
        print(f'Ready to scarping {number_last_page} page from Open Library Telkom University')
        print('=' * 32)
        return result
    
    def get_search_result(self, content: str) -> list:
        paginations = self.get_pagination(content)
        
        results = []
        
        for pagination in paginations:
            content = self.session.get(pagination).text
            
            parsed = BeautifulSoup(content, "html.parser")
                
            search_results = parsed.find("div", class_="row row-imtelkom") \
                                   .find("div", class_="col-md-9") \
                                   .find_all("div", class_="col-md-6 col-sm-6 col-xs-12")
            
            pages = []
            
            for result in search_results:
                result_url = result \
                                .find('div', class_='media-body') \
                                .find('h4', class_='media-heading') \
                                .find('a') \
                                .get('href')
                
                pages.append(self.base_url + result_url)
                    
            results.extend(pages)
                    
        return results
    
    def parse_results(self, content: str) -> set:
        urls = self.get_search_result(content)
        length = len(urls)
        
        for i in range(len(urls)):   
            time.sleep(2)         
            response = self.session.get(urls[i]).text
            
            yield i+1, length, self.parse_result(response)
            print(f'Scarping {i+1} data')
            
    def parse_result(self, content):
        parsed = BeautifulSoup(content, "html.parser")

        header = parsed \
                    .find("div", class_="page-header page-header-imtelkom") \
                    .find("h1") \
                    .find(text=True, recursive=False)

        result = {}
        result["title"] = header

        catalog_attributes = parsed.find_all("div", class_="catalog-attributes")

        general_information = catalog_attributes[0] \
                            .find("div", class_="col-md-3 col-sm-8 col-xs-12")

        classification = general_information\
                    .find_all("p")[-3]\
                    .get_text()\
                    .strip()
        result["classification"] = self.remove_html_tags(classification)

        type_publication = general_information\
                    .find_all("p")[-2]\
                    .get_text()\
                    .strip()
        result["type_publication"] = self.remove_html_tags(type_publication)

        subject = general_information\
                    .find_all("p")[-1] \
                    .get_text() \
                    .strip()
        result["subject"] = self.remove_html_tags(subject)

        try:
            abstract_section = parsed.find_all("div", class_="abstract")
            keywords = abstract_section[-1].get_text().strip()

            abstract = abstract_section[:-1]
            abstract = "\n".join([a.get_text().strip() for a in abstract])

            result["abstract"] = self.remove_html_tags(abstract)
            result["keywords"] = self.remove_html_tags(keywords)
        except (AttributeError, IndexError) as e:
            result["abstract"] = ""
            result["keywords"] = ""

        if not result["abstract"]:
            try:
                col_md_section = parsed.find("div", class_="col-md-7 col-sm-12 col-xs-12")
                abstract = col_md_section.find_all("p")[1].get_text().strip()
                result["abstract"] = self.remove_html_tags(abstract)
            except (AttributeError, IndexError) as e:
                print(f"Error parsing alternative abstract: {e}")
                result["abstract"] = ""

        authors_info, publisher_info, _ = catalog_attributes[1].find_all("div", class_="col-md-4 col-sm-4 col-xs-12")  

        get_table_row = lambda elm: elm.find("table").find_all("tr")
        parse_elm = lambda elm: elm.find_all("td")[1].get_text().strip()

        author, type, lecturer, translator = get_table_row(authors_info)
        publisher_name, publisher_city, publish_year = get_table_row(publisher_info)

        result["author"] = parse_elm(author)
        result["lecturer"] = parse_elm(lecturer)
        result["publisher"] = parse_elm(publisher_name)
        result["publish_year"] = parse_elm(publish_year)

        return result

    def remove_html_tags(self, text: str) -> str:
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)