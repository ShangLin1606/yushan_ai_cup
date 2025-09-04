import json
import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from models.CorpusManager import CorpusManager


class JsonProcessor:
    def __init__(self, category:str, source_path:str):
        self.category = category

        self.category_source_path =  os.path.join(source_path, category)
        if not os.path.exists(self.category_source_path) or not os.listdir(self.category_source_path):
            raise FileNotFoundError(f"Reference data is missing, the path is incorrect, or the directory is empty: {self.category_source_path}")

        self.corpusmanager = CorpusManager(category)
        
    def create_and_save_json_content(self) -> dict:
        formatted_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'formatted_data')
        os.makedirs(formatted_data_path, exist_ok=True)
        
        if not os.path.exists(f"{formatted_data_path}/{self.category}_all_text.json"):
            file_ls = self._get_json_file_names()

            corpus_dict_all = {}
            for file in file_ls:
                file_name = os.path.join(self.category_source_path, file)
                corpus_dict = self._extract_qa_from_json(file_name)
                corpus_dict_all.update(corpus_dict)
            self.corpusmanager.save_corpus(corpus_dict_all)
            # print(f'The {self.category} Json data has been extracted and returned as a dictionary, and saved to data/formatted_data/.') 
            return  corpus_dict_all

        else:
            # print(f'The {self.category} Json context dictionary already exists and has been returned from data/formatted_data/ as a dictionary.')
            corpus_dict_all = self.corpusmanager.load_corpus()
            return  corpus_dict_all
        
    def _get_json_file_names(self) -> list[str]:
        return [f for f in os.listdir(self.category_source_path) if f.endswith('.json')]

    def _extract_qa_from_json(self, file_name:str) -> dict:
        corpus_dict = {}
        with open(file_name, 'r', encoding='utf-8') as f:
           json_file = json.load(f)

        for key, values in json_file.items():
            each_faq_str = ''
            for value in values:
                question = value.get('question', '')
                answers = ' '.join(value.get('answers', []))
                each_faq_str += f"Q:{question},Ans:{answers}"
                each_faq_str += '\n---\n'
            corpus_dict[key] = each_faq_str.strip()
        return corpus_dict

if __name__=='__main__':
    category = 'faq'
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'reference')
    corpus_m = JsonProcessor(category, file_path)
    corpus_m.create_and_save_json_content()

    
   