import json
import os


class CorpusManager:
    def __init__(self, category:str):
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'formatted_data')
        self.category = category

    def save_corpus(self, corpus_dict:dict) -> bool:
        # try:
        os.makedirs(self.file_path, exist_ok=True)
        file_name = os.path.join(self.file_path, f'{self.category}_all_text.json')

        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as file:
                try:
                    corpus_dict_all = json.load(file)
                except json.JSONDecodeError:
                    print(f"Warning: {file_name} is not properly formatted and has been reset to an empty dictionary.")
                    corpus_dict_all = {}
                    return False
        
            corpus_dict_all.update(corpus_dict)
        
        else:
            corpus_dict_all = corpus_dict

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(corpus_dict_all, f, ensure_ascii=False, indent=4)
        
        return True
    
        # except Exception as e:
            # print(f"Error saving corpus: {e}")
            
            # return False
            
    def load_corpus(self, category:str = None) -> dict:
        if category is None:
            category = self.category

        file_name = os.path.join(self.file_path, f'{category}_all_text.json')
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {file_name} is not properly formatted.")
                return {}
        else:
            source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'reference')
            if category == 'faq':
                from models.PDFProcessor import PDFTextImageExtractor
                pdf_processor = PDFTextImageExtractor(category, source_path)
                corpus_dict = pdf_processor.create_and_save_pdf_content()
            else:
                from models.JsonProcessor import JsonProcessor
                json_processor = JsonProcessor(category, source_path)
                corpus_dict = json_processor.create_and_save_json_content()
            return corpus_dict

    