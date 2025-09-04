import os
import json
from langchain_core.documents import Document
from models.LangChainModel import LangChainModel
from models.CorpusManager import CorpusManager


class DocumentManager:
    def __init__(self, category: str):
        self.category = category
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'formatted_data')
        self.file_name = f"{self.file_path}/{self.category}_summarized_documents.json"
        self.llm_model = LangChainModel()
        self.corpusmanager = CorpusManager(self.category)

    def create_summarized_documents(self) -> list[Document]:
        if not os.path.exists(self.file_name):
            """Creates and saves summarized documents based on the category."""
            corpus_dict = self.corpusmanager.load_corpus()
            
            documents = []
            json_documents = []

            for filename, content in corpus_dict.items():
                #faq資料很短，不需要進行摘要
                if self.category == 'faq':
                    summarized_context = content
                else:
                    summarized_context = self.llm_model.get_document_summary(content)
                    
                document = Document(page_content=summarized_context, metadata={"source": filename, "qa_category": self.category})
                documents.append(document)
                
                json_document = {
                    "page_content": summarized_context,
                    "metadata": {"source": filename, "qa_category": self.category}
                }
                json_documents.append(json_document)
        
            self.save_summarized_documents_to_json(json_documents)
            # print(f"{self.category}_summarized_json_documents have been saved to the path {self.file_path}.")
            return documents
        else:
            # print(f"{self.category}_summarized_documents already exists in {self.file_path}.")
            return self.get_summarized_documents()
    

    def get_summarized_documents(self) -> list[Document]:
        """Loads summarized documents if they exist, otherwise returns None."""
        if os.path.exists(self.file_name):
            try:
                with open(self.file_name, 'r', encoding='utf-8') as f:
                    json_documents = json.load(f)
                    documents = [
                    Document(
                        page_content=doc["page_content"],
                        metadata=doc["metadata"]
                    )
                    for doc in json_documents
                ]
                return documents
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {self.file_name}.")
                return []
        else:
            # print(f"The file {self.category}_summarized_documents does not exist. Creating now, please wait.")
            documents = self.create_summarized_documents()
            return documents

    def save_summarized_documents_to_json(self, json_documents: list):
        """Saves summarized documents to a JSON file."""
        try:
            with open(self.file_name, 'w', encoding='utf-8') as f:
                json.dump(json_documents, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving file {self.file_name}: {e}")

if __name__ == '__main__':
    c_s = ['insurance', 'finance']
    for c in c_s:
        documents_manager = DocumentManager(c)
        documents_manager.create_summarized_documents()