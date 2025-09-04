import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

class InitializationController:
    def __init__(self, category, pdf_processor, json_processor, document_manager, faiss_index_manager):
        self.category = category
        self.pdf_processor = pdf_processor
        self.json_processor = json_processor
        self.document_manager = document_manager
        self.faiss_index_manager = faiss_index_manager

    def initialize_data_and_index(self):
        # try:
        if self.category == 'faq':
            self.json_processor.create_and_save_json_content()

        else:
            self.pdf_processor.create_and_save_pdf_content()
            
        self.document_manager.create_summarized_documents()
        self.faiss_index_manager.create_index()

        # except Exception as e:
        #     print(f"Error during initialization: {e}")
