import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from typing import Tuple, List

class QuestionController:
    def __init__(self, faiss_index_manager, llm_model):
        self.faiss_index_manager = faiss_index_manager
        self.llm_model = llm_model

    def handle_question(self, human_question:str, category:str) -> Tuple[str, List[str]]:
        if not human_question.strip() or len(human_question) < 3:
            print("Received empty or invalid question.")
            return "Invalid input.", []
        
        # try:
        documents_context, sources = self.faiss_index_manager.search(human_question, category)
        response = self.llm_model.get_response(documents_context, human_question)
        return response, sources
        # except Exception as e:
        #     print(f"Error handling question '{human_question}': {e}")
        #     return "An error occurred while processing the question.", []



