import os
import argparse
from typing import Tuple
from tqdm import tqdm

from controllers.InitializationController import InitializationController
from controllers.QuestionController import QuestionController
from models.PDFProcessor import PDFTextImageExtractor
from models.JsonProcessor import JsonProcessor
from models.DocumentManager import DocumentManager
from models.FAISSIndexManager import FAISSIndexManager
from models.LangChainModel import LangChainModel

import json

def initialize_components(source_path:str, category:str) -> Tuple[PDFTextImageExtractor, JsonProcessor, DocumentManager, FAISSIndexManager]:
    pdf_processor = PDFTextImageExtractor(category, source_path)
    json_processor = JsonProcessor(category, source_path)
    document_manager = DocumentManager(category)
    faiss_index_manager = FAISSIndexManager(category)
    return pdf_processor, json_processor, document_manager, faiss_index_manager

def run_initialization(category:str, pdf_processor:PDFTextImageExtractor, json_processor:JsonProcessor, document_manager:DocumentManager, faiss_index_manager:FAISSIndexManager):
    # try:
    init_controller = InitializationController(category, pdf_processor, json_processor, document_manager, faiss_index_manager)
    init_controller.initialize_data_and_index()
    # except Exception as e:
    #     print(f"Error during initialization: {e}")

def answer_question(human_question:str, category:str, faiss_index_manager:FAISSIndexManager, llm_model:LangChainModel):
    question_controller = QuestionController(faiss_index_manager, llm_model)
    response, sources_num = question_controller.handle_question(human_question, category)
    return response, sources_num

def calculate_accuracy():
    print('calculate_accuracy')
    with open('data/dataset/preliminary/ground_truths_example.json', 'r', encoding = 'utf-8') as f:
        ground_truths = json.load(f)

    with open('data/model_output/model_output.json', 'r', encoding = 'utf-8') as f:
        model_output = json.load(f)

    ground_truths_dict = {}
    for item in ground_truths['ground_truths']:
        qid = item['qid']
        retrieve = item['retrieve']
        ground_truths_dict[qid] = retrieve
        
    model_output_dict = {}
    for item in model_output['ansewers']:
        qid = item['qid']
        retrieve = int(item['retrieve'])
        model_output_dict[qid] =  retrieve

    total_count = len(model_output_dict)
    matching_count = 0
    for qid in model_output_dict:
        if qid in ground_truths_dict and model_output_dict[qid] == ground_truths_dict[qid]:
            matching_count += 1
    
    if total_count > 0:
        matching_ratio = round(matching_count/total_count, 2)
    
    print("總數量:", total_count)
    print("相同的 retrieve 數量:", matching_count)
    print(f"相同的比例: {matching_ratio}")
    
def main():
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    args = parser.parse_args()


    # 第一步:初始化
    if True:
        question_categories = ['finance', 'insurance']
        faiss_index_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        for category in tqdm(question_categories, desc="Building FAISS Indexes for Each Category"):
            if not os.path.exists(os.path.join(faiss_index_file_path, f"{category}_faiss_index")):
                pdf_processor, json_processor, document_manager, faiss_index_manager = initialize_components(args.source_path, category)
                run_initialization(category, pdf_processor, json_processor, document_manager, faiss_index_manager)
                # print(f'The {category} FAISS index has been created.')
            else:
                # print(f'The {category} FAISS index already exists.')
                pass
        print(f'***All FAISS indexes are ready.***') 

    
    #第二步:載入問題
    if True:
        with open(args.question_path, 'r', encoding='utf-8') as f:
            question_file = json.load(f)
        
        output = []
        faiss_index_manager = FAISSIndexManager()
        llm_model = LangChainModel()
        
        final_result = {}
        
        for question in tqdm(question_file['questions'], desc="Loading questions: Finding nearest document and LLM response"):
            result = {}
            qid = question.get('qid')
            query = question.get('query')
            query_category = question.get('category')

            response, sources_num = answer_question(query, query_category, faiss_index_manager, llm_model)

            result['qid'] = qid
            result['retrieve'] = int(sources_num[0])
            result['query'] = query
            result['category'] = query_category
            result['response'] = response


            output.append(result)
        
        final_result['ansewers'] = output
        
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_output'), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        
        print(f'***Result already exists in the path ({args.output_path}).***') 

    #第三步:測試output正確率
    if True:
        calculate_accuracy()

        #python app.py --question_path "data/dataset/preliminary/questions_example.json" --source_path "data/reference" --output_path "data/model_output/model_output.json"
        #長requirement、readme檔案、改檔名class駝峰、snake
if __name__ == "__main__":
    main()



            

            

            

