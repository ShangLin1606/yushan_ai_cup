from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
from config.Config import Config


config = Config()

class LangChainModel:
    def __init__(self):
        # 初始化配置和模型
        self.chat_model = ChatOpenAI(model='gpt-4o-mini', api_key=config.OPENAI_API_KEY)
        self.str_parser = StrOutputParser()

    def _generate_response(self, prompt_template, input_data: dict) -> str:
        try:
            chain = prompt_template | self.chat_model | self.str_parser
            return chain.invoke(input_data)
        except Exception as e:
            print(f"Error invoking the chain: {e}")
        return "An error occurred while processing the request."
    
    
    def get_ocr_prompt(self) -> ChatPromptTemplate:
        # 返回一個帶有客製化提示語的 prompt
        return ChatPromptTemplate.from_messages([
            ('system', '你是一個專門處理玉山銀行文檔的智能機器人，負責將 PDF 圖片通過 OCR 轉成文字的結果，整理為清晰的文字結構，供後續 RAG（檢索增強生成）使用。請遵循以下要求：1. 僅根據收到的 OCR 內容整理資訊，不新增或減少資訊。2. 僅在格式混亂或錯字情況下進行調整，保持內容忠實於 OCR 結果。3. 確保數字、日期、貨幣符號等與 OCR 結果一致，絕不修改。目標是將收到的文字整理為具有一致格式的結構化文檔，以便在後續步驟中無需進行額外的清理或修改。'),
            ('human', 'OCR內容:{input}')
        ])
    
    # def get_response(self, ocr_message):


    def get_formatted_text_from_ocr(self, ocr_message:str) -> str:

        return self._generate_response(self.get_ocr_prompt(), {"input": ocr_message})
        


    def get_document_summary_prompt(self) -> ChatPromptTemplate:
        # 返回一個帶有客製化提示語的 prompt
        return ChatPromptTemplate.from_messages([
            ('system', '你是一個專門為玉山銀行處理文檔的智能助手，負責將接收到的文檔精簡為約 500字的摘要，並在摘要中包含文檔中的關鍵字。請嚴格遵循以下要求進行摘要：1.僅根據提供的文檔內容進行摘要：禁止添加或推測任何額外信息。2.確保摘要與原文內容的意思一致：保持信息完整和準確，不得扭曲或省略重要細節。3.包含並強調文檔中的重要關鍵字：將重要的專業術語、公司名稱、數字、日期、貨幣符號等準確呈現於摘要中，確保摘要易於檢索和理解。4.精確保留數字、日期、貨幣符號等：摘要中的數字、日期和貨幣符號必須與原始文檔一致，不可更改。5.保持專業和清晰的語氣：用簡明的語句撰寫摘要，保證格式統一、結構清晰。6.不偏離原文：務必維持與原文信息相符的語意，並包含重點用詞。你的目標是將接收到的文檔轉換為格式統一、結構清晰、包含關鍵字的摘要，以便於後續的處理和檢索增強生成（RAG）工作流程。'),
            ('human', '文檔內容:{input}')
        ])
    
    # def get_response(self, ocr_message):


    def get_document_summary(self, document_message:str) -> str:

        return self._generate_response(self.get_document_summary_prompt(), {"input": document_message})


    def get_response_prompt(self) -> ChatPromptTemplate:
        # 返回一個帶有客製化提示語的 prompt
        return ChatPromptTemplate.from_messages([
            ('system', '你是一個專門為玉山銀行客戶提供精確回答的智能助手，透過檢索增強生成（RAG）技術回應客戶的提問。請根據以下內容回覆客戶問題，僅根據提供的信息進行回答：\n\n{rag_content}\n\n請遵循以下要求：'
                   '1. 嚴格依據提供的 RAG 內容，不添加或推測額外信息。'
                   '2. 確保所有數字、日期、貨幣符號等細節保持一致，不進行任何修改。'
                   '3. 回答需簡潔、清晰，確保信息準確無誤。'),
            ('human', '客戶問題: {input}')
        ])
    


    def get_response(self, rag_content:str, human_message:str) -> str:

        if not rag_content or not human_message:
            print("Empty input detected.")
            print(rag_content)
            return "Invalid input provided."
    
        return self._generate_response(self.get_response_prompt(), {
            "rag_content": rag_content,
            "input": human_message
        })



        