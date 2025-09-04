from langchain_huggingface import HuggingFaceEmbeddings
from models.DocumentManager import DocumentManager
from models.CorpusManager import CorpusManager

import os

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from operator import itemgetter

class FAISSIndexManager:
    def __init__(self, create_index_category:str=None):
        self.embedding_model = HuggingFaceEmbeddings(model_name="moka-ai/m3e-large")
        self.index_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.create_index_category = create_index_category

    def create_index(self, category:str=None):
        if category is None:
            if self.create_index_category is None:
                raise ValueError("You must provide a category either as an argument or during initialization.")

            category = self.create_index_category

        
        # 載入或創建 FAISS 索引
        # 檢查是否已存在本地索引
        index_path = os.path.join(self.index_directory, f"{category}_faiss_index")

        if os.path.exists(index_path):
            # 從本地載入 FAISS 索引
            # print(f"The {category} FAISS index already exists.")
            pass

        else:
            # 如果不存在索引，則創建新的 FAISS 索引
            # print(f"Creating a new {category} FAISS index...")
            document_manager = DocumentManager(category)

            # 設置索引的維度
            summarized_documents = document_manager.get_summarized_documents()
            
            if not summarized_documents:
                summarized_documents = document_manager.create_summarized_documents()

            #給範例讓 FAISS 知道索引的維度
            dimension = len(self.embedding_model.embed_query(summarized_documents[0].page_content))  # 確定向量的維度
            index = faiss.IndexFlatL2(dimension)
            
            # 創建向量存儲
            vector_store = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=InMemoryDocstore({}),  # 基於內存的文檔儲存
                index_to_docstore_id={}  # 空字典初始化，保存向量索引到文檔的映射
            )

            #加載文檔並生成唯一 ID
            uuids = [str(uuid4()) for _ in range(len(summarized_documents))]

            # 添加文檔到向量存儲
            vector_store.add_documents(documents=summarized_documents, ids=uuids)

            #將 FAISS 索引保存到本地
            vector_store.save_local(f"{self.index_directory}/{category}_faiss_index")

            # print(f"Completed creating the {category} FAISS index.")
 

    def search(self, query:str, question_category:str) -> tuple[str, list]:
        if not os.path.exists(f"{self.index_directory}/{question_category}_faiss_index"):
            # 從本地載入 FAISS 索引
            # print(f"The {question_category} FAISS index does not exist. Creating now, please wait.")
            self.create_index(question_category)

        try:
            vector_store = FAISS.load_local(
                    f"{self.index_directory}/{question_category}_faiss_index",
                    embeddings=self.embedding_model,
                    # 僅當來源可信時使用此選項
                    allow_dangerous_deserialization=True
                )
            
        except Exception as e:
            print(f"Error loading the FAISS index: {e}")
            return None, None

        
        search_results = vector_store.similarity_search_with_score(query, k=5)
        corpus_manager = CorpusManager(question_category)
        
        corpus_dict = corpus_manager.load_corpus(question_category)

        high_score_documents = []
        sources_num = []
        
        #目前玉山競賽只要選出唯一值
        # for document, score in search_results:
        #     if score > 0.9:
        #         source = document.metadata['source']
        #         high_score_documents.append(corpus_dict.get(source, ''))
        #         sources_num.append(source)
        
        if high_score_documents:
            combined_documents_context = '\n---\n'.join(high_score_documents)
            
            return combined_documents_context, sources_num
        
        else:
            max_score_document = max(search_results, key=itemgetter(1))
            documents_context = corpus_dict.get(max_score_document[0].metadata['source'], '')
            sources_num.append(max_score_document[0].metadata['source'])
            
            return documents_context, sources_num