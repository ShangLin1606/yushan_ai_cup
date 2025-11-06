# Yushan AI Cup — 金融法金文件檢索問答系統（RAG）

**角色**：資料科學家 / 系統實作  
**定位**：以檔案處理 + 向量檢索 + LLM 生成為核心的金融/保險/FAQ 分類問答系統，符合玉山 AI Cup 題目輸出格式。

---

## 目標
- 將競賽提供的 **參考資料（PDF / JSON）** 轉為可檢索的知識庫  
- 建立 **FAISS 向量索引**（依類別：`faq`、`finance`、`insurance`）  
- 以 **檢索增強生成（RAG）** 流程回答問題，並輸出 **官方格式的 JSON**  
- 內建 **離線評測**（以 `ground_truths_example.json` 比對）流程，便於迭代

---

## 系統架構
1. **資料擷取與正規化**
   - **PDF**：`PDFProcessor` 結合 `pdfplumber` / `PyMuPDF` 取得文字；影像頁透過 `PIL + pytesseract` OCR
   - **OCR 後處理**：使用 `LangChain` + `ChatOpenAI(gpt-4o-mini)` 的 **OCR 清理 Prompt**，自動修整錯字與版面
   - **JSON**：`JsonProcessor` 解析 FAQ 類資料
   - 所有文本透過 `CorpusManager` 彙整與持久化（`data/formatted_data/*_all_text.json`）

2. **文件摘要（可選）**
   - `DocumentManager` 以 LLM 對長文做 **重點摘要**，降低噪音與 Token 成本
   - FAQ 類（短文）直接入庫，不做摘要

3. **向量索引**
   - `FAISSIndexManager` 使用 **moka-ai/m3e-large**（`langchain_huggingface`）建 **FAISS** 嵌入索引
   - 依類別建立目錄（如 `models/finance_faiss_index/`），含 `index.faiss` 與 `index.pkl`

4. **檢索與生成**
   - `QuestionController`：依問題的 `category`（`faq/finance/insurance`）檢索 Top-K 文段 + 來源清單
   - `LangChainModel`：以 **回答 Prompt** 呼叫 `ChatOpenAI(gpt-4o-mini)` 生成最終答案（中文）

5. **CLI / 產出格式**
   - `app.py` 以 `argparse` 收參數，**輸入題目路徑**、**參考資料根目錄**、**輸出 JSON 路徑**  
   - 若對應類別的 FAISS 索引不存在，會自動先執行 **初始化流程**（擷取→摘要→建索引）  
   - 產生符合官方規範的 `model_output.json`

---

## 主要模組
- `controllers/InitializationController.py`：一鍵初始化（資料處理→摘要→索引）
- `controllers/QuestionController.py`：問題處理、檢索與答案組裝
- `models/PDFProcessor.py`：PDF 解析、影像 OCR、OCR 後處理（LLM 清理）
- `models/JsonProcessor.py`：FAQ JSON 解析與彙整
- `models/DocumentManager.py`：摘要與文件分段
- `models/FAISSIndexManager.py`：m3e-large 嵌入、FAISS 建索引/查詢
- `models/LangChainModel.py`：`ChatOpenAI(gpt-4o-mini)`、Prompt 管線
- `config/Config.py`：`.env` 管理（`OPENAI_API_KEY`）

---

## 執行方式（範例）
```bash
# 產出符合競賽格式的答案 JSON
python app.py \
  --question_path "data/dataset/preliminary/questions_example.json" \
  --source_path   "data/reference" \
  --output_path   "data/model_output/model_output.json"
