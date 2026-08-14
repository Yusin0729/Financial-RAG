# 財報蒟蒻 RAG 系統 (Multi-Financial-Report RAG System)

本專案為一套針對**多年度、多公司財務報告（PDF）**設計的專業檢索增強生成（Retrieval-Augmented Generation, RAG）知識庫建置與檢索系統。

透過自動化 PDF 結構標籤化、狀態機切分（State Machine Chunking）、顯式上下文前綴注入（Context Header Prefix）以及本地開源向量模型，協助使用者快速建置可精確追蹤頁碼與會計科目的財報向量知識庫，並提供高精確度的檢索與重排序功能。

---

## 核心功能與特色

1. **PDF 結構化與 Metadata 自動標籤化**
   - 自動解析財報目錄（TOC），動態對應頁碼至大標（`Evidence_Main_Section`）與子標（`Evidence_Sub_Section`）。
   - 保留頁碼標籤（`Page`）、公司 ID（`Doc_Id`）、年份（`Years`）與表格註記。

2. **精確 Chunk 切分與 Context Header 前綴注入**
   - 採用狀態機（`NORMAL` / `TABLE` / `FINANCIAL_STATEMENT`）保護四大財務報表與表格不被割裂。
   - 每個 Chunk 自動注入顯式 Context Header 前綴（包含文件 ID、年份、頁碼、章節與附註小標），解決向量化過程中的上下文遺失問題。

3. **100% 本地端離線向量庫建置 (Zero API Limits)**
   - 使用輕量且專為中文優化的本地開源向量模型 **`BAAI/bge-base-zh-v1.5`**（768 維度）。
   - 離線執行，免 API Key 依賴，零 429 速率限制問題。
   - 向量檔（`.npy`）與結構化中繼資料（`.json`）分類保存。

4. **高精確度問答檢索與多重加權重排序 (Reranking)**
   - 基於餘弦相似度（Cosine Similarity）進行 Top-K 向量初階檢索。
   - 提供 MixRAG 重排序演算法，結合向量分、BM25 關鍵字匹配、財務報表章節升頻（Section Boost）與 H-RCL 表格對照加成。
   - 檢索結果自動匯出為標準 Markdown 報告，精確標明引文頁碼與 Chunk 內容。

---

## 專案目錄結構

```text
Financial RAG/
├── main.py                     # 主程式進入點 (提供互動式選單 1-7)
├── markdown.py                 # PDF 轉檔與 Metadata 標籤化模組 (正式版)
├── markdown_test.py            # PDF 轉檔與 Metadata 標籤化模組 (MixRAG 測試版)
├── chunk.py                    # 文本 Chunk 切分與狀態機模組 (正式版)
├── chunk_test.py               # 文本與 H-RCL 表格階層 Chunk 切分模組 (MixRAG 測試版)
├── embedding.py                # 向量嵌入與資料庫建置模組 (正式版 - bge-base-zh-v1.5)
├── embedding_test.py           # 向量嵌入與資料庫建置模組 (測試版 - bge-base-zh-v1.5)
├── retrieval.py                # 向量檢索與相似度搜尋模組 (正式版)
├── retrieval_test.py           # 向量檢索與相似度搜尋模組 (測試版)
├── reranker_test.py            # MixRAG 重排序二次過濾模組
├── tools/                      # 輔助工具箱 (表格標籤、OCR 解密等)
│   ├── table_tagger.py         # 表格標籤包裹與 Metadata 提取工具
│   └── markdown_ocr_unlock.py  # PDF 解密與 RapidOCR 備援辨識工具
├── Financial Statements/       # [輸入] 存放待處理 PDF 財報檔案
├── Markdown/                   # [中間產物] 轉換後之標籤化 Markdown 檔案
├── Chunks/                     # [中間產物] 切分後帶 Context Header 之 Chunk JSON/MD 檔案
├── Vector_Database/            # [輸出] 向量庫 (.npy 向量陣列與 .json 中繼資料)
└── response/                   # [輸出] 檢索與重排序結果 Markdown 報告
```

---

## 環境需求與安裝

### 1. Python 環境
建議使用 Python 3.10 或更高版本。

### 2. 必要套件一鍵安裝指令
請於專案根目錄開啟終端機（Terminal / PowerShell）執行以下指令完成安裝：

```bash
pip install sentence-transformers numpy python-dotenv pypdf pdf_inspector rapidocr-onnxruntime
```

### 3. 所需外部套件詳細清單與用途說明

| 套件名稱 (Package Name) | 主要功能與用途 (Purpose & Description) | 需求層級 |
| :--- | :--- | :--- |
| **`sentence-transformers`** | 載入並執行本地開源向量模型（`BAAI/bge-base-zh-v1.5`），為 Chunk 與查詢生成 768 維度向量 | **核心必要** |
| **`numpy`** | 處理向量陣列計算、儲存 `.npy` 檔案以及進行向量餘弦相似度（Cosine Similarity）矩陣點積 | **核心必要** |
| **`python-dotenv`** | 自動讀取並載入 `.env` 檔案中的環境變數設定 | **核心必要** |
| **`pdf_inspector`** | PDF 轉 Markdown 核心解析器（功能類似 marker），負責將 PDF 內文與表格轉換為 Markdown 管道符號 `\|` 格式 | **核心必要** |
| **`pypdf`** | 檢測 PDF 空白密碼鎖與自動進行無密碼解密作業（`tools/markdown_ocr_unlock.py`） | **核心必要** |
| **`rapidocr-onnxruntime`** | 當選單啟用 OCR 模式時，針對圖檔或掃描頁 PDF 進行 RapidOCR 繁體中文文字辨識 | **選用 (OCR 模式)** |

---

## 使用說明 (Quick Start)

### 步驟 1：放置財報檔案
將欲建立知識庫的 PDF 財報檔案放置於 `Financial Statements/` 資料夾中。
*(檔名建議格式：`2024Q4_2303_Financial_Report.pdf`)*

### 步驟 2：啟動主選單
執行 `main.py` 啟動主控台介面：

```bash
python main.py
```

### 步驟 3：執行選單功能
主選單提供以下操作選項：

```text
==========================================================================
                            財報蒟蒻 RAG 系統
      [系統狀態] OCR 模式: 【 關閉 】 | 問題重寫: 【 關閉 】
==========================================================================
  --- [ 建立資料庫 ] ---

  [1] 建立完整資料庫 (自動執行 PDF 轉檔 -> Chunk 切分 -> 向量建庫)
  [2] PDF 轉為 Markdown
  [3] 讀取 Markdown 建立向量資料庫

  --- [ 回答問題 ] ---

  [4] 完整回答財報問題 (開發中)
  [5] 問題檢索 (輸入提問，回傳 Top-10 相關區塊並匯出 Markdown)
  [6] 重排序 (結合關鍵字與財務章節加權進行二次精確排序)
  [7] 答案生成 (開發中)

  --- [ 系統控制 ] ---

  [S] 切換 OCR 狀態 (開啟/關閉掃描頁 OCR 辨識)
  [W] 切換問題重寫狀態
  [Q] 退出系統
==========================================================================
```

---

## 檢索報告輸出範例

當您執行選項 `[5]` 或 `[6]` 進行問答檢索時，系統會自動在 `response/` 資料夾產生 Markdown 報告：

```markdown
# 財報問題檢索結果 - Top 10
**查詢提問**：營業收入為多少？
**檢索時間**：2026-08-14 14:00:00

================================================================================

### 【排名 #1 | 相似度得分: 0.8254】

Page    P15
Golden_Context  [財報上下文 | 文件: 2024Q4_2303_Financial_Report | 年份: 2024 | 頁碼: P15 | 章節: 六、 合併綜合損益表]
...
Doc_Id  2024Q4_2303_Financial_Report.pdf
Years   2024
Evidence_Main_Section   六、 合併綜合損益表
```

---

## 授權與維護
本專案為財報 RAG 專用架構，程式碼遵循模組化、高穩定度與嚴格架構維護原則。
