# ==============================================================================
# 【模組說明】：retrieval_test.py (本地開源 BAAI/bge-base-zh-v1.5 向量檢索與相似度搜尋測試模組)
# 【主要功能】：接收使用者輸入的問答檢索問題，使用本地端 BAAI/bge-base-zh-v1.5 模型將問題轉為 768 維查詢向量，
#              並與 'Vector_Database/' 資料夾中由 embedding_test 建置的 768 維向量進行餘弦相似度計算，
#              回傳最相關的前 Top-K 個 Chunk 檢索結果，並將結果以標準 Markdown 格式儲存於 'response/'。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 檔案路徑與目錄搜尋、建立 response 輸出目錄與讀取環境變數。
# 2. sys: 控制系統路徑與主控台 UTF-8 編碼相容性。
# 3. json: 讀取向量資料庫中繼資料與 Chunk 文字內容。
# 4. glob: 搜尋 Vector_Database 目錄下之所有 .npy 與 .json 檔案。
# 5. datetime: 產生檢索 Markdown 輸出檔名之時間戳記。
# 6. numpy: 處理向量餘弦相似度矩陣點積計算。
# 7. sentence_transformers: 本地端加載並執行開源 BAAI/bge-base-zh-v1.5 向量模型。
# ==============================================================================

import os
import sys
import json
import glob
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[retrieval_test.py] [Error] 未找到 'sentence-transformers' 套件，請先在終端機執行 'pip install sentence-transformers'！")

VECTOR_DB_DIR = os.path.join(CURRENT_DIR, "Vector_Database")
RESPONSE_DIR = os.path.join(CURRENT_DIR, "response")
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

_RETRIEVAL_MODEL_CACHE = None


def get_bge_retrieval_model():
    """單例加載 BAAI/bge-base-zh-v1.5 本地向量模型"""
    global _RETRIEVAL_MODEL_CACHE
    if _RETRIEVAL_MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "\n[系統提示] 本地 Python 環境尚未安裝 'sentence-transformers' 套件！\n"
                "請在終端機 (Terminal) 中執行以下命令完成安裝：\n\n"
                "    pip install sentence-transformers\n"
            )
        print(f"[retrieval_test.py] 正在載入本地開源向量模型 '{MODEL_NAME}' (免 API Key / 零 429 限流)...")
        _RETRIEVAL_MODEL_CACHE = SentenceTransformer(MODEL_NAME)
    return _RETRIEVAL_MODEL_CACHE


def compute_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """計算查詢向量與文件向量矩陣之間的餘弦相似度 (Cosine Similarity)"""
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-10
    q_unit = query_vec / query_norm

    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1e-10
    d_unit = doc_vecs / doc_norms

    similarities = np.dot(d_unit, q_unit)
    return similarities


def search_relevant_chunks(query: str, top_k: int = 10, filter_year: str = None) -> list[dict]:
    """【主要檢索功能】：使用本地 bge 模型將查詢轉為向量，在 Vector_Database/ 中進行比對"""
    if not query.strip():
        print("[retrieval_test.py] 查詢字串不可為空。")
        return []

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"[retrieval_test.py] [Error] 找不到向量資料庫目錄 '{VECTOR_DB_DIR}'，請先執行選單 [3] 建立向量資料庫！")
        return []

    npy_files = glob.glob(os.path.join(VECTOR_DB_DIR, "*_vectors.npy"))
    if not npy_files:
        print(f"[retrieval_test.py] [Error] 在 '{VECTOR_DB_DIR}' 目錄下未找到任何向量檔，請先執行選單 [3] 建立向量資料庫！")
        return []

    print(f"[retrieval_test.py] 正在使用本地端 BAAI/bge-base-zh-v1.5 模型為「{query}」生成向量並檢索...")

    # 1. 使用本地 bge 模型產生 768 維查詢向量
    model = get_bge_retrieval_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    query_vec = np.array(query_vec, dtype=np.float32)

    all_matches = []

    for npy_path in npy_files:
        meta_path = npy_path.replace("_vectors.npy", "_metadata.json")
        if not os.path.exists(meta_path):
            continue

        doc_vecs = np.load(npy_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        chunks = meta_data.get("chunks", [])
        doc_id = meta_data.get("doc_id", "")
        doc_year = meta_data.get("doc_year", "")

        if filter_year and str(doc_year) != str(filter_year):
            continue

        if len(doc_vecs) == 0 or len(chunks) == 0:
            continue

        scores = compute_cosine_similarity(query_vec, doc_vecs)

        for idx, score in enumerate(scores):
            if idx < len(chunks):
                chk = chunks[idx]
                all_matches.append({
                    "score": float(score),
                    "doc_id": doc_id,
                    "doc_year": doc_year,
                    "chunk": chk
                })

    all_matches.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_matches[:top_k]

    return top_results


def save_retrieval_response_markdown(query: str, results: list[dict], output_dir: str = None) -> str:
    """匯出 Markdown 結果檔至 response 資料夾"""
    if output_dir is None:
        output_dir = RESPONSE_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"retrieval_top{len(results)}_{timestamp}.md"
    out_filepath = os.path.join(output_dir, out_filename)

    md_lines = []
    md_lines.append(f"# 財報問題檢索結果 (retrieval_test - BAAI/bge-base-zh-v1.5 本地模型) - Top {len(results)}")
    md_lines.append(f"**查詢提問**：{query}")
    md_lines.append(f"**檢索時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n" + "=" * 80 + "\n")

    for rank, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        chk = res.get("chunk", {})

        pages = chk.get("pages", [])
        pages_str = ", ".join(pages) if isinstance(pages, list) else str(pages or "")
        doc_id = chk.get("doc_id", res.get("doc_id", ""))
        doc_id_str = f"{doc_id}.pdf" if doc_id and not doc_id.endswith(".pdf") else doc_id

        years_str = chk.get("years", "") or str(chk.get("doc_year", res.get("doc_year", "")))
        relevant_items_str = chk.get("relevant_items", "")
        main_sec = chk.get("evidence_main_section", "")
        sub_sec = chk.get("evidence_sub_section", "")
        note_title = chk.get("evidence_note_title", "")
        golden_context = chk.get("content", "").strip()

        md_lines.append(f"### 【排名 #{rank} | 相似度得分: {score:.4f}】\n")
        md_lines.append(f"Page\t{pages_str}")
        md_lines.append(f"Golden_Context\t{golden_context}")
        md_lines.append(f"Doc_Id\t{doc_id_str}")
        md_lines.append(f"Years\t{years_str}")
        md_lines.append(f"Relevant_items\t{relevant_items_str}")
        md_lines.append(f"Evidence_Main_Section\t{main_sec}")
        md_lines.append(f"Evidence_Sub_Section\t{sub_sec}")
        md_lines.append(f"Evidence_Note_Title\t{note_title}")
        md_lines.append("\n" + "-" * 80 + "\n")

    with open(out_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[retrieval_test.py] [Success] 檢索結果已成功寫入 Markdown 檔並儲存於：")
    print(f"  -> '{out_filepath}'")

    return out_filepath


def interactive_retrieval_search():
    """選單 [5] 專用互動式查詢介面"""
    print("\n" + "=" * 74)
    print("        財報蒟蒻 RAG 系統 - 問題檢索 (retrieval_test - BAAI/bge-base-zh-v1.5 本地端)")
    print("=" * 74)

    query = input("請輸入您想檢索的財報問題: ").strip()
    if not query:
        print("[系統提示] 未輸入任何內容，取消檢索。")
        return

    top_k_str = input("請輸入欲返回的最佳 Chunk 數量 (預設 10): ").strip()
    top_k = int(top_k_str) if top_k_str.isdigit() and int(top_k_str) > 0 else 10

    results = search_relevant_chunks(query=query, top_k=top_k)

    if not results:
        print("\n[系統提示] 未找到相關的 Chunk 結果。")
        return

    print("\n" + "=" * 74)
    print(f"   [檢索結果] 針對問題: 「{query}」檢索出 Top {len(results)} 個最相關 Chunks：")
    print("=" * 74)

    for rank, res in enumerate(results, 1):
        score = res["score"]
        chk = res["chunk"]
        pages_str = ", ".join(chk.get("pages", [])) if isinstance(chk.get("pages"), list) else str(chk.get("pages", ""))
        main_sec = chk.get("evidence_main_section", "")
        sub_sec = chk.get("evidence_sub_section", "")
        note_title = chk.get("evidence_note_title", "")
        rel_items = chk.get("relevant_items", "")
        doc_id = chk.get("doc_id", res.get("doc_id", ""))
        doc_id_str = f"{doc_id}.pdf" if doc_id and not doc_id.endswith(".pdf") else doc_id

        print(f"\n【排名 #{rank} | 相似度得分: {score:.4f}】")
        print(f"  Page:                   {pages_str}")
        print(f"  Golden_Context:         [已內嵌 Chunk Markdown 內容]")
        print(f"  Doc_Id:                 {doc_id_str}")
        print(f"  Years:                  {chk.get('years', res.get('doc_year', ''))}")
        print(f"  Relevant_items:         {rel_items}")
        print(f"  Evidence_Main_Section:  {main_sec}")
        print(f"  Evidence_Sub_Section:   {sub_sec}")
        print(f"  Evidence_Note_Title:    {note_title}")
        print(f"  --- [ Chunk 內容預覽 ] ---")
        print(chk.get("content", "").strip())
        print("-" * 74)

    save_retrieval_response_markdown(query=query, results=results)


if __name__ == "__main__":
    interactive_retrieval_search()
