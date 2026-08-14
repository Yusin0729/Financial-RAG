# ==============================================================================
# 【模組說明】：embedding_test.py (本地端開源 BAAI/bge-base-zh-v1.5 向量嵌入與資料庫建置測試模組)
# 【主要功能】：讀取 'Chunks/' 資料夾中由 chunk_test 切分出的 Chunk JSON 資料，
#              使用本地端開源模型 BAAI/bge-base-zh-v1.5 (768 維向量) 將文字轉化為向量，
#              100% 離線執行、零 API Key 依賴、零 429 頻率限制，
#              將向量 (.npy) 與中繼資料 (.json) 儲存於 'Vector_Database/' 建置本地知識庫。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 檔案路徑與目錄搜尋。
# 2. sys: 控制系統路徑與主控台 UTF-8 編碼相容性。
# 3. json: 讀寫 Chunk 資料與向量資料庫中繼資料。
# 4. glob: 搜尋 Chunks 目錄下之所有 JSON 檔案。
# 5. numpy: 處理向量陣列計算與 .npy 檔案儲存。
# 6. shutil (內建模組): 用於遞迴清理 Vector_Database 資料夾內的舊檔案。
# 7. sentence_transformers: 本地端加載並執行開源 BAAI/bge-base-zh-v1.5 向量模型。
# ==============================================================================

import os
import sys
import json
import glob
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import numpy as np

# 嘗試引用 sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[embedding_test.py] [Error] 未找到 'sentence-transformers' 套件，請先在終端機執行 'pip install sentence-transformers'！")

# 目錄與本地模型設定
CHUNKS_DIR = os.path.join(CURRENT_DIR, "Chunks")
VECTOR_DB_DIR = os.path.join(CURRENT_DIR, "Vector_Database")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
_MODEL_CACHE = None


def clean_directory(dir_path: str):
    """清空指定目錄內的所有檔案與子目錄（保留 .gitkeep）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return
    for item in os.listdir(dir_path):
        if item == ".gitkeep":
            continue
        item_path = os.path.join(dir_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"[embedding_test.py] [Warning] 無法刪除舊檔案 '{item_path}': {e}")


def get_bge_model():
    """單例加載 BAAI/bge-base-zh-v1.5 本地向量模型"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "\n[系統提示] 本地 Python 環境尚未安裝 'sentence-transformers' 套件！\n"
                "請在終端機 (Terminal) 中執行以下命令完成安裝：\n\n"
                "    pip install sentence-transformers\n"
            )
        print(f"[embedding_test.py] 正在載入本地開源向量模型 '{MODEL_NAME}' (免 API Key / 零 429 限流)...")
        _MODEL_CACHE = SentenceTransformer(MODEL_NAME)
    return _MODEL_CACHE


def build_embeddings_for_chunk_file(chunk_json_path: str, output_dir: str = None) -> bool:
    """
    讀取單個 Chunk JSON 檔案，呼叫本地 BAAI/bge-base-zh-v1.5 模型產生 768 維向量，
    將向量 (numpy array) 與中繼資料 (JSON) 儲存至 Vector_Database/
    """
    if not os.path.exists(chunk_json_path):
        print(f"[embedding_test.py] [Error] 找不到 Chunk JSON 檔: {chunk_json_path}")
        return False

    if output_dir is None:
        output_dir = VECTOR_DB_DIR
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(chunk_json_path)
    doc_name = filename.replace("_chunks.json", "").replace(".json", "")

    print(f"[embedding_test.py] 正在為 '{filename}' 建立本地 BAAI/bge-base-zh-v1.5 向量資料庫...")

    with open(chunk_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        print(f"[embedding_test.py] '{filename}' 中沒有任何 Chunks。")
        return False

    # 取出待向量化的內容文字 (包含 Context Prefix 與 H-RCL 階層描述)
    contents = [chk["content"] for chk in chunks]

    # 1. 載入本地 bge 模型並批次生成向量 (預設正規化以利 Cosine 點積計算)
    model = get_bge_model()
    print(f"  -> 開始進行 100% 本地端向量計算 ({len(contents)} 個 Chunks)...")
    vectors_np = model.encode(contents, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    vectors_np = np.array(vectors_np, dtype=np.float32)

    # 2. 儲存向量檔 (.npy) 與 結構化中繼資料檔 (.json)
    npy_path = os.path.join(output_dir, f"{doc_name}_vectors.npy")
    meta_path = os.path.join(output_dir, f"{doc_name}_metadata.json")

    np.save(npy_path, vectors_np)

    metadata_store = {
        "source_file": data.get("source_file", filename),
        "doc_id": data.get("doc_id", doc_name),
        "doc_year": data.get("doc_year", ""),
        "total_chunks": len(chunks),
        "embedding_model": MODEL_NAME,
        "vector_dimension": vectors_np.shape[1] if vectors_np.ndim > 1 else 0,
        "chunks": chunks
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)

    print(f"[embedding_test.py] [Success] 成功完成本地 768 維向量資料庫建置！")
    print(f"  -> 向量檔: '{npy_path}' (Shape: {vectors_np.shape})")
    print(f"  -> 中繼資料檔: '{meta_path}'")

    return True


def build_all_embeddings(chunks_dir: str = None, output_dir: str = None) -> bool:
    """自動遍歷 Chunks/ 目錄下所有 JSON 檔並建置向量庫"""
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR
    if output_dir is None:
        output_dir = VECTOR_DB_DIR

    json_files = glob.glob(os.path.join(chunks_dir, "*_chunks.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(chunks_dir, "*.json"))

    if not json_files:
        print(f"[embedding_test.py] 在 '{chunks_dir}' 目錄下找不到任何 Chunks JSON 檔案。")
        return False

    print(f"[embedding_test.py] 清理舊的 Vector_Database 資料夾內容...")
    clean_directory(output_dir)

    print(f"[embedding_test.py] 找到 {len(json_files)} 個 Chunk JSON 檔案，準備建置向量知識庫...")
    for jpath in json_files:
        build_embeddings_for_chunk_file(jpath, output_dir=output_dir)

    return True


if __name__ == "__main__":
    build_all_embeddings()
