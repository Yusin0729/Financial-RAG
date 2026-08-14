# ==============================================================================
# 【模組說明】：reranker_test.py (MixRAG 重排序 Reranker 獨立模組 - 選單 [6] 專用)
# 【主要功能】：對初階檢索出之候選 Chunks (Top 20) 進行二次重排序 (Reranking)。
#              結合關鍵字匹配 (BM25 Frequency)、財務科目權重升頻 (Financial Statement Boost)
#              與 H-RCL 表格階層對齊分數，精確升頻核心財報數據至 Ranking #1，並自動輸出 Markdown。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 檔案路徑與目錄搜尋、建立 response 輸出目錄。
# 2. sys: 控制系統路徑與主控台 UTF-8 編碼相容性。
# 3. json: 讀取向量與中繼資料。
# 4. datetime: 產生重排序 Markdown 輸出檔名之時間戳記。
# 5. re: 解析關鍵字與章節對齊度。
# 6. retrieval_test: 呼叫初階向量檢索搜尋 Top 20 候選 Chunk。
# ==============================================================================

import os
import sys
import json
import re
import glob
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from retrieval_test import search_relevant_chunks

RESPONSE_DIR = os.path.join(CURRENT_DIR, "response")

# 核心財務報表關鍵字對照表
FINANCIAL_KEYWORD_MAP = {
    "綜合損益表": ["六、 合併綜合損益表", "損益", "淨利", "收入", "毛利", "費用", "每股盈餘"],
    "資產負債表": ["五、 合併資產負債表", "資產", "負債", "現金", "應收帳款", "存貨", "權益"],
    "現金流量表": ["八、 合併現金流量表", "現金流量", "營運活動", "投資活動", "籌資活動"],
    "權益變動表": ["七、 合併權益變動表", "股本", "資本公積", "保留盈餘"]
}


def compute_bm25_keyword_score(query: str, content: str) -> float:
    """計算簡單關鍵字頻率得分 (BM25 替代加權)"""
    tokens = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+', query)
    if not tokens:
        return 0.0

    match_count = 0
    for tok in tokens:
        if len(tok) >= 2 and tok in content:
            match_count += content.count(tok)

    return min(match_count * 0.02, 0.20)  # 最高給予 0.20 的增益分數


def rerank_candidate_chunks(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    【MixRAG Reranker 重排序演算法】
    對初階檢索結果計算綜合 Reranking 得分：
    Rerank Score = Vector Cosine Score + Financial Section Boost + Keyword Match Boost + H-RCL Table Boost
    """
    reranked_results = []

    for item in candidates:
        base_score = item["score"]
        chk = item["chunk"]
        content = chk.get("content", "")
        main_sec = chk.get("evidence_main_section", "")

        # 1. 財務報表章節與關鍵字對齊加成 (Section Alignment Boost)
        section_boost = 0.0
        for stmt_name, kw_list in FINANCIAL_KEYWORD_MAP.items():
            if any(kw in query for kw in kw_list):
                if any(kw in main_sec for kw in kw_list) or stmt_name in main_sec:
                    section_boost += 0.12  # 命中關鍵章節給予 +0.12 顯著升頻加成
                    break

        # 2. H-RCL 表格階層描述加成 (H-RCL Boost)
        hrcl_boost = 0.05 if "[H-RCL 階層列欄語意對照路徑" in content else 0.0

        # 3. 關鍵字匹配加成 (Keyword Boost)
        keyword_boost = compute_bm25_keyword_score(query, content)

        final_score = base_score + section_boost + hrcl_boost + keyword_boost

        reranked_item = dict(item)
        reranked_item["original_score"] = base_score
        reranked_item["score"] = round(final_score, 4)
        reranked_item["rerank_details"] = {
            "base_vector_score": round(base_score, 4),
            "section_boost": round(section_boost, 4),
            "hrcl_boost": round(hrcl_boost, 4),
            "keyword_boost": round(keyword_boost, 4)
        }
        reranked_results.append(reranked_item)

    # 依加權重排序得分由高到低排序
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    return reranked_results[:top_k]


def save_rerank_response_markdown(query: str, results: list[dict], output_dir: str = None) -> str:
    """將重排序 (Rerank) 結果寫入 Markdown 檔案並儲存於 response 資料夾"""
    if output_dir is None:
        output_dir = RESPONSE_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"rerank_top{len(results)}_{timestamp}.md"
    out_filepath = os.path.join(output_dir, out_filename)

    md_lines = []
    md_lines.append(f"# 財報問題重排序結果 (MixRAG Reranker) - Top {len(results)}")
    md_lines.append(f"**查詢提問**：{query}")
    md_lines.append(f"**重排序時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n" + "=" * 80 + "\n")

    for rank, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        orig_score = res.get("original_score", 0.0)
        details = res.get("rerank_details", {})
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

        md_lines.append(f"### 【重排序排名 #{rank} | 綜合評分: {score:.4f} (原向量分: {orig_score:.4f})】")
        md_lines.append(f"**得分加成明細**：[章節對齊: +{details.get('section_boost', 0):.2f} | H-RCL對齊: +{details.get('hrcl_boost', 0):.2f} | 關鍵字匹配: +{details.get('keyword_boost', 0):.2f}]\n")
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

    print(f"\n[reranker_test.py] [Success] 重排序結果已成功寫入 Markdown 檔：")
    print(f"  -> '{out_filepath}'")

    return out_filepath


def interactive_rerank_search():
    """選單 [6] 專用互動式重排序介面"""
    print("\n" + "=" * 74)
    print("            財報蒟蒻 RAG 系統 - MixRAG 重排序 (reranker_test)")
    print("=" * 74)

    query = input("請輸入您想檢索並重排序的財報問題: ").strip()
    if not query:
        print("[系統提示] 未輸入任何內容，取消檢索與重排序。")
        return

    top_k_str = input("請輸入欲返回的最佳 Chunk 數量 (預設 5): ").strip()
    top_k = int(top_k_str) if top_k_str.isdigit() and int(top_k_str) > 0 else 5

    # 1. 呼叫初階向量檢索取得 Top 20 候選 Chunk
    print(f"\n[步驟 1/2] 執行初階向量檢索 (搜取 Top 20 候選結果)...")
    candidates = search_relevant_chunks(query=query, top_k=20)

    if not candidates:
        print("\n[系統提示] 初階檢索未找到任何相關結果。")
        return

    # 2. 執行 MixRAG 重排序
    print(f"\n[步驟 2/2] 執行 MixRAG (H-RCL + 章節對齊) 重排序二次過濾...")
    reranked_results = rerank_candidate_chunks(query=query, candidates=candidates, top_k=top_k)

    print("\n" + "=" * 74)
    print(f"   [重排序結果] 針對問題: 「{query}」重排序精選 Top {len(reranked_results)} 個最相關結果：")
    print("=" * 74)

    for rank, res in enumerate(reranked_results, 1):
        score = res["score"]
        orig_score = res["original_score"]
        details = res["rerank_details"]
        chk = res["chunk"]
        pages_str = ", ".join(chk.get("pages", [])) if isinstance(chk.get("pages"), list) else str(chk.get("pages", ""))
        main_sec = chk.get("evidence_main_section", "")
        doc_id = chk.get("doc_id", res.get("doc_id", ""))
        doc_id_str = f"{doc_id}.pdf" if doc_id and not doc_id.endswith(".pdf") else doc_id

        print(f"\n【重排序排名 #{rank} | 綜合評分: {score:.4f} (原向量分: {orig_score:.4f})】")
        print(f"  得分加成:               [章節對齊: +{details['section_boost']:.2f} | H-RCL對齊: +{details['hrcl_boost']:.2f} | 關鍵字: +{details['keyword_boost']:.2f}]")
        print(f"  Page:                   {pages_str}")
        print(f"  Doc_Id:                 {doc_id_str}")
        print(f"  Evidence_Main_Section:  {main_sec}")
        print(f"  --- [ Chunk 內容預覽 ] ---")
        print(chk.get("content", "").strip())
        print("-" * 74)

    save_rerank_response_markdown(query=query, results=reranked_results)


if __name__ == "__main__":
    interactive_rerank_search()
