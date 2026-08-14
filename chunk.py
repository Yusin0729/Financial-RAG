# ==============================================================================
# 【模組說明】：chunk.py (文本 Chunk 切分與狀態機模組)
# 【主要功能】：讀取標籤化後的 Markdown 財報資料，利用 Event Indexing 與狀態機 (State Machine)
#              執行高效能、高精確度之 Chunk 切分。採用「Context Prefix 注入」與「結構化 Metadata」
#              雙軌機制，包含持久性頁碼追蹤 (Running Page Tracking)、第一行標題精確提取 (Evidence_Note_Title)，
#              保護表格 (TABLE) 與四大報表 (FINANCIAL_STATEMENT) 完整性，將結果輸出為 JSON 與 Markdown 檔儲存於 'Chunks/' 資料夾。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 處理檔案路徑、目錄搜尋與建立 Chunk 輸出目錄。
# 2. sys: 控制系統路徑與主控台 UTF-8 編碼相容性。
# 3. json: 讀寫 Chunk 結構化 JSON 檔案。
# 4. re: 使用正則表達式高效解析 Markdown 標籤與章節標題。
# 5. glob: 搜尋 Markdown 目錄下之所有 .md 檔案。
# 6. shutil (內建模組): 用於遞迴清理 Chunks 資料夾內的舊檔案。
# ==============================================================================

import os
import sys
import json
import re
import glob
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Chunk 輸出目錄
CHUNKS_DIR = os.path.join(CURRENT_DIR, "Chunks")
MARKDOWN_DIR = os.path.join(CURRENT_DIR, "Markdown")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# 四大財務報表關鍵字清單
FINANCIAL_STATEMENT_NAMES = ["綜合損益表", "資產負債表", "現金流量表", "權益變動表"]


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
            print(f"[chunk.py] [Warning] 無法刪除舊檔案 '{item_path}': {e}")


def is_financial_statement(main_sec: str) -> bool:
    """判斷給定的主章節名稱是否屬於四大財務報表"""
    if not main_sec:
        return False
    return any(name in main_sec for name in FINANCIAL_STATEMENT_NAMES)


def build_context_prefix(doc_id: str, years: str, pages: list[str], main_sec: str, sub_sec: str, note_title: str = "") -> str:
    """建立顯式 Context Header (文字前綴)，包含 Evidence_Note_Title 供向量 Embedding 與 LLM 背景閱讀"""
    pages_str = ", ".join(pages) if pages else "N/A"
    sec_str = main_sec if main_sec else "未分類章節"
    if sub_sec:
        sec_str += f" > {sub_sec}"
    note_title_str = note_title if note_title else "無"

    header = (
        f"[財報上下文 | 文件: {doc_id} | 年份: {years} | 頁碼: {pages_str} | 章節: {sec_str} | 附註小標: {note_title_str}]\n"
        f"{'=' * 80}"
    )
    return header


def extract_first_line_note_title(lines: list[str]) -> str:
    """
    檢查 Chunk 的第一個非標籤實質行 (First Line)。
    若第一個實質行為 <u>...</u> 或 ## 標題，則提取該標題文字作為 Evidence_Note_Title；
    若第一個實質行非標題，則直接傳回空字串 ""。
    """
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    underline_pattern = re.compile(r'<u>(.*?)</u>')
    metadata_tags = (
        "<!-- Page:",
        "<!-- Doc_Id:",
        "<!-- Years:",
        "<!-- Relevant_items:",
        "<!-- Evidence_Main_Section:",
        "<!-- Evidence_Sub_Section:"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(tag) for tag in metadata_tags):
            continue

        # 檢查第一行是否為標題
        m_under = underline_pattern.search(stripped)
        if m_under:
            return m_under.group(1).strip()

        m_head = heading_pattern.match(stripped)
        if m_head:
            return m_head.group(2).strip()

        # 第一行非標題，直接結束搜尋並傳回空字串
        break

    return ""


def clean_chunk_body(lines: list[str]) -> str:
    """清洗重複性高的元資料標籤註解，保留乾淨的內文與表格內容"""
    cleaned_lines = []
    metadata_tags = (
        "<!-- Page:",
        "<!-- Doc_Id:",
        "<!-- Years:",
        "<!-- Relevant_items:",
        "<!-- Evidence_Main_Section:",
        "<!-- Evidence_Sub_Section:"
    )

    for line in lines:
        stripped = line.strip()
        # 移除已注入 Context Header 的頂層元素標籤，保留 Table 特殊標籤
        if any(stripped.startswith(tag) for tag in metadata_tags):
            continue
        cleaned_lines.append(line)

    return "".join(cleaned_lines).strip()


def calculate_content_length(lines: list[str]) -> int:
    """計算 Chunk 中實質文字字數 (不含標籤註解)"""
    char_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        char_count += len(stripped)
    return char_count


def scan_event_index(lines: list[str]) -> dict[int, dict]:
    """
    【Pass 1: Tag Scanner / Event Indexer】
    對 Markdown 所有行進行快速標籤搜尋，不進行語意分析，建立 Line Index -> Event 對照表。
    """
    event_index = {}

    page_pattern = re.compile(r'<!--\s*Page:\s*(.*?)\s*-->', re.IGNORECASE)
    doc_id_pattern = re.compile(r'<!--\s*Doc_Id:\s*(.*?)\s*-->', re.IGNORECASE)
    years_pattern = re.compile(r'<!--\s*Years:\s*(.*?)\s*-->', re.IGNORECASE)
    relevant_items_pattern = re.compile(r'<!--\s*Relevant_items:\s*(.*?)\s*-->', re.IGNORECASE)
    main_sec_pattern = re.compile(r'<!--\s*Evidence_Main_Section:\s*(.*?)\s*-->', re.IGNORECASE)
    sub_sec_pattern = re.compile(r'<!--\s*Evidence_Sub_Section:\s*(.*?)\s*-->', re.IGNORECASE)
    table_start_pattern = re.compile(r'<!--\s*TABLE_START\s*-->', re.IGNORECASE)
    table_end_pattern = re.compile(r'<!--\s*TABLE_END\s*-->', re.IGNORECASE)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    underline_pattern = re.compile(r'<u>(.*?)</u>')

    for idx, line in enumerate(lines):
        line_str = line.strip()

        m_table_start = table_start_pattern.search(line_str)
        if m_table_start:
            event_index[idx] = {"type": "TABLE_START"}
            continue

        m_table_end = table_end_pattern.search(line_str)
        if m_table_end:
            event_index[idx] = {"type": "TABLE_END"}
            continue

        m_page = page_pattern.search(line_str)
        if m_page:
            event_index[idx] = {"type": "PAGE", "val": m_page.group(1).strip()}
            continue

        m_doc = doc_id_pattern.search(line_str)
        if m_doc:
            event_index[idx] = {"type": "DOC_ID", "val": m_doc.group(1).strip()}
            continue

        m_years = years_pattern.search(line_str)
        if m_years:
            event_index[idx] = {"type": "YEARS", "val": m_years.group(1).strip()}
            continue

        m_rel = relevant_items_pattern.search(line_str)
        if m_rel:
            event_index[idx] = {"type": "RELEVANT_ITEMS", "val": m_rel.group(1).strip()}
            continue

        m_main = main_sec_pattern.search(line_str)
        if m_main:
            event_index[idx] = {"type": "EVIDENCE_MAIN_SECTION", "val": m_main.group(1).strip()}
            continue

        m_sub = sub_sec_pattern.search(line_str)
        if m_sub:
            event_index[idx] = {"type": "EVIDENCE_SUB_SECTION", "val": m_sub.group(1).strip()}
            continue

        m_head = heading_pattern.match(line_str)
        if m_head:
            event_index[idx] = {"type": "SECTION_HEADING", "val": m_head.group(2).strip()}
            continue

        m_under = underline_pattern.search(line_str)
        if m_under:
            event_index[idx] = {"type": "UNDERLINE_HEADING", "val": m_under.group(1).strip()}
            continue

    return event_index


def process_chunking(input_md_path: str, output_dir: str = None) -> bool:
    """
    【Pass 2: State Machine Chunk Generator】
    根據 Event Index 與狀態機 (NORMAL, TABLE, FINANCIAL_STATEMENT) 切分 Chunk，
    使用 Running Page 追蹤避免頁碼遺失，精確提取第一行標題 (Evidence_Note_Title)，
    並注入 Context Prefix 與綁定結構化 Metadata。
    """
    if not os.path.exists(input_md_path):
        print(f"[chunk.py] [Error] 找不到輸入 Markdown 檔: {input_md_path}")
        return False

    if output_dir is None:
        output_dir = CHUNKS_DIR
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_md_path)
    name_only, _ = os.path.splitext(filename)

    print(f"[chunk.py] 開始進行 Markdown Chunking 處理 (Running Page + Context Prefix + State Machine): '{filename}'...")

    with open(input_md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pass 1: 建立 Event Index
    event_index = scan_event_index(lines)

    # State Machine 變數初始化
    state = "NORMAL"  # "NORMAL", "TABLE", "FINANCIAL_STATEMENT"

    chunks = []
    current_lines = []
    current_pages = []
    running_page = ""
    current_main_sec = ""
    current_sub_sec = ""
    current_doc_id = name_only
    current_year = ""
    current_relevant_items = ""

    def flush_chunk(chunk_type: str = None):
        nonlocal current_lines, current_pages, running_page, current_main_sec, current_sub_sec, current_doc_id, current_year, current_relevant_items
        if not current_lines:
            return

        body_text = clean_chunk_body(current_lines)
        if not body_text:
            current_lines = []
            return

        effective_type = chunk_type if chunk_type else state
        char_count = calculate_content_length(current_lines)
        pages_list = list(current_pages) if current_pages else ([running_page] if running_page else [])

        # 精確提取該 Chunk 的「第一行標題」作為 Evidence_Note_Title
        note_title = extract_first_line_note_title(current_lines)

        # 組合 Context Header 注入前綴
        prefix_header = build_context_prefix(
            doc_id=current_doc_id,
            years=current_year,
            pages=pages_list,
            main_sec=current_main_sec,
            sub_sec=current_sub_sec,
            note_title=note_title
        )

        full_content = f"{prefix_header}\n\n{body_text}"

        chunk_id = len(chunks) + 1
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_type": effective_type,
            "doc_id": current_doc_id,
            "years": current_year,
            "relevant_items": current_relevant_items,
            "pages": pages_list,
            "evidence_main_section": current_main_sec,
            "evidence_sub_section": current_sub_sec,
            "evidence_note_title": note_title,
            "char_count": char_count,
            "content": full_content
        })

        current_lines = []
        current_pages = [running_page] if running_page else []

    for idx, line in enumerate(lines):
        evt = event_index.get(idx)

        # 【優先級 Priority 0】：無論處於何種狀態 (NORMAL / TABLE / FINANCIAL_STATEMENT)，遇 PAGE 標籤皆優先更新頁碼
        if evt and evt["type"] == "PAGE":
            page_val = evt["val"]
            running_page = page_val
            if page_val not in current_pages:
                # 在 NORMAL 模式下，若跨頁超過 2 頁，切斷目前 CHUNK
                if state == "NORMAL" and len(current_pages) >= 2:
                    flush_chunk(chunk_type="NORMAL")
                if page_val not in current_pages:
                    current_pages.append(page_val)

        # 狀態 1: TABLE 保護模式 (最高優先級 Priority 1)
        if state == "TABLE":
            current_lines.append(line)
            if evt and evt["type"] == "TABLE_END":
                flush_chunk(chunk_type="TABLE")
                state = "NORMAL"
            continue

        # 狀態 2: FINANCIAL_STATEMENT 保護模式 (Priority 2)
        if state == "FINANCIAL_STATEMENT":
            if evt and evt["type"] == "EVIDENCE_MAIN_SECTION":
                new_main = evt["val"]
                if new_main != current_main_sec:
                    # 當前報表結束，切斷並重置狀態
                    flush_chunk(chunk_type="FINANCIAL_STATEMENT")
                    current_main_sec = new_main
                    if is_financial_statement(new_main):
                        state = "FINANCIAL_STATEMENT"
                    else:
                        state = "NORMAL"
            current_lines.append(line)
            continue

        # 狀態 3: NORMAL 模式
        if evt:
            evt_type = evt["type"]

            # 1. 遇到 TABLE_START -> 轉為 TABLE 狀態 (Priority 1)
            if evt_type == "TABLE_START":
                flush_chunk(chunk_type="NORMAL")
                state = "TABLE"
                current_lines.append(line)
                continue

            # 2. 遇到 DOC_ID
            if evt_type == "DOC_ID":
                current_doc_id = evt["val"]

            # 4. 遇到 YEARS
            elif evt_type == "YEARS":
                current_year = evt["val"]

            # 遇到 RELEVANT_ITEMS
            elif evt_type == "RELEVANT_ITEMS":
                current_relevant_items = evt["val"]

            # 5. 遇到 EVIDENCE_MAIN_SECTION (Priority 3 & Priority 2)
            elif evt_type == "EVIDENCE_MAIN_SECTION":
                new_main = evt["val"]
                if new_main != current_main_sec:
                    if current_lines:
                        flush_chunk(chunk_type="NORMAL")
                    current_main_sec = new_main
                    if is_financial_statement(new_main):
                        state = "FINANCIAL_STATEMENT"

            # 6. 遇到 EVIDENCE_SUB_SECTION (Priority 4)
            elif evt_type == "EVIDENCE_SUB_SECTION":
                new_sub = evt["val"]
                if new_sub != current_sub_sec and new_sub:
                    if current_lines:
                        flush_chunk(chunk_type="NORMAL")
                    current_sub_sec = new_sub

            # 7. 遇到章節標題 SECTION_HEADING 或 UNDERLINE_HEADING (Priority 5 & Priority 6)
            elif evt_type in ["SECTION_HEADING", "UNDERLINE_HEADING"]:
                # 只有當目前 CHUNK 字數 >= 100 時，才允許依據章節標題進行切分
                if calculate_content_length(current_lines) >= 100:
                    flush_chunk(chunk_type="NORMAL")

        # 將該行加入目前 CHUNK
        current_lines.append(line)

    # 處理最後未 flush 的 Chunk
    if current_lines:
        flush_chunk()

    # 將切分結果寫入 JSON 檔與 Markdown 檔
    json_out_path = os.path.join(output_dir, f"{name_only}_chunks.json")
    md_out_path = os.path.join(output_dir, f"{name_only}_chunks.md")

    json_result = {
        "source_file": filename,
        "doc_id": current_doc_id,
        "doc_year": current_year,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    md_chunk_lines = []
    for chk in chunks:
        pages_str = ", ".join(chk["pages"])
        md_chunk_lines.append(f"<!-- CHUNK_START: {chk['chunk_id']} | Type: {chk['chunk_type']} | Pages: {pages_str} -->")
        md_chunk_lines.append(chk["content"])
        md_chunk_lines.append(f"<!-- CHUNK_END: {chk['chunk_id']} -->")
        md_chunk_lines.append("")

    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_chunk_lines))

    print(f"[chunk.py] [Success] 成功切分 {len(chunks)} 個帶 Context Prefix 與 Running Page 之 Chunks！")
    print(f"  -> JSON 檔: '{json_out_path}'")
    print(f"  -> MD 檔:   '{md_out_path}'")

    return True


def chunk_all_markdown_files(markdown_dir: str = None, output_dir: str = None) -> bool:
    """自動處理 Markdown 目錄下所有的 .md 檔案"""
    if markdown_dir is None:
        markdown_dir = MARKDOWN_DIR
    if output_dir is None:
        output_dir = CHUNKS_DIR

    md_files = glob.glob(os.path.join(markdown_dir, "*.md"))
    if not md_files:
        print(f"[chunk.py] 在 '{markdown_dir}' 目錄下未找到任何 .md 檔案。")
        return False

    # 清空 Chunks 輸出資料夾，確保只保留本次切分的檔案
    print(f"[chunk.py] 清理舊的 Chunks 資料夾內容...")
    clean_directory(output_dir)

    print(f"[chunk.py] 找到 {len(md_files)} 個 Markdown 檔案進行 Chunking...")
    for md_path in md_files:
        process_chunking(md_path, output_dir=output_dir)
    return True


if __name__ == "__main__":
    chunk_all_markdown_files()
