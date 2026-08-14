# ==============================================================================
# 【模組說明】：chunk_test.py (文本與 H-RCL 表格階層 Chunk 切分測試模組 - MixRAG 版本)
# 【主要功能】：讀取標籤化後的 Markdown 財報資料，利用 Event Indexing 與狀態機 (State Machine)
#              執行高效能 Chunk 切分。針對表格導入 MixRAG (arXiv:2504.09554) 論文之
#              H-RCL (Hierarchy Row-and-Column-Level) 表格階層路徑轉換算法，
#              將二維矩陣表格轉化為精確之「列欄階層文字路徑」，消除向量特徵稀釋與座標遺失問題，
#              並將結果輸出為 JSON 與 Markdown 檔儲存於 'Chunks/' 資料夾。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 處理檔案路徑、目錄搜尋與建立 Chunk 輸出目錄。
# 2. sys: 控制系統路徑與主控台 UTF-8 編碼相容性。
# 3. json: 讀寫 Chunk 結構化 JSON 檔案。
# 4. re: 使用正則表達式高效解析 Markdown 標籤與表格管道符號。
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
            print(f"[chunk_test.py] [Warning] 無法刪除舊檔案 '{item_path}': {e}")


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


def parse_table_to_hrcl(table_lines: list[str], main_sec: str = "") -> str:
    """
    【MixRAG 論文 H-RCL 演算法實作】
    讀取 Markdown 矩陣表格，解析表頭欄位（Columns）與列項目（Row Items），
    轉化為具備完整父子階層路徑之自然語言描述文 (Hierarchy Row-and-Column-Level Representation)。
    """
    cleaned_rows = []
    for line in table_lines:
        s = line.strip()
        if not s or s.startswith("<!--"):
            continue
        if s.startswith("|") and s.endswith("|"):
            # 分解管道符號表格列
            cells = [c.strip() for c in s.split("|")[1:-1]]
            # 排除純分隔線如 |---|---|
            if all(re.match(r'^[-:]+$', c) for c in cells if c):
                continue
            cleaned_rows.append(cells)

    if not cleaned_rows:
        return ""

    # 嘗試抓取表頭列 (Header Rows)
    header_col_names = []
    data_rows = []
    
    for idx, row in enumerate(cleaned_rows):
        # 若包含「代碼」或「項目」或「金額」視為表頭列
        row_str = " ".join(row)
        if "代碼" in row_str or "項目" in row_str or "科目" in row_str or "金額" in row_str:
            header_col_names = row
        else:
            data_rows.append(row)

    if not data_rows:
        return ""

    hrcl_statements = []
    hrcl_statements.append("\n[H-RCL 階層列欄語意對照路徑 (MixRAG 論文格式)]:")

    running_category = main_sec if main_sec else "財務項目"

    for row in data_rows:
        if not any(row):
            continue

        # 第一欄通常為會計科目/代碼/項目名稱
        item_name = row[0] if len(row) > 0 else ""
        
        # 若整列只有第一欄有文字，代表為大類別標題 (例如：流動資產、營業費用)
        non_empty = [c for c in row if c]
        if len(non_empty) == 1:
            running_category = non_empty[0]
            continue

        # 組合每一欄的數據與表頭對應
        col_details = []
        for c_idx in range(1, len(row)):
            val = row[c_idx]
            if not val or val == "-":
                continue
            col_name = header_col_names[c_idx] if c_idx < len(header_col_names) else f"欄位_{c_idx}"
            col_details.append(f"{col_name}: {val}")

        if col_details:
            details_str = " | ".join(col_details)
            hrcl_statements.append(f"- 階層路徑: {running_category} > {item_name} -> {details_str}")

    return "\n".join(hrcl_statements)


def extract_first_line_note_title(lines: list[str]) -> str:
    """檢查 Chunk 的第一個非標籤實質行提取 Evidence_Note_Title"""
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

        m_under = underline_pattern.search(stripped)
        if m_under:
            return m_under.group(1).strip()

        m_head = heading_pattern.match(stripped)
        if m_head:
            return m_head.group(2).strip()

        break

    return ""


def clean_chunk_body(lines: list[str]) -> str:
    """清洗元資料標籤註解，保留乾淨的內文與表格內容"""
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
        if any(stripped.startswith(tag) for tag in metadata_tags):
            continue
        cleaned_lines.append(line)

    return "".join(cleaned_lines).strip()


def calculate_content_length(lines: list[str]) -> int:
    """計算 Chunk 中實質文字字數"""
    char_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        char_count += len(stripped)
    return char_count


def scan_event_index(lines: list[str]) -> dict[int, dict]:
    """【Pass 1: Tag Scanner / Event Indexer】"""
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
    【Pass 2: State Machine Chunk Generator (MixRAG + H-RCL 升級版)】
    """
    if not os.path.exists(input_md_path):
        print(f"[chunk_test.py] [Error] 找不到輸入 Markdown 檔: {input_md_path}")
        return False

    if output_dir is None:
        output_dir = CHUNKS_DIR
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_md_path)
    name_only, _ = os.path.splitext(filename)

    print(f"[chunk_test.py] 開始進行 MixRAG (H-RCL) Markdown Chunking 處理: '{filename}'...")

    with open(input_md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    event_index = scan_event_index(lines)
    state = "NORMAL"

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

        note_title = extract_first_line_note_title(current_lines)

        prefix_header = build_context_prefix(
            doc_id=current_doc_id,
            years=current_year,
            pages=pages_list,
            main_sec=current_main_sec,
            sub_sec=current_sub_sec,
            note_title=note_title
        )

        # 若 Chunk 為表格類型 (TABLE)，自動注入 MixRAG H-RCL 階層欄位表示法
        hrcl_text = ""
        if effective_type == "TABLE":
            hrcl_text = parse_table_to_hrcl(current_lines, main_sec=current_main_sec)

        if hrcl_text:
            full_content = f"{prefix_header}\n\n{body_text}\n{hrcl_text}"
        else:
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
            "char_count": len(full_content),
            "content": full_content
        })

        current_lines = []
        current_pages = [running_page] if running_page else []

    for idx, line in enumerate(lines):
        evt = event_index.get(idx)

        # 【優先級 Priority 0】：遇 PAGE 標籤皆優先更新頁碼與清單
        if evt and evt["type"] == "PAGE":
            page_val = evt["val"]
            running_page = page_val
            if page_val not in current_pages:
                if state == "NORMAL" and len(current_pages) >= 2:
                    flush_chunk(chunk_type="NORMAL")
                if page_val not in current_pages:
                    current_pages.append(page_val)

        # 狀態 1: TABLE 保護模式
        if state == "TABLE":
            current_lines.append(line)
            if evt and evt["type"] == "TABLE_END":
                flush_chunk(chunk_type="TABLE")
                state = "NORMAL"
            continue

        # 狀態 2: FINANCIAL_STATEMENT 保護模式
        if state == "FINANCIAL_STATEMENT":
            if evt and evt["type"] == "EVIDENCE_MAIN_SECTION":
                new_main = evt["val"]
                if new_main != current_main_sec:
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

            if evt_type == "TABLE_START":
                flush_chunk(chunk_type="NORMAL")
                state = "TABLE"
                current_lines.append(line)
                continue

            elif evt_type == "DOC_ID":
                current_doc_id = evt["val"]

            elif evt_type == "YEARS":
                current_year = evt["val"]

            elif evt_type == "RELEVANT_ITEMS":
                current_relevant_items = evt["val"]

            elif evt_type == "EVIDENCE_MAIN_SECTION":
                new_main = evt["val"]
                if new_main != current_main_sec:
                    if current_lines:
                        flush_chunk(chunk_type="NORMAL")
                    current_main_sec = new_main
                    if is_financial_statement(new_main):
                        state = "FINANCIAL_STATEMENT"

            elif evt_type == "EVIDENCE_SUB_SECTION":
                new_sub = evt["val"]
                if new_sub != current_sub_sec and new_sub:
                    if current_lines:
                        flush_chunk(chunk_type="NORMAL")
                    current_sub_sec = new_sub

            elif evt_type in ["SECTION_HEADING", "UNDERLINE_HEADING"]:
                if calculate_content_length(current_lines) >= 100:
                    flush_chunk(chunk_type="NORMAL")

        current_lines.append(line)

    if current_lines:
        flush_chunk()

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

    print(f"[chunk_test.py] [Success] 成功完成 {len(chunks)} 個 MixRAG (H-RCL) 帶 Context Prefix 之 Chunks 切分！")
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
        print(f"[chunk_test.py] 在 '{markdown_dir}' 目錄下未找到任何 .md 檔案。")
        return False

    print(f"[chunk_test.py] 清理舊的 Chunks 資料夾內容...")
    clean_directory(output_dir)

    print(f"[chunk_test.py] 找到 {len(md_files)} 個 Markdown 檔案進行 Chunking...")
    for md_path in md_files:
        process_chunking(md_path, output_dir=output_dir)
    return True


if __name__ == "__main__":
    chunk_all_markdown_files()
