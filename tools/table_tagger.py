# ==============================================================================
# 【模組說明】：tools/table_tagger.py (表格標籤與區塊解析處理模組)
# 【功能概要】：辨識 Markdown 中的表格區塊，動態追蹤 Evidence_Note_Title 與 Evidence_Sub_Items，
#              為表格加上獨立標籤 (TABLE_START / TABLE_END) 並保留於同一 Markdown 檔內。
#
# ==============================================================================
# 【主要核心工具與模組簡介】：
# 1. re (標準函式庫模組)
#    - 用途：使用正則表達式匹配表格標題、會計科目編號與層級結構。
# ==============================================================================

import re

def normalize_chinese_numbers(text: str) -> str:
    """將康熙部首國字數字 (如 ⼀⼆三) 轉為標準 CJK 國字數字 (一二三)"""
    trans_table = str.maketrans("⼀⼆三四五六七八九⼗", "一二三四五六七八九十")
    return text.translate(trans_table)



def extract_table_title(table_text: str, current_note_title: str) -> str:
    """從表格內容上方或第一行嘗試提取表格標題描述"""
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    for line in lines[:3]:
        clean_line = line.replace("|", "").strip()
        if any(kw in clean_line for kw in ["資產負債表", "損益表", "權益變動表", "現金流量表", "附表"]):
            return clean_line
    return current_note_title or "財務表格"

def extract_table_years(raw_table_str: str, note_title: str = "", default_year: str = "") -> str:
    """
    精準提取表格年份：
    1. 首選：僅掃瞄表格「最上面列 (Header)」與「最左欄 (col_0)」。
    2. 備援：若首選未找到任何年份，才掃瞄表格上方說明/標題文字，且僅在包含 '民國+年份+年' 格式時才提取。
    """
    chinese_digit_trans = str.maketrans("○〇零一二三四五六七八九", "000123456789")
    years_found = set()

    lines = [line.strip() for line in raw_table_str.strip().split("\n") if line.strip()]
    if not lines:
        return default_year

    # 1. 首選：僅抽取表格「最上面列 (Header Row)」與「最左欄 (col_0)」
    target_cells = []
    # 抽取表格前 2 列 (欄位標頭區)
    for line in lines[:2]:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        target_cells.extend(parts)

    # 抽取後續每一行的最左欄 (col_0)
    for line in lines[2:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[1]:
            target_cells.append(parts[1])

    header_left_text = " ".join(target_cells)
    header_left_text_norm = header_left_text.translate(chinese_digit_trans)

    # 比對西元年 (如 2020, 2019)
    west_years = re.findall(r'(?<!\d)(19\d{2}|20\d{2})(?:年|年度|\/|\.|\b)', header_left_text_norm)
    for y in west_years:
        years_found.add(int(y))

    # 比對民國年 (如 109年, 108年度, 109.12.31, 一○九年)
    roc_matches = re.findall(r'(?<!\d)(1\d{2})\s*(?:年|年度|\/|\.|\b)', header_left_text_norm)
    for ry in roc_matches:
        roc_val = int(ry)
        if 90 <= roc_val <= 130:
            years_found.add(roc_val + 1911)

    if years_found:
        sorted_years = sorted(list(years_found), reverse=True)
        return ", ".join(str(y) for y in sorted_years)

    # 2. 備援：若最上列/最左欄未找到，掃瞄上面一列 (Note Title / 說明文字)
    # 規則：只有在具備出現 "民國 + 年份數字 + 年" 格式時才加進來
    text_above = (note_title or "").translate(chinese_digit_trans)
    roc_above_matches = re.findall(r'民國\s*(1\d{2}|\d{2,3})\s*年', text_above)
    for ry in roc_above_matches:
        roc_val = int(ry)
        if 90 <= roc_val <= 130:
            years_found.add(roc_val + 1911)

    if years_found:
        sorted_years = sorted(list(years_found), reverse=True)
        return ", ".join(str(y) for y in sorted_years)

    return default_year

def extract_table_relevant_items(raw_table_str: str, note_title: str = "") -> str:
    """
    從表格第一欄 (col_0) 與 附註標題中提取會計科目與關鍵概念：
    - 不限制數量，進行精準去重
    - 剔除所有數字與其他特殊符號，但保留中英文與括號 ( ) / （ ）
    """
    items = []
    seen = set()

    # 1. 處理附註標題 Note Title
    if note_title:
        clean_note = re.sub(r'</?u>', '', note_title)
        clean_note = re.sub(r'^\(?\d+\)?\s*[\.、]?\s*', '', clean_note).strip()
        clean_note = re.sub(r'[^\u4e00-\u9fa5A-Za-z\(\)（）]', '', clean_note).strip()
        clean_note = re.sub(r'\(\s*\)|（\s*）', '', clean_note).strip()
        if clean_note and len(clean_note) >= 2 and clean_note not in seen:
            seen.add(clean_note)
            items.append(clean_note)

    # 2. 處理表格第一欄 (col_0)
    lines = raw_table_str.strip().split("\n")
    ignore_kws = {"項目", "代碼", "代號", "單位", "金額", "比率", "附註", "合計", "總計", "小計", "註"}

    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            col_0 = parts[1]  # 第一個管道與第二個管道之間的內容
            
            clean_col = re.sub(r'</?u>', '', col_0)
            clean_col = re.sub(r'^\(?\d+\)?\s*[\.、]?\s*', '', clean_col).strip()
            # 只保留中英文與半形/全形括號，剔除數字與其他符號
            clean_col = re.sub(r'[^\u4e00-\u9fa5A-Za-z\(\)（）]', '', clean_col).strip()
            clean_col = re.sub(r'\(\s*\)|（\s*）', '', clean_col).strip()  # 清理空括號

            if not clean_col or clean_col in ignore_kws:
                continue

            if len(clean_col) >= 2 and clean_col not in seen:
                seen.add(clean_col)
                items.append(clean_col)

    return ", ".join(items)

def process_page_tables(page_content: str, doc_id: str, page_id: str,
                        main_section: str, sub_section: str,
                        doc_year: str = "",
                        table_counter_start: int = 1) -> tuple[str, list[dict], int]:
    """
    解析單頁 Markdown 內容中的表格：
    - 維持 running_note_title 及 running_sub_item 狀態
    - 生成包含獨立標籤的 TABLE_START / TABLE_END 區塊
    - 回傳：(處理後的頁面內文, 本頁抽取的獨立表格清單, 更新後的表格計數)
    """
    lines = page_content.split("\n")
    new_lines = []
    extracted_tables = []
    
    running_note_title = ""
    running_sub_item = ""
    table_counter = table_counter_start

    i = 0
    while i < len(lines):
        line = lines[i]
        norm_line = normalize_chinese_numbers(line.strip())

        # 1. 檢測 Note Title (如: "1. 現金及約當現金", "附表四：期末持有...")
        m_note = re.search(r'(\d+\.\s*<u>.*?</u>|\d+\.\s*[\u4e00-\u9fa5]+|附表[一二三四五六七八九十\d]+[：:].*)', norm_line)
        if m_note:
            running_note_title = re.sub(r'</?u>', '', m_note.group(1)).strip()

        # 2. 檢測 Sub Item (如: "(1) 採用權益法之投資明細如下：")
        m_sub_item = re.search(r'(\(\d+\)\s*[\u4e00-\u9fa5A-Za-z0-9_\-\s：:]+)', norm_line)
        if m_sub_item:
            running_sub_item = m_sub_item.group(1).strip()

        # 3. 檢測表格開始 (管道符號 | 開頭)
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            
            raw_table_str = "\n".join(table_lines)
            table_id = f"{page_id}_T{table_counter}"
            table_title = running_note_title or "財務表格"
            table_years = extract_table_years(raw_table_str, running_note_title, doc_year)
            table_relevant_items = extract_table_relevant_items(raw_table_str, running_note_title)

            # 組裝獨立表格 Metadata
            tag_lines = [
                "<!-- TABLE_START -->",
                f"<!-- Table_Id: {table_id} -->",
                f"<!-- Table_Title: {table_title} -->",
                f"<!-- Doc_Id: {doc_id} -->",
                f"<!-- Years: {table_years} -->",
                f"<!-- Relevant_items: {table_relevant_items} -->",
                f"<!-- Evidence_Main_Section: {main_section} -->",
                f"<!-- Evidence_Sub_Section: {sub_section} -->",
                raw_table_str,
                "<!-- TABLE_END -->"
            ]
            full_table_block = "\n".join(tag_lines)

            table_meta = {
                "table_id": table_id,
                "table_title": table_title,
                "doc_id": doc_id,
                "page_id": page_id,
                "main_section": main_section,
                "sub_section": sub_section,
                "note_title": running_note_title,
                "sub_item": running_sub_item,
                "raw_table": raw_table_str,
                "full_block": full_table_block
            }
            extracted_tables.append(table_meta)
            table_counter += 1

            # 留在原始 Markdown 檔案中，並帶有 TABLE_START / TABLE_END 標籤
            new_lines.append(full_table_block)
            continue
        else:
            new_lines.append(line)
            i += 1

    processed_content = "\n".join(new_lines)
    return processed_content, extracted_tables, table_counter

def merge_cross_page_tables(tables: list[dict]) -> list[dict]:
    """
    合併隔頁或跨頁連續的相同主題表格：
    如果前後兩個表格屬於相同的 main_section、sub_section、note_title 且頁碼連續，將其 raw_table 進行合併。
    """
    if not tables:
        return []

    merged = []
    current_table = tables[0].copy()

    for next_table in tables[1:]:
        # 提取頁碼數字進行比對
        curr_p = int(re.search(r'_P(\d+)', current_table["page_id"]).group(1)) if re.search(r'_P(\d+)', current_table["page_id"]) else 0
        next_p = int(re.search(r'_P(\d+)', next_table["page_id"]).group(1)) if re.search(r'_P(\d+)', next_table["page_id"]) else 0

        same_section = (current_table["main_section"] == next_table["main_section"] and 
                        current_table["sub_section"] == next_table["sub_section"])
        same_note = (current_table["note_title"] == next_table["note_title"])
        is_adjacent_page = (next_p == curr_p + 1 or next_p == curr_p)

        if same_section and (same_note or is_adjacent_page):
            # 進行表格行合併（移除第二個表格的表頭標頭管道行）
            next_lines = next_table["raw_table"].split("\n")
            clean_next_lines = [l for l in next_lines if not re.match(r'\|?\s*[-:]+\s*\|', l)]
            
            # 如果欄位名稱對齊列重複出現，將其跳過
            if clean_next_lines and ("代 碼" in clean_next_lines[0] or "項 目" in clean_next_lines[0] or "有價證券" in clean_next_lines[0]):
                clean_next_lines = clean_next_lines[1:]

            current_table["raw_table"] += "\n" + "\n".join(clean_next_lines)
            
            # 更新 full_block
            tag_lines = [
                "<!-- TABLE_START -->",
                f"<!-- Table_Id: {current_table['table_id']} -->",
                f"<!-- Table_Title: {current_table['table_title']} -->",
                f"<!-- Doc_Id: {current_table['doc_id']} -->",
                "<!-- Years: -->",
                "<!-- Relevant_items: -->",
                f"<!-- Evidence_Main_Section: {current_table['main_section']} -->",
                f"<!-- Evidence_Sub_Section: {current_table['sub_section']} -->",
                current_table["raw_table"],
                "<!-- TABLE_END -->"
            ]
            current_table["full_block"] = "\n".join(tag_lines)
        else:
            merged.append(current_table)
            current_table = next_table.copy()

    merged.append(current_table)
    return merged
