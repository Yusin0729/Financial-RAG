# ==============================================================================
# 【模組說明】：markdown_test.py (PDF 轉 Markdown 轉檔測試模組 - MixRAG 版本)
# 【功能概要】：讀取 'Financial Statements/' 資料夾內的 PDF 財報檔案，
#              動態解析第二頁目錄頁碼對照表，自動標記各頁大標與子標 Metadata，
#              為表格加上獨立標籤，並呼叫 chunk_test 執行細粒度 H-RCL Chunking。
# ==============================================================================
# 【主要核心工具與模組簡介】：
# 1. pdf_inspector: 快速解析數位版 PDF 內容，自動將表格轉換為 Markdown 管道符號 '|' 格式。
# 2. tools.markdown_ocr_unlock: 檢查 PDF 空白密碼鎖並自動解密與 OCR Fallback。
# 3. tools.table_tagger: 為表格加上獨立 Metadata 標籤並追蹤 Note Title 狀態。
# 4. shutil: 自動清空 Markdown 與 Chunks 資料夾內的舊檔案。
# 5. chunk_test: 執行 MixRAG H-RCL 細粒度表格與文本 Chunking。
# ==============================================================================

import os
import sys
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

_parent_sp = os.path.join(os.path.dirname(CURRENT_DIR), "Lib", "site-packages")
if os.path.exists(_parent_sp) and _parent_sp not in sys.path:
    sys.path.append(_parent_sp)

import glob
import re
import pdf_inspector

from tools.markdown_ocr_unlock import unlock_pdf_if_needed, perform_ocr_fallback
from tools.table_tagger import process_page_tables, merge_cross_page_tables


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
            print(f"[markdown_test.py] [Warning] 無法刪除舊檔案 '{item_path}': {e}")


def normalize_chinese_numbers(text: str) -> str:
    trans_table = str.maketrans("⼀⼆三四五六七八九⼗", "一二三四五六七八九十")
    return text.translate(trans_table)


def parse_page_range(range_str: str) -> tuple[int, int]:
    nums = re.findall(r'\d+', range_str)
    if not nums:
        return (0, 0)
    if len(nums) == 1:
        val = int(nums[0])
        return (val, val)
    return (int(nums[0]), int(nums[1]))


def build_toc_mapping(result_pages) -> list[dict]:
    toc_entries = []
    toc_text = ""
    for p in result_pages[:3]:
        raw = p.markdown or ""
        if "目錄" in raw or "⽬ 錄" in raw or "項 ⽬" in raw or "項 目" in raw:
            toc_text += "\n" + raw

    if not toc_text and len(result_pages) >= 2:
        toc_text = result_pages[1].markdown or ""

    toc_text = normalize_chinese_numbers(toc_text)

    item_pattern = re.compile(
        r'([一二三四五六七八九十]+\s*[、\.]\s*[\u4e00-\u9fa5]+|\([一二三四五六七八九十]+\)\s*[\u4e00-\u9fa5]+)'
    )
    items = item_pattern.findall(toc_text)

    pages_part = ""
    if "頁 次" in toc_text or "⾴ 次" in toc_text:
        parts = re.split(r'頁\s*次|⾴\s*次', toc_text)
        if len(parts) > 1:
            pages_part = parts[1]
    else:
        pages_part = toc_text

    range_pattern = re.compile(r'\d+(?:\s*[-~－]\s*\d+)?')
    page_ranges = range_pattern.findall(pages_part)

    current_main = ""
    page_range_idx = 0

    for item in items:
        item_str = item.strip()
        if re.match(r'^[一二三四五六七八九十]+', item_str):
            current_main = item_str
            sub_title = ""
        else:
            sub_title = item_str

        if page_range_idx < len(page_ranges):
            p_str = page_ranges[page_range_idx]
            page_range_idx += 1
            sp, ep = parse_page_range(p_str)
        else:
            sp, ep = (0, 0)

        toc_entries.append({
            "main": current_main,
            "sub": sub_title,
            "start_page": sp,
            "end_page": ep
        })

    return toc_entries


def get_sections_for_page(page_num: int, toc_mapping: list[dict], current_main: str, current_sub: str) -> tuple[str, str]:
    matched_mains = []
    matched_subs = []

    for entry in toc_mapping:
        if entry["start_page"] > 0 and entry["start_page"] <= page_num <= entry["end_page"]:
            if entry["main"] and entry["main"] not in matched_mains:
                matched_mains.append(entry["main"])
            if entry["sub"] and entry["sub"] not in matched_subs:
                matched_subs.append(entry["sub"])

    main_res = "; ".join(matched_mains) if matched_mains else current_main
    if matched_mains:
        sub_res = "; ".join(matched_subs) if matched_subs else ""
    else:
        sub_res = current_sub
    return main_res, sub_res


def convert_pdf_to_markdown(reports_dir: str = None, output_dir: str = None, enable_ocr: bool = False) -> bool:
    if reports_dir is None:
        reports_dir = os.path.join(CURRENT_DIR, "Financial Statements")

    if output_dir is None:
        output_dir = os.path.join(CURRENT_DIR, "markdown")

    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(reports_dir, "*.pdf"))
    if not pdf_files:
        print(f"[markdown_test.py] 沒有在 '{reports_dir}' 資料夾下找到任何 PDF 檔案。")
        return False

    # 清空輸出目錄 (markdown/ 與 Chunks/)，確保資料夾內僅保留本次產生的檔案
    print(f"[markdown_test.py] 清理舊的 Markdown 與 Chunks 資料夾內容...")
    clean_directory(output_dir)
    chunks_dir = os.path.join(CURRENT_DIR, "Chunks")
    clean_directory(chunks_dir)

    mode_str = "【開啟 OCR 全量辨識模式】" if enable_ocr else "【關閉 OCR 快速提取模式】"
    print(f"[markdown_test.py] 找到 {len(pdf_files)} 個 PDF 檔案進行轉換。模式: {mode_str}")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        name_only, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name_only}.md")

        print(f"[markdown_test.py] 正在解析 '{filename}'...")

        _m = re.match(r'(\d{4})[Qq]\d*_(\d+)', filename)
        year = _m.group(1) if _m else "YYYY"
        company_id = _m.group(2) if _m else "XXXX"

        working_path = unlock_pdf_if_needed(pdf_path)

        try:
            result = pdf_inspector.extract_pages_markdown(working_path)
            md_lines = []
            extracted_tables_all = []

            toc_mapping = build_toc_mapping(result.pages)

            running_main = ""
            running_sub = ""
            table_counter = 1

            for pm in result.pages:
                page_num = pm.page + 1
                markdown_text = pm.markdown or ""

                if pm.needs_ocr or len(markdown_text.strip()) < 30:
                    if enable_ocr:
                        print(f"  -> 第 {page_num} 頁為掃描頁。正在進行 RapidOCR 辨識...")
                        ocr_text = perform_ocr_fallback(working_path, page_num)
                        if ocr_text:
                            markdown_text = ocr_text
                        else:
                            markdown_text = f"(第 {page_num} 頁為掃描頁，OCR 未找到文字)"
                    else:
                        print(f"  -> 第 {page_num} 頁為掃描或空白頁。[OCR 已關閉] 已跳過。")
                        markdown_text = f"(第 {page_num} 頁為掃描頁，OCR 模式未開啟)"

                norm_text = normalize_chinese_numbers(markdown_text)

                m_main = re.search(r'([一二三四五六七八九十]+\s*[、\.]\s*[\u4e00-\u9fa5]+)', norm_text)
                if m_main:
                    running_main = m_main.group(1).strip()
                    running_sub = ""

                m_sub = re.search(r'(\([一二三四五六七八九十]+\)\s*[\u4e00-\u9fa5]+)', norm_text)
                if m_sub:
                    running_sub = m_sub.group(1).strip()

                main_section, sub_section = get_sections_for_page(page_num, toc_mapping, running_main, running_sub)
                page_id = f"{year}_{company_id}_P{page_num}"

                processed_text, page_tables, table_counter = process_page_tables(
                    page_content=markdown_text,
                    doc_id=name_only,
                    page_id=page_id,
                    main_section=main_section,
                    sub_section=sub_section,
                    doc_year=year,
                    table_counter_start=table_counter
                )
                extracted_tables_all.extend(page_tables)

                md_lines.append(f"<!-- Page: {page_id} -->")
                md_lines.append(f"<!-- Doc_Id: {name_only} -->")
                md_lines.append(f"<!-- Years: {year} -->")
                md_lines.append(f"<!-- Relevant_items: -->")
                md_lines.append(f"<!-- Evidence_Main_Section: {main_section} -->")
                md_lines.append(f"<!-- Evidence_Sub_Section: {sub_section} -->")
                md_lines.append(processed_text)
                md_lines.append("")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            print(f"[markdown_test.py] 成功儲存 Markdown 檔至 '{output_path}'")

            # 呼叫 chunk_test 執行細粒度 H-RCL Chunking 並將結果輸出至 Chunks 資料夾
            try:
                from chunk_test import process_chunking
                process_chunking(output_path)
            except Exception as chunk_err:
                print(f"[markdown_test.py] [Warning] 執行 chunk_test 失敗: {chunk_err}")

        except Exception as e:
            print(f"[markdown_test.py] [Error] 轉換 '{filename}' 失敗: {e}")

        finally:
            if working_path != pdf_path and os.path.exists(working_path):
                try:
                    os.remove(working_path)
                except Exception:
                    pass

    print("[markdown_test.py] 所有 PDF 轉換為 Markdown 完畢！")
    return True


if __name__ == "__main__":
    convert_pdf_to_markdown(enable_ocr=False)
