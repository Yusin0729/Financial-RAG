# ==============================================================================
# 【模組說明】：tools/markdown_ocr_unlock.py (PDF 解密與 RapidOCR 工具模組)
# 【主要功能】：
# 1. unlock_pdf_if_needed: 檢查 PDF 是否有空白密碼鎖，有鎖則自動解密。
# 2. perform_ocr_fallback: 當遇到圖片掃描頁時，啟動 RapidOCR 引擎辨識圖中繁體中文。
# ==============================================================================
# 【使用之外部套件 (External Packages)】：
# 1. pikepdf: 用於檢查 PDF 加密狀態並進行解密存檔。
# 2. pdfplumber: 用於開啟 PDF 頁面並渲染成圖片 (PIL Image)。
# 3. numpy: 用於將 PIL 圖像轉換為陣列矩陣傳遞給 RapidOCR 運算。
# 4. rapidocr_onnxruntime: 光學文字辨識 (OCR) 引擎，負責識別圖片中的文字。
# ==============================================================================

import os
import sys

# 嘗試引用 pikepdf 套件（如果沒有安裝也不會直接崩潰）
try:
    import pikepdf
    _HAS_PIKEPDF = True
except ImportError:
    _HAS_PIKEPDF = False

def unlock_pdf_if_needed(pdf_path: str) -> str:
    """
    【函式功能】：檢查 PDF 是否有空白密碼鎖。如果有鎖，自動解碼並存成 .unlocked.tmp 暫存檔。
    """
    if not _HAS_PIKEPDF:
        return pdf_path
    try:
        with pikepdf.open(pdf_path) as pdf:
            if pdf.is_encrypted:
                unlocked_path = pdf_path + ".unlocked.tmp"
                pdf.save(unlocked_path)
                return unlocked_path
    except Exception:
        pass
    return pdf_path

def perform_ocr_fallback(pdf_path: str, page_num: int) -> str | None:
    """
    【函式功能】：針對掃描圖片頁，渲染成高解析度影像，並啟動 RapidOCR 引擎辨識繁體中文文字。
    """
    try:
        import pdfplumber
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        
        # 開啟 PDF 並渲染指定頁面
        with pdfplumber.open(pdf_path) as pdf:
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]          # 頁碼轉為 0-indexed 索引
                im = page.to_image(resolution=300)      # 設定解析度為 300 DPI (提升繁體筆劃清晰度)
                pil_img = im.original                   # 取得原始圖檔
                
                ocr = RapidOCR()                         # 初始化 RapidOCR 辨識引擎
                res = ocr(np.array(pil_img))            # 輸入影像並執行文字辨識
                
                ocr_text = ""
                if res:
                    res_list = res[0] if isinstance(res, tuple) else res
                    if hasattr(res_list, 'txts') and res_list.txts:
                        ocr_text = "\n".join(res_list.txts)
                    elif isinstance(res_list, list):
                        txts = [item[1] for item in res_list if isinstance(item, (list, tuple)) and len(item) >= 2]
                        ocr_text = "\n".join(txts)
                
                # 嘗試引用 opencc 進行簡轉繁
                try:
                    import opencc
                    cc = opencc.OpenCC('s2twp')
                    ocr_text = cc.convert(ocr_text)
                except ImportError:
                    pass

                if ocr_text.strip():
                    print(f"  -> 第 {page_num} 頁 RapidOCR 辨識完成。")
                    return ocr_text.strip()
    except Exception as e:
        print(f"  [Warning] 第 {page_num} 頁 OCR 處理失敗: {e}")
    return None

