# ==============================================================================
# 【模組說明】：tools/scratch_compare_ocr.py (臨時測試腳本：Marker PDF 轉 JSON 測試)
# 【主要功能】：讀取 Financial Statements/ 下的 PDF 財報檔案，
#              使用 Marker 進行 AI 版面分析並將結果直接輸出為 JSON 檔案格式。
#
# ==============================================================================
# 【主要核心工具與模組簡介】：
# 1. os (標準函式庫模組)
#    - 用途：處理檔案與資料夾路徑，設定系統環境變數。
# 2. sys (標準函式庫模組)
#    - 用途：調整 Python 套件搜尋路徑 sys.path。
# 3. time (標準函式庫模組)
#    - 用途：計算模型載入與轉檔耗時。
# 4. marker-pdf (外部深度學習套件)
#    - 用途：進行 PDF 的 AI 版面分析與表格/文字結構化輸出。
# ==============================================================================

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 自動載入專案環境 site-packages 套件路徑
_site_packages = os.path.join(PROJECT_ROOT, "Lib", "site-packages")
if os.path.exists(_site_packages) and _site_packages not in sys.path:
    sys.path.append(_site_packages)

# 設定 llama-server.exe 路徑
llama_bin = os.path.join(PROJECT_ROOT, "Scripts", "llama-server.exe")
if os.path.exists(llama_bin):
    os.environ["LLAMA_CPP_BINARY"] = llama_bin


def run_marker_to_json(pdf_path: str, output_path: str, target_pages: list[int] = [8]):
    """【Marker 原生 JSON 模式】：指定頁數使用 Marker 進行 AI 版面分析並輸出為 JSON"""
    print(f"\n=== 正在載入 Marker 深度學習模型 (指定頁數: 第 {target_pages} 頁, 輸出格式: JSON) ===")
    start_total = time.time()
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser

        print("  [初始化] 正在載入 Marker / Surya 模型與權重...")
        page_str = ",".join(str(p - 1) for p in target_pages)
        config_parser = ConfigParser({
            "page_range": page_str,
            "use_llm": False,
            "output_format": "json"
        })

        models = create_model_dict()
        converter = PdfConverter(
            artifact_dict=models,
            config=config_parser.generate_config_dict(),
            renderer=config_parser.get_renderer(),
            llm_service=None
        )
        # 過濾掉 LLM 後處理器，使用純 AI 視覺與排版引擎
        converter.processor_list = [p for p in converter.processor_list if "LLM" not in p.__class__.__name__]

        print(f"  開始執行 Marker AI 版面分析 (轉換頁碼: {target_pages})...")

        rendered = converter(pdf_path)
        json_text, ext, images = text_from_rendered(rendered)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_text)

        total_elapsed = time.time() - start_total
        print(f"[成功] Marker JSON 轉檔完成！成果已輸出至: {output_path}")
        print(f"  ==> 指定 {len(target_pages)} 頁 Marker 總耗時: {total_elapsed:.2f} 秒\n")

    except Exception as e:
        total_elapsed = time.time() - start_total
        print(f"[錯誤] Marker 執行失敗 (耗時 {total_elapsed:.2f} 秒): {e}")


if __name__ == "__main__":
    # 預設測試 2023Q4 財報檔
    pdf_file = os.path.join(PROJECT_ROOT, "Financial Statements", "2023Q4_2303_Financial_Report.pdf")
    if not os.path.exists(pdf_file):
        pdf_file = os.path.join(PROJECT_ROOT, "Financial Statements", "2016Q4_2303_Financial_Report.pdf")

    # 輸出至 test/ 資料夾中的 JSON 檔案
    marker_out_json = os.path.join(PROJECT_ROOT, "test", "2023Q4_Marker_JSON.json")

    print(f"目標 PDF 檔案: {pdf_file}")
    if not os.path.exists(pdf_file):
        print(f"[錯誤] 找不到 PDF 檔案: {pdf_file}")
    else:
        # 測試第 8 頁（或可自訂頁碼列表）的 Marker JSON 轉檔
        run_marker_to_json(pdf_file, marker_out_json, target_pages=[8])
