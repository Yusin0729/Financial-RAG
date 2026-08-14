# ==============================================================================
# 【模組說明】：main.py (財報蒟蒻 RAG 系統主選單介面與進入點)
# 【主要功能】：提供主控台互動選單，負責引導使用者進行 PDF 轉檔、向量資料庫建立與問答檢索。
# ==============================================================================
# 【使用之外部/內建套件 (Packages Used)】：
# 1. os: 控制系統命令（如 clear/cls 清除畫面）。
# 2. sys: 控制系統路徑、輸出流編碼與程式退出。
# 3. io: 處理 Windows 主控台 UTF-8 輸出編碼相容性。
# 4. markdown (內建模組): 呼叫 convert_pdf_to_markdown 執行 PDF 轉轉為 Markdown。
# ==============================================================================

import os
import sys

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def clear_screen():
    """清除主控台畫面"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    """主選單介面 layout"""
    enable_ocr = False
    enable_query_rewrite = False
    
    while True:
        clear_screen()
        ocr_status_str = "開啟" if enable_ocr else "關閉"
        rewrite_status_str = "開啟" if enable_query_rewrite else "關閉"
        
        print("=" * 74)
        print("                            財報蒟蒻 RAG 系統")
        print(f"      [系統狀態] OCR 模式: 【 {ocr_status_str:^4} 】 | 問題重寫: 【 {rewrite_status_str:^4} 】")
        print("=" * 74)
        print("  --- [ 建立資料庫 ] ---")
        print()
        print("  [1] 建立完整資料庫")
        print("  [2] PDF 轉為 Markdown")
        print("  [3] 讀取 Markdown 建立向量資料庫")
        print()
        print("  --- [ 回答問題 ] ---")
        print()
        print("  [4] 完整回答財報問題")
        print("  [5] 問題檢索")
        print("  [6] 重排序")
        print("  [7] 答案生成")
        print()
        print("  --- [ 系統控制 ] ---")
        print()
        print("  [S] 切換 OCR 狀態")
        print("  [W] 切換問題重寫狀態")
        print("  [Q] 退出系統")
        print("=" * 74)
        
        choice = input("請選擇操作選項 (1-7 或 S, W, Q): ").strip().upper()
        
        if choice == '1':
            print("\n=== [步驟 1/2] PDF 轉為 Markdown ===")
            try:
                from markdown import convert_pdf_to_markdown
                convert_pdf_to_markdown(enable_ocr=enable_ocr)
            except Exception as e:
                print(f"[錯誤] 轉檔過程發生異常: {e}")
            input("\n按任意鍵繼續...")
        elif choice == '2':
            print("\n=== PDF 轉檔為 Markdown ===")
            try:
                from markdown import convert_pdf_to_markdown
                convert_pdf_to_markdown(enable_ocr=enable_ocr)
            except Exception as e:
                print(f"[錯誤] 轉檔過程發生異常: {e}")
            input("\n按任意鍵繼續...")
        elif choice == '3':
            print("\n=== 讀取 Chunks 建立向量資料庫 ===")
            try:
                from embedding import build_all_embeddings
                build_all_embeddings()
            except Exception as e:
                print(f"[錯誤] 建立向量資料庫過程發生異常: {e}")
            input("\n按任意鍵繼續...")
        elif choice == '4':
            print("\n[系統提示] 選項 [4] 完整回答財報問題 (尚未連接功能)")
            input("按任意鍵繼續...")
        elif choice == '5':
            try:
                from retrieval import interactive_retrieval_search
                interactive_retrieval_search()
            except Exception as e:
                print(f"[錯誤] 問題檢索過程發生異常: {e}")
            input("\n按任意鍵繼續...")
        elif choice == '6':
            try:
                from reranker_test import interactive_rerank_search
                interactive_rerank_search()
            except Exception as e:
                print(f"[錯誤] 重排序過程發生異常: {e}")
            input("\n按任意鍵繼續...")
        elif choice == '7':
            print("\n[系統提示] 選項 [7] 答案生成 (尚未連接功能)")
            input("按任意鍵繼續...")
        elif choice == 'S':
            enable_ocr = not enable_ocr
            print(f"\n[系統提示] 切換 OCR 狀態為: {'開啟' if enable_ocr else '關閉'}")
            input("按任意鍵繼續...")
        elif choice == 'W':
            enable_query_rewrite = not enable_query_rewrite
            print(f"\n[系統提示] 切換問題重寫狀態為: {'開啟' if enable_query_rewrite else '關閉'}")
            input("按任意鍵繼續...")
        elif choice == 'Q':
            print("\n感謝使用財報蒟蒻 RAG 系統，再見！")
            sys.exit(0)
        else:
            print("\n[系統提示] 無效的選項，請輸入 1-7 或 S, W, Q。")
            input("按任意鍵繼續...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n系統被使用者強制中斷，再見！")
        sys.exit(0)
