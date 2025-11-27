@echo off
echo 正在啟動冠毅的股票分析儀...
cd /d "%~dp0"
python -m streamlit run stock_analyze.py
pause

5.  存檔並關閉。

**完成！**
以後你只要點兩下這個 **`開始分析.bat`**，它就會自動跳出一個黑視窗幫你執行指令，然後打開網頁給你看。你可以把這個檔案傳送到桌面當作捷徑！