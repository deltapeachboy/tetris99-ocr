import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("テトリス99 解析ツール（チーム番号対応版）")
st.write("背景透けの完全除去と、チームバトルのチーム番号（2, 3, 9など）の取得に完全対応しました。")

uploaded_file = st.file_uploader("リザルト画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 1. 1280x720にサイズを統一
    img_resized = cv2.resize(img, (1280, 720))

    # 中央リザルト全体の範囲
    y_start, y_end = 195, 655
    x_start, x_end = 370, 830
    ranking_area = img_resized[y_start:y_end, x_start:x_end]

    st.image(ranking_area, caption="解析対象エリア", use_container_width=True)

    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['ja', 'en'])

    reader = load_ocr()

    # 11行分の高さを計算
    row_height = (y_end - y_start) / 11.0
    data_rows = []

    # 【極高精度】透ける背景を完全に消し去り、文字だけを滑らかに浮かび上がらせる前処理
    def preprocess_perfect(crop_img, zoom_factor=3.0, threshold_val=180):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # 滑らかに拡大
        resized = cv2.resize(gray, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        # 二値化で背景のキャラクター（服や髪）を真っ黒にして完全に消去
        _, thresh = cv2.threshold(resized, threshold_val, 255, cv2.THRESH_BINARY)
        return thresh

    # 星マークの置換補正
    def clean_level_text(text):
        if not text or text == "---":
            return "---"
        for char in ['+', '*', '.', ':', '"', '`', "'", 'x', 'r', 'y', 's', 'A', 'a', 'g']:
            text = text.replace(char, '★')
        text = text.replace(' ', '')
        return text

    # 順位テキストのクリーンアップ（# を先頭に付与し、余計な文字を排除）
    def clean_rank_text(text, index):
        if not text:
            return f"#{index + 1}"
        # 数字と#以外を除去
        cleaned = "".join([c for c in text if c.isdigit() or c == '#'])
        if not cleaned.startswith('#'):
            cleaned = f"#{cleaned}"
        return cleaned

    with st.spinner("AI画像認識を実行中（チーム番号・順位抽出中）..."):
        for i in range(11):
            r_y_start = int(i * row_height)
            r_y_end = int((i + 1) * row_height)
            row_img = ranking_area[r_y_start:r_y_end, :]

            # --- 各エリアの切り出し ---
            # 順位: 0 〜 55px
            # 名前: 100 〜 325px （チームバトルの「チーム番号」を含めるため 100 開始に設定）
            # レベル: 335 〜 430px
            rank_crop = row_img[:, 0:55]
            name_crop = row_img[:, 100:325]
            level_crop = row_img[:, 335:430]

            # すべてのエリアに高精度背景消去を適用
            rank_preprocessed = preprocess_perfect(rank_crop, zoom_factor=3.0, threshold_val=170)
            name_preprocessed = preprocess_perfect(name_crop, zoom_factor=3.0, threshold_val=180)
            level_preprocessed = preprocess_perfect(level_crop, zoom_factor=3.5, threshold_val=180)

            # --- OCR実行 ---
            # 1. 順位の動的読み取り
            rank_res = reader.readtext(rank_preprocessed, detail=0, allowlist='0123456789#')
            raw_rank = rank_res[0] if rank_res else f"#{i+1}"
            detected_rank = clean_rank_text(raw_rank, i)

            # 2. 名前の読み取り
            name_res = reader.readtext(name_preprocessed, detail=0)
            detected_name = name_res[0] if name_res else "---"

            # 3. レベルの読み取り
            level_res = reader.readtext(
                level_preprocessed,
                detail=0,
                allowlist='0123456789★☆*+.:"`xry sAa'
            )
            raw_level = level_res[0] if level_res else "---"
            detected_level = clean_level_text(raw_level)

            data_rows.append({
                "順位": detected_rank,
                "名前": detected_name,
                "レベル/ランク": detected_level
            })

    # 表示用データフレーム
    df = pd.DataFrame(data_rows)
    st.write("### 抽出データ（チーム番号・順位対応版）")
    st.dataframe(df)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="このテーブルをCSVとして保存",
        data=csv,
        file_name="tetris99_complete_team_table.csv",
        mime="text/csv"
    )