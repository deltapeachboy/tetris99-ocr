import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("テトリス99 解析ツール（画像処理最適化版）")
st.write("画像の二値化による文字崩れを防ぎ、コントラスト調整によって認識精度を向上させました。")

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


    # 【新設計】AIに最適な「コントラスト強調＋滑らか拡大」を行う関数
    def preprocess_for_ocr(crop_img, zoom_factor=3):
        # グレースケール化
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # コントラスト調整（黒をより黒く、白をより白くして文字を際立たせる）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 滑らかに拡大処理
        resized = cv2.resize(contrast_enhanced, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        return resized


    # 星マークの化け補正（ドットやコロン、特定のアルファベットに化けた場合もカバー）
    def clean_level_text(text):
        if not text or text == "---":
            return "---"

        # 誤認識されやすい記号や文字を「★」に置換
        for char in ['+', '*', '.', ':', '"', '`', "'", 'x', 'r']:
            text = text.replace(char, '★')

        # 不要なスペースを除去
        text = text.replace(' ', '')
        return text


    with st.spinner("AI画像認識を実行中..."):
        for i in range(11):
            r_y_start = int(i * row_height)
            r_y_end = int((i + 1) * row_height)
            row_img = ranking_area[r_y_start:r_y_end, :]

            # 【調整】名前の開始位置を 102 にして、1文字目の欠けを防止
            name_crop = row_img[:, 102:330]  # 名前
            level_crop = row_img[:, 330:460]  # レベル/ランク

            # 専用のコントラスト強調・拡大を適用
            name_preprocessed = preprocess_for_ocr(name_crop, zoom_factor=3.0)
            level_preprocessed = preprocess_for_ocr(level_crop, zoom_factor=3.5)

            # --- OCR実行 ---
            # 1. 名前の読み取り
            name_res = reader.readtext(name_preprocessed, detail=0)
            detected_name = name_res[0] if name_res else "---"

            # 2. レベルの読み取り（数字、★、および化けやすい記号をあらかじめ許可）
            level_res = reader.readtext(
                level_preprocessed,
                detail=0,
                allowlist='0123456789★☆*+.:"`xr '
            )
            raw_level = level_res[0] if level_res else "---"
            detected_level = clean_level_text(raw_level)

            # 3. 順位は自動付与
            detected_rank = f"#{i + 1}"

            data_rows.append({
                "順位": detected_rank,
                "名前": detected_name,
                "レベル/ランク": detected_level
            })

    # 表示用データフレーム
    df = pd.DataFrame(data_rows)
    st.write("### 抽出データ（調整版）")
    st.dataframe(df)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="このテーブルをCSVとして保存",
        data=csv,
        file_name="tetris99_adjusted_table.csv",
        mime="text/csv"
    )