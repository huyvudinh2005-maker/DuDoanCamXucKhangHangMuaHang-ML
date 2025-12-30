import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Import thư viện SVM
from sklearn.metrics import accuracy_score

# Cấu hình trang web
st.set_page_config(
    page_title="Dự đoán Cảm xúc (SVM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Dự đoán Cảm xúc Khách hàng (SVM)")
st.markdown("Demo bài toán Sentiment Analysis sử dụng **Support Vector Machine (SVM)**.")

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Đọc dữ liệu (đảm bảo file csv nằm cùng thư mục)
        df = pd.read_csv("Customer_Sentiment.csv")
        return df
    except FileNotFoundError:
        st.error("Không tìm thấy file 'Customer_Sentiment.csv'. Vui lòng kiểm tra lại.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Thông tin dữ liệu")
    st.sidebar.write(f"Tổng số dòng: {df.shape[0]}")
    
    if st.sidebar.checkbox("Xem dữ liệu gốc (10 dòng đầu)"):
        st.subheader("Dữ liệu mẫu")
        st.dataframe(df.head(10))

    # -------------------------------------------------------------------
    # 2. TRỰC QUAN HÓA
    # -------------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phân bố Cảm xúc")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.subheader("Tỷ lệ")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # -------------------------------------------------------------------
    # 3. HUẤN LUYỆN MÔ HÌNH (SVM)
    # -------------------------------------------------------------------
    @st.cache_resource
    def train_model(data):
        # Lấy dữ liệu
        X_text = data['review_text'].fillna('')
        y = data['sentiment']

        # Vector hóa (TF-IDF)
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        X = tfidf.fit_transform(X_text)

        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Khởi tạo mô hình SVM
        # kernel='linear' thường tốt cho text classification
        # probability=True để tính được % độ tin cậy (nhưng sẽ làm train chậm hơn một chút)
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Đánh giá
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return model, tfidf, acc

    st.write("---")
    st.info("Đang huấn luyện mô hình SVM... (Quá trình này có thể mất 1-2 phút vì SVM chậm hơn Logistic Regression)")
    
    # Hiển thị spinner trong khi train
    with st.spinner('Đang train model... Vui lòng đợi...'):
        model, tfidf_vectorizer, accuracy = train_model(df)
    
    st.success(f"Huấn luyện xong! Độ chính xác trên tập kiểm tra: **{accuracy:.2%}**")

    # -------------------------------------------------------------------
    # 4. DỰ ĐOÁN
    # -------------------------------------------------------------------
    st.header("🔍 Thử nghiệm Dự đoán")
    user_input = st.text_area("Nhập review của khách hàng:", placeholder="Type something here...")

    if st.button("Phân tích"):
        if user_input.strip() == "":
            st.warning("Vui lòng nhập nội dung!")
        else:
            # Dự đoán
            input_vec = tfidf_vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            probability = model.predict_proba(input_vec).max()

            st.write("---")
            st.subheader("Kết quả:")
            
            if prediction == "positive":
                st.success(f"😊 Tích cực (Positive) - Độ tin cậy: {probability:.2%}")
            elif prediction == "negative":
                st.error(f"😡 Tiêu cực (Negative) - Độ tin cậy: {probability:.2%}")
            else:
                st.info(f"😐 Trung tính (Neutral) - Độ tin cậy: {probability:.2%}")