import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Korean News Headline EDA")

uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    # load data
    df = pd.read_csv(uploaded_file)
    st.write("### 데이터 미리보기")
    st.write(df.head())

    # 데이터 기본 정보 표시
    st.write("### 데이터 정보")
    st.write(df.describe(include='all'))

    # Null 값 확인
    st.write("### Null 값 확인")
    st.write(df.isnull().sum())

    # 클래스 분포 시각화
    st.write("### 클래스 분포")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='target', ax=ax)
    plt.title("Class Distribution")
    st.pyplot(fig)

    # 텍스트 길이 분포
    st.write("### 텍스트 길이 분포")
    df['text_length'] = df['text'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(df['text_length'], kde=True, ax=ax)
    plt.title("Text Length Distribution")
    st.pyplot(fig)

    # 텍스트 샘플과 라벨 보기
    st.write("### 텍스트 샘플과 라벨")
    num_samples = st.slider("샘플 수 선택", 1, 20, 5)
    st.write(df[['text', 'target']].sample(num_samples))

    # 특정 라벨에 따른 텍스트 샘플 확인
    st.write("### 특정 라벨의 텍스트 샘플")
    selected_label = st.selectbox("라벨 선택", sorted(df['target'].unique()))
    label_samples = df[df['target'] == selected_label]['text'].sample(num_samples)
    st.write(label_samples)


# Streamlit 설정
st.header("Classified Output Visualization")

# # 데이터 디렉터리 설정
# BASE_DIR = os.getcwd()
# DATA_DIR = os.path.join(BASE_DIR, 'data')

# # classified_output.csv 파일 경로
# csv_file_path = os.path.join(DATA_DIR, 'classified_output_zero_shot_numeric.csv')

# # Read csv file
# try:
#     df = pd.read_csv(csv_file_path)
# except FileNotFoundError:
#     st.error("파일을 찾을 수 없습니다.")
#     st.stop()

classified_output = st.file_uploader("라벨 수정한 CSV 파일 업로드", type="csv")

if classified_output:
    df = pd.read_csv(classified_output)
    # 데이터 미리보기
    st.write("### 데이터 미리보기")
    st.write(df.head())

    # 데이터 기본 정보 표시
    st.write("### 데이터 정보")
    st.write(df.describe(include='all'))

    # Null 값 확인
    st.write("### Null 값 확인")
    st.write(df.isnull().sum())

    # 타겟 라벨 정의
    labels = {
        0: "생활문화",
        1: "스포츠",
        2: "정치",
        3: "사회",
        4: "IT_과학",
        5: "경제",
        6: "국제"
    }

    # 클래스 분포 시각화
    st.write("### Target과 Predicted Target 클래스 분포 비교")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(data=df, x='target', color='skyblue', label='Target', alpha=0.6, ax=ax)
    sns.countplot(data=df, x='predicted_target', color='salmon', label='Predicted Target', alpha=0.6, ax=ax)

    plt.title("Target vs Predicted Target Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    st.pyplot(fig)

    # target과 predicted_target이 같은 경우
    st.write("### Target과 Predicted Target이 같은 경우")
    correct_predictions = df[df['target'] == df['predicted_target']]
    num_correct = len(correct_predictions)

    if num_correct > 0:
        st.write(f"개수: {num_correct}")
        st.write(correct_predictions[['text', 'target', 'predicted_target']].head(20).assign(
            target=correct_predictions['target'].map(labels),
            predicted_target=correct_predictions['predicted_target'].map(labels)
        ))
    else:
        st.write("일치하는 데이터가 없습니다.")

    # target과 predicted_target이 다른 경우
    st.write("### Target과 Predicted Target이 다른 경우")
    incorrect_predictions = df[df['target'] != df['predicted_target']]
    num_incorrect = len(incorrect_predictions)

    if num_incorrect > 0:
        st.write(f"개수: {num_incorrect}")
        st.write(incorrect_predictions[['text', 'target', 'predicted_target']].head(20).assign(
            target=incorrect_predictions['target'].map(labels),
            predicted_target=incorrect_predictions['predicted_target'].map(labels)
        ))
    else:
        st.write("불일치하는 데이터가 없습니다.")