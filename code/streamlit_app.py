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

