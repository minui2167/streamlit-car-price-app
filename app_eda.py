import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def run_eda():
    st.subheader('데이터 분석')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

    # 라디오버튼을 이용해서, 데이터프레임과, 통계치를 선택해서
    # 보여줄 수 있도록 만든다.

    radio_menu = ['데이터프레임', '통계치']
    selected = st.radio('선택하세요', radio_menu)

    if selected == radio_menu[0]:
        st.dataframe(car_df)
    elif selected == radio_menu[1]:
        st.dataframe(car_df.describe())

    # 컬럼명을 보여주고, 컬럼을 선택하면,
    # 해당 컬럼의 최대값 데이터와, 최소값 데이터를 보여준다.

    col_list = car_df.columns[4:]
    selected_col = st.selectbox('최대 최소 원하는 컬럼 선택', col_list)
    df_max = car_df.loc[car_df[selected_col] == car_df[selected_col].max()]
    df_min = car_df.loc[car_df[selected_col] == car_df[selected_col].min()]

    st.text('{} 컬럼의 최대값에 해당하는 데이터 입니다.'.format(selected_col))
    st.dataframe(df_max)
    st.text('{} 컬럼의 최소값에 해당하는 데이터 입니다.'.format(selected_col))
    st.dataframe(df_min)

    # 유저가 선택한 컬럼들만, pairplot 그리고
    # 그 아래, 상관계수를 보여준다.

    column_list = st.multiselect('컬럼 선택', col_list)

    if len(column_list) > 1:
        fig1 = sb.pairplot(data = car_df[column_list])
        st.pyplot(fig1)

        st.text('선택하신 컬럼끼리의 상관계수 입니다.')
        st.dataframe(car_df[column_list].corr())
    
        fig2 = plt.figure()
        sb.heatmap(data = car_df[column_list].corr(), annot = True, fmt = '.2f', vmin = -1, vmax = 1, cmap = 'coolwarm', linewidths = 0.5)
        st.pyplot(fig2)

    ### 고객 이름 컬럼을 검색할 수 있도록 만듭니다.
    ### he 라고 넣으면, he가 이름에 들어있는 고객들의 데이터를 
    ### 가져옵니다.
    word = st.text_input('이름을 검색할 단어를 입력하세요!')
    st.dataframe(car_df.loc[car_df['Customer Name'].str.lower().str.contains(word.lower())])    