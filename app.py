from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Student Performance",
    page_icon='🔮',
)

with open('./assets/styles/style.css', 'r') as style:
    st.markdown(f'<style>{style.read()}</style>', unsafe_allow_html=True)

# HEADER AND DESCRIPTION

st.markdown("<h1 align='center' style='color: #a54aff;'>Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 align='center' style='color: #ddbbff;'>A Machine Learning Model to Predict Student Performance</h5>", unsafe_allow_html=True)

st.markdown('''<br><br><p style='text-align: justify'>
            I'm so excited to announce and deploy my first Machine Learning project using Linear Regression. 
            The aim here is to predict students performance based in the time studied. 
            For this, I'll be using the following resources: <br><br>
            ❖ <a href='https://www.kaggle.com/'> Kaggle </a>: an open source dataset repository. <br>
            ❖ <a href='https://python.org/'> Python 3 </a>: a programming language that lets us integrate systems more effectively. <br>
            ❖ <a href='https://streamlit.io/'> Streamlit </a>: a sharable webapp builder from Python data scripts. <br>
            ❖ <a href='https://plotly.com/'>Plotly </a>: a great graphing, analytics, and statistics amount of tools. <br>
            ❖ <a href='https://pandas.pydata.org/'>Pandas </a>: an open source data analysis and manipulation tool built on top of Python. <br>
            ❖ <a href='https://numpy.org/'>Numpy </a>: a fundamental package for scientific computing with Python. <br>
            ❖ <a href='https://scikit-learn.org/stable/'>Scikit-Learn </a>: a simple and efficient amount of tools for predictive data analysis. <br>
            </p></br>''', unsafe_allow_html=True)

# SETTING UP

st.markdown("<br><h3 style='color: #a54aff;'><i>Setting up</i></h3>", unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>So, without any further ado, let's jump right into our project. 
            Here, I'll be importing our libraries and getting a sample of our dataset.
            </p>''', unsafe_allow_html=True)

st.markdown('''
            <div align='center' class="alert alert-info info-link">
                <strong>INFO:</strong> Click <a href='https://www.kaggle.com/datasets/himanshunakrani/student-study-hours'> here </a> to download the data I used in this project directly from Kaggle.
            </div>
            ''', unsafe_allow_html=True)

st.code('''
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np   

uri = './assets/data/dataset.csv'
df = pd.read_csv(uri)

df.sample(5)
''', language='python')

uri = './assets/data/dataset.csv'
df = pd.read_csv(uri)

setting_up_col1, setting_up_col2, setting_up_col3 = st.columns(3)

with setting_up_col2:
    st.table(df.sample(5))

with setting_up_col3:
    st.markdown('''
    As we can see, we got two columns: <br><br>
      &nbsp;&nbsp;<code>Hours</code>: the amount of time spent studying. <br>
      &nbsp;&nbsp;<code>Scores</code>: the performance of the student. <br><br>
    A good starting point is to plot the data to see if there's any obvious trends.
    ''', unsafe_allow_html=True)

# PLOTTING CORRELAION

st.markdown("<br><h3 style='color: #a54aff;'><i>Correlation</i></h3>", unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'> As we only have two columns, let's check the correlation between them by using numpy and plotly.
            </p>''', unsafe_allow_html=True)

st.code('''
fig = px.scatter(df, x='Hours', y='Scores', color_discrete_sequence=['#a54aff'])
fig.layout.title = 'Correlation between Scores and Hours'
fig.layout.template = 'plotly_dark'
fig.show()
''')

st.markdown('''<p align='center'><i>Correlation: <strong><code>{}</code></strong></i></p>'''.format(np.corrcoef([df.Hours, df.Scores])[1,0].round(3)), unsafe_allow_html=True)

fig = px.scatter(df, x='Hours', y='Scores', color_discrete_sequence=['#a54aff'])
fig.layout.title = 'Scatter between Scores and Hours'
fig.layout.template = 'plotly_dark'
st.plotly_chart(fig)

st.write()

st.markdown('''<p style='text-align: justify'>
According to the above correlation, we can see that the amount of time spent studying has a strong correlation with the student performance, 
indicating that the amount of time spent studying is a good predictor of the performance.
</p>''', unsafe_allow_html=True)

# PREPARING THE DATA

st.markdown("<br><h3 style='color: #a54aff;'><i>Preparing the data and making a fist prediction</i></h3>", unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'> Now, let's prepare the data for our model. Here, I'll be using <code>Hour</code> as our independent variable and <code>Scores</code> as our dependent variable.
            </p>''', unsafe_allow_html=True)

st.markdown('''
            <div align='center' class="alert alert-info info-link mx-auto">
                <strong>NOTE:</strong> Here, I'm using an example with <code style='background-color: rgb(7, 62, 72, 0.15); color: darkgreen; '>6.8</code> hours of study.
            </div>
            ''', unsafe_allow_html=True)            

st.code('''
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 

X = df['Hours'].values.reshape(-1,1)
y = df['Scores'].values.reshape(-1,1)

model = KNeighborsRegressor()
model.fit(X, y)

user_hour = 6.8
print('Your predicted score is: ', model.predict(user_hour))
print('Coefficients: ', LinearRegression().fit(X, y).coef_)
''', language='python')

X = df['Hours'].values.reshape(-1,1)
y = df['Scores'].values.reshape(-1,1)

model = KNeighborsRegressor()
model.fit(X, y)

user_hour = [[6.8]]

st.write('''<p style='text-align: end;'>Your predicted score is: <code>{}</code>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
Coefficient: <code>{}</code>
</p>'''.format(
    model.predict(user_hour)[0][0],
    LinearRegression().fit(X, y).coef_[0][0].round(3)
    ), unsafe_allow_html=True)

# You can also predict

st.markdown("<br><h3 style='color: #a54aff;'><i>You can also predict ☺</i></h3>", unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'> 
            Yes, you can also predict your own score based on the amount of time you spend studying. 
            For this, please fill the form below and click <strong>Predict</strong> to get your score. 
            </p>''', unsafe_allow_html=True)

if 'count' not in st.session_state:
    st.session_state.user_name = None
    st.session_state.user_hours = None
    

def update_results(name, hours):
    st.session_state.user_name = name
    st.session_state.user_hours = hours

with st.form(key='my_form'):
    predict_col1, predict_col2 = st.columns(2)

    with predict_col1:
        st.session_state.user_name = st.text_input(label='What is your name?', key='update_name', placeholder='Enter your name') 

    with predict_col2:
        st.session_state.user_hours = st.text_input('Hours spent studying', key='update_hours')
    
    submit = st.form_submit_button(label='Predict', on_click=update_results, args=(st.session_state.user_name, st.session_state.user_hours))


if submit:
    if st.session_state.user_name is not None or st.session_state.user_hours is not None:
        try:
            st.markdown('Hello, <strong>{}</strong>! Thanks for using this performance predictor.'.format(st.session_state.user_name), unsafe_allow_html=True)
            st.write('As you have spent <strong style=\'color: #ddbbff;\'>{}</strong> hours studying, your predicted score is: <strong style=\'color: #ddbbff;\'>{}</strong>.'.format(st.session_state.user_hours, model.predict([[float(st.session_state.user_hours)]])[0][0].round(3)), unsafe_allow_html=True)

            if model.predict([[float(st.session_state.user_hours)]])[0][0].round(3) < 50:
                st.markdown('''
                <div align='center' class="alert alert-danger info-link mx-auto">
                    😕 <strong>WARNING:</strong> Your predicted score is less than 50. I highly recommend you to study more.
                </div>
                ''', unsafe_allow_html=True)  
            elif 50 < model.predict([[float(st.session_state.user_hours)]])[0][0].round(3) < 70:
                st.markdown('''
                <div align='center' class="alert alert-info info-link mx-auto">
                    🤓 <strong>NOTE:</strong> Your predicted score is less than 70. I'm pretty sure you can make it better.
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div align='center' class="alert alert-success info-link mx-auto">
                    🤩 <strong>SUCCESS:</strong> Your predicted score is above 70. I'm sure you'll get very qualified.
                </div>
                ''', unsafe_allow_html=True)

        except ValueError as e:
            st.write('❌ In order to predict your score, please enter a valid number.')
    else:
        st.write('Please fill the form above')