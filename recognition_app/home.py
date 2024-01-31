import streamlit as st

st.set_page_config(page_title='Rec system', layout='wide')
st.header('Recognition system')

with st.spinner('Loading model and connecting to db...'):
    import face_rec

st.success('Model and db loaded successfully')