import streamlit as st
from home import face_rec

st.set_page_config(page_title='Report', layout='wide')
st.subheader('Reporting')


# retrieve logs data and show on report.py after extracting from redis list
name = 'attendance:logs'
def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list


# tabs to show info
tab1, tab2 = st.tabs(['Registered data', 'Logs'])

with tab1:
    if st.button('Refresh Data'):
        # retrieve the data from redis db
        with st.spinner('Retrieving data from redis db...'):
            redis_face_db = face_rec.retrieve_data(name='academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))
