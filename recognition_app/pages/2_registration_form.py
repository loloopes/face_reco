import av
import cv2
import numpy as np
import streamlit as st
from home import face_rec
from streamlit_webrtc import webrtc_streamer

# st.set_page_config(page_title='Registration form', layout='wide')
st.subheader('Registration form')

# init registration form
registration_form = face_rec.RegistrationForm()

# step-1: collect person name and role in a form
person_name = st.text_input(label='Name', placeholder='First and last name')
role = st.selectbox(label='Select role', options=('Student', 'Teacher'))

# step-2: collect person facial embeddings
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')
    reg_img, embedding = registration_form.get_embedding(img)

    # step-1: save data into local computer as txt
    if embedding is not None:
        with open('face_embedding.txt', mode = 'ab') as f:
            np.savetxt(f, embedding)

    # step-2: save data into redis 

    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
                    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# ste-3: save data in redis db
if st.button('Submit'):
    returned_val = registration_form.save_data_in_db(person_name, role)
    if returned_val == True:
        st.success(f'{person_name} registered succesfully!')

    elif returned_val == 'invalid name':
        st.error('Invalid name, it cannot be empty')

    elif returned_val == 'invalid file':
        st.error('face_embedding.txt was not found, refresh the page and try again.')