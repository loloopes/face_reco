import time

import av
from home import face_rec, st
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title='Predictions', layout='wide')
st.subheader('Real-Time recognition system')


# retrieve the data from redis db
with st.spinner('Retrieving data from redis db...'):
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success('Data succesfully retrieved from redis!')

# time
wait_time = 30
set_time = time.time()
realtime_pred = face_rec.RealTimePred()

# real time prediction
def video_frame_callback(frame):
    global set_time

    img = frame.to_ndarray(format='bgr24')
    pred_img = realtime_pred.face_prediction(img, redis_face_db,
                                        'facial_features',
                                        ['Name', 'Role'],
                                        thresh=0.5)
    
    time_now = time.time()
    diff_time = time_now - set_time
    if diff_time >= wait_time:
        realtime_pred.save_logs_redis()
        set_time = time.time()
        print('save data to redis db')

    return av.VideoFrame.from_ndarray(pred_img, format='bgr24')

webrtc_streamer(key='realtime_prediction', video_frame_callback=video_frame_callback)