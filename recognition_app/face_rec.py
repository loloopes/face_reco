import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# connect to redis client
r = redis.StrictRedis(
    host="redis-11755.c308.sa-east-1-1.ec2.cloud.redislabs.com", 
    port=11755,
    username="default",
    password="CzdAM8ReaYrKSySrvShim7abK4ZENraC",
)

# retrieve data from db
def retrieve_data(name):
    retrieved_dict = r.hgetall(name)
    retrieved_series = pd.Series(retrieved_dict)
    retrieved_series = retrieved_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieved_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieved_series.index = index
    retrieved_df = retrieved_series.to_frame().reset_index()
    retrieved_df.columns = ['name_role', 'facial_features']
    retrieved_df[['Name', 'Role']] = retrieved_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieved_df[['Name', 'Role', 'facial_features']]

# config face analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                       root='models',
                       providers=['CUDAExecutionProvider', 'CUDAExecutionProvider'])
faceapp.prepare(ctx_id=0,
                det_size=(640,640),
                det_thresh=0.5)


# ml search algorithm
def ml_search_algorithm(df,
                        feature_column,
                        test_vector,
                        name_role=['Name', 'Role'],
                        thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the df (data collection)
    dataframe = df.copy()
    # step-2: index face embedding from df and covert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # step3: calculate cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


# real time prediction, saving logs every 1 minute
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])


    def reset_logs(self):
        self.logs = dict(name=[], role=[], current_time=[])

    
    def save_logs_redis(self):
        # step-1: create a logs df
        data_frame = pd.DataFrame(self.logs)

        # step-2: drop duplicate information
        data_frame.drop_duplicates('name', inplace=True)

        # step-3: push data to redis database
        # encode the data
        name_list = data_frame['name'].tolist()
        role_list = data_frame['role'].tolist()
        ctime_list = data_frame['current_time'].tolist()
        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f'{name}@{role}@{ctime}'
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)
        
        self.reset_logs()



    def face_prediction(self,
                        test_image,
                        df,
                        feature_column,
                        name_role=['Name', 'Role'],
                        thresh=0.5):
        # step-0: find the time
        current_time = str(datetime.now())

        # step-1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        # step-2: use for loop to extract each embedding to feed ml_search_algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(df,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(test_copy, (x1,y1), (x2,y2), color)
            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            cv2.putText(test_copy, current_time, (x1,y2+10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        return test_copy
    


# Registration form
class RegistrationForm:
    def __init__(self):
        self.samples = 0


    def reset(self):
        self.samples = 0


    def get_embedding(self, frame):
        # get results from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.samples += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

            # add text samples info
            text = f'samples = {self.samples}'
            cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,0), 2)

            #facial embeddings
            embeddings = res['embedding']

        return frame, embeddings
    

    def save_data_in_db(self, name, role):
        # validate name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'invalid name'
        else:
            return 'invalid name'
        
        # validate face_embedding.txt 
        if 'face_embedding.txt' not in os.listdir():
            return 'invalid file'

        # step-1: load face-embedding.txt
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        # step-2: convert into array (proper shape)
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # step-3: calculate mean embedding
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # step-4: save into redis db
        r.hset(name='academy:register', key=key, value=x_mean_bytes)

        os.remove('face_embedding.txt')
        self.reset()

        return True