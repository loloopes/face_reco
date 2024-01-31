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


# config face analysis
faceapp = FaceAnalysis(name='buffalo_l',
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


def face_prediction(test_image,
                    df,
                    feature_column,
                    name_role=['Name', 'Role'],
                    thresh=0.5):
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

    return test_copy