# from joblib import load
# from src.utils.utils import load_object
# from src.constants import MODEL_FOLDER, PREPROCESSOR_FILE, MODEL_FILE
# import os
# from src.exception import CustomException
# import sys
# from pydantic import BaseModel, ValidationError
# from typing import Literal

# class Data(BaseModel):
#     carat: float
#     cut: Literal['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
#     color: Literal['D', 'E', 'F', 'G', 'H', 'I', 'J']
#     clarity: Literal['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
#     depth: float
#     table: float
#     x: float
#     y: float
#     z: float


    
# def datafun(data: Data):
#     print(data)  
# # try:

# #     data = {
# #         'carat': 1.52,
# #         'cut': 'Premium', 
# #         'color': 'F',
# #         'clarity': 'VSh2',
# #         'depth': 62.2, 
# #         'table': 58.0,
# #         'x': 7.27,
# #         'y': 7.33,
# #         'z': 4.55
# #     }
# #     datafun(data)
# # except Exception as e:
# #     raise CustomException(e, sys)

# # model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
# # preprocessor_path = os.path.join(MODEL_FOLDER, PREPROCESSOR_FILE)
# # print(model_path)

# # preprocessor = load(filename=preprocessor_path)
# # print(preprocessor)
# # model = load_object(model_path)


# import pandas as pd
# data = {
#     'carat': 1.52,
#     'cut': 'Premium', 
#     'color': 'F',
#     'clarity': 'VSd2',
#     'depth': '62.2', 
#     'table': 58.0,
#     'x': 7.27,
#     'y': 7.33,
#     'z': 4.55
# }
# try:
#     datafun(data)
# except ValidationError as e:
#     raise CustomException(e, sys)



# # data = pd.DataFrame([data])
# # print(type(data))

# # transform_data = preprocessor.transform(data)
# # print(transform_data)
# # transform_data = pd.DataFrame(transform_data, columns=preprocessor.get_feature_names_out())
# # print(transform_data)
# # predicted_value = model.predict(transform_data)
# # print("The predicted value is",round(predicted_value[0],2))



from src.exception import CustomException
from pydantic import BaseModel, ValidationError, StrictFloat
from typing import Literal

class User(BaseModel):
    carat: StrictFloat
    cut: Literal['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color: Literal['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity: Literal['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    depth: StrictFloat
    table: StrictFloat
    x: StrictFloat
    y: StrictFloat
    z: StrictFloat

def datafun(user: User):
    print(user)

data = {
    'carat': 1.52,
    'cut': 'Premium', 
    'color': 'F',
    'clarity': 'VS2', 
    'depth': 62.2,   
    'table': 58,
    'x': 9,
    'y': 7.33,
    'z': 4.55
}

try:
    User.model_validate(data)
    # user_instance = User.model_validate(data)  # Validate and create an instance of User
    # datafun(user_instance)
except ValidationError as e:
    print(e)



"""
demo.py file can be deleted
this is just experimenation of the different different modules
"""