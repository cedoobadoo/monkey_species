import os
import pandas as pd

def config_create():
    classes = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
    training_path = 'training'
    validation_path = 'validation'

    path_list = []
    class_list = []

    for cls in classes:
        class_path = os.path.join(training_path, cls)
        print(class_path)
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                filename = os.path.join(class_path, filename)
                path_list.append(filename)
                class_list.append(cls)

    data = {'paths':path_list, 'classes':class_list}
    df_training = pd.DataFrame(data)
    print(df_training)

    for cls in classes:
        class_path = os.path.join(validation_path, cls)
        print(class_path)
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                filename = os.path.join(class_path, filename)
                path_list.append(filename)
                class_list.append(cls)

    data = {'paths':path_list, 'classes':class_list}
    df_validation = pd.DataFrame(data)
    print(df_validation)

    return df_training, df_validation
