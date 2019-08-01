from config_create import config_create
from model import model
import keras
import math

def train(batch_size, epochs):
    df_training, df_validation = config_create()
    arc = model(10)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    arc.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    flow_training = datagen.flow_from_dataframe(df_training, x_col='paths', y_col='classes', target_size=(156, 156), batch_size=batch_size)
    flow_validation = datagen.flow_from_dataframe(df_validation, x_col='paths', y_col='classes', target_size=(156, 156), batch_size=batch_size)

    arc.fit_generator(flow_training, epochs=epochs, validation_data=flow_validation,
                      steps_per_epoch=math.ceil(len(flow_validation) / batch_size),
                      validation_steps=math.ceil(len(flow_validation)/ batch_size))

    arc.save('model.h5')

train(8, 10)

