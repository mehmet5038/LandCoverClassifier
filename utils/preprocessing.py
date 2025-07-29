from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_directory, image_size=(256, 256), batch_size=16, isSkyView=True, validation_split=0.15):
    if isSkyView:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.3,
            shear_range=10,
            width_shift_range=0.15,
            height_shift_range=0.15,
            channel_shift_range=30,
            brightness_range=[0.6, 1.4],
            horizontal_flip=True,
            validation_split=0.15
        )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        data_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen