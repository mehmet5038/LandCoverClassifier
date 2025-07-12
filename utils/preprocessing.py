from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def get_data_generators(data_directory, image_size=(256, 256), batch_size=16, validation_split=0.15):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.1,
        height_shift_range=0.1,
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

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    ax1.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    ax1.set_title("Model Doğruluğu")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Doğruluk")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history["loss"], label="Eğitim Kaybı")
    ax2.plot(history.history["val_loss"], label="Doğrulama Kaybı")
    ax2.set_title("Model Kaybı")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Kayıp")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()