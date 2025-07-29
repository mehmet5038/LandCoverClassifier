import matplotlib.pyplot as plt


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