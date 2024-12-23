import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

from data import load_datasets

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def build_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_mobilenetv2_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def hyperparameter_tuning(train_ds, val_ds):
    models = {
        'VGG16': build_vgg16_model,
        'ResNet50': build_resnet50_model,
        'MobileNetV2': build_mobilenetv2_model
    }

    results = []
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    num_classes = len(train_ds.class_names)

    for name, model_fn in models.items():
        model = model_fn(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
        val_accuracy = max(history.history['val_accuracy'])

        results.append({'Model': name, 'Val Accuracy': val_accuracy})

    print("\nModel Performance Before Tuning")
    for result in results:
        print(result)

    # Add hyperparameter tuning logic (e.g., varying learning rates, optimizers)
    tuned_results = []

    for name, model_fn in models.items():
        model = model_fn(input_shape, num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
        val_accuracy = max(history.history['val_accuracy'])

        tuned_results.append({'Model': name, 'Val Accuracy': val_accuracy})

    print("\nModel Performance After Tuning")
    for result in tuned_results:
        print(result)

if __name__ == "__main__":
    data_dir = '/Users/samuelsetsofia/dev/projects/EyeDisease_Classifier_DeepLearning/dataset'
    output_dir = '/Users/samuelsetsofia/dev/projects/EyeDisease_Classifier_DeepLearning/output_dir'
    classes = ['cataract', 'glaucoma', 'diabetic_retinopathy', 'normal']

    # Assume datasets are already split; load them
    train_ds, val_ds, test_ds = load_datasets(output_dir, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    # Run hyperparameter tuning
    hyperparameter_tuning(train_ds, val_ds)
