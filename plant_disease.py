import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a plant disease classifier with MobileNetV2 only."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root with class folders directly inside. Use this if you do not have separate train/validate folders.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Training directory with class folders inside.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation directory with class folders inside.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_outputs",
        help="Directory to save models, logs, and class names.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--initial-epochs",
        type=int,
        default=12,
        help="Epochs with frozen MobileNetV2 base.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=18,
        help="Epochs after unfreezing the top MobileNetV2 layers.",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Unfreeze MobileNetV2 layers from this index onward.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for stage 1 training.",
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for stage 2 fine-tuning.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split used with image_dataset_from_directory.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout before the classifier head.",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=256,
        help="Dense units in the classification head.",
    )
    return parser.parse_args()


def build_datasets_from_single_dir(data_dir: str, img_size: int, batch_size: int, validation_split: float):
    common_args = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        validation_split=validation_split,
        seed=SEED,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_args,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common_args,
    )

    class_names = train_ds.class_names

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    def prepare_train(images, labels):
        images = tf.cast(images, tf.float32)
        images = data_augmentation(images, training=True)
        return preprocess_input(images), labels

    def prepare_eval(images, labels):
        images = tf.cast(images, tf.float32)
        return preprocess_input(images), labels

    train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def build_datasets_from_train_val_dirs(train_dir: str, val_dir: str, img_size: int, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False,
    )

    train_classes = train_ds.class_names
    val_classes = val_ds.class_names
    if train_classes != val_classes:
        raise ValueError(
            "Train and validation folders do not contain the same classes in the same order."
        )

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    def prepare_train(images, labels):
        images = tf.cast(images, tf.float32)
        images = data_augmentation(images, training=True)
        return preprocess_input(images), labels

    def prepare_eval(images, labels):
        images = tf.cast(images, tf.float32)
        return preprocess_input(images), labels

    train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, train_classes


def get_class_weights(data_dir: str, class_names):
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    y = []

    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        image_count = sum(
            1
            for item in class_dir.iterdir()
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        y.extend([class_to_index[class_name]] * image_count)

    if not y:
        return None

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y))
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def resolve_dataset_sources(args):
    if args.train_dir and args.val_dir:
        train_ds, val_ds, class_names = build_datasets_from_train_val_dirs(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
        class_weights = get_class_weights(args.train_dir, class_names)
        return train_ds, val_ds, class_names, class_weights

    if args.data_dir:
        train_ds, val_ds, class_names = build_datasets_from_single_dir(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
        )
        class_weights = get_class_weights(args.data_dir, class_names)
        return train_ds, val_ds, class_names, class_weights

    raise ValueError("Provide either --data-dir or both --train-dir and --val-dir.")


def build_model(num_classes: int, img_size: int, dropout: float, dense_units: int):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def save_class_names(class_names, output_dir: str):
    class_names_path = Path(output_dir) / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def steps_per_epoch(dataset):
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    if cardinality < 0:
        return None
    return cardinality


def main():
    args = parse_args()
    set_seed()
    ensure_dir(args.output_dir)

    train_ds, val_ds, class_names, class_weights = resolve_dataset_sources(args)
    save_class_names(class_names, args.output_dir)
    model, base_model = build_model(
        num_classes=len(class_names),
        img_size=args.img_size,
        dropout=args.dropout,
        dense_units=args.dense_units,
    )

    compile_model(model, args.learning_rate)

    callbacks_stage1 = [
        ModelCheckpoint(
            filepath=str(Path(args.output_dir) / "best_stage1.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(str(Path(args.output_dir) / "training_log.csv"), append=False),
    ]

    print(f"Classes found: {class_names}")
    print(f"Class weights: {class_weights}")
    print(f"Training samples per epoch: {steps_per_epoch(train_ds)}")
    print(f"Validation samples per epoch: {steps_per_epoch(val_ds)}")

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    base_model.trainable = True
    for layer in base_model.layers[:args.fine_tune_at]:
        layer.trainable = False

    compile_model(model, args.fine_tune_learning_rate)

    callbacks_stage2 = [
        ModelCheckpoint(
            filepath=str(Path(args.output_dir) / "best_model.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(Path(args.output_dir) / "fine_tune_log.csv"), append=False),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_stage1.history["loss"]),
        epochs=args.initial_epochs + args.fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    val_loss, val_accuracy, val_top3 = model.evaluate(val_ds, verbose=1)
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final validation top-3 accuracy: {val_top3 * 100:.2f}%")

    final_model_path = Path(args.output_dir) / "plant_disease_mobilenetv2.h5"
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")


if __name__ == "__main__":
    main()
