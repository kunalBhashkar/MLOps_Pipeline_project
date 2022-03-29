
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflow_transform as tft

from absl import logging
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from typing import List, Text

import constants

LABEL = constants.LABEL
BATCH_SIZE = 32
EPOCHS = 50

def _input_fn(
    file_pattern: List[Text],
    data_accessor: tfx.components.DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int,
) -> tf.data.Dataset:

    """
    Generates a dataset of features that can be used to train
    and evaluate the model.

    Args:
        file_pattern: List of paths or patterns of input data files.
        data_accessor: An instance of DataAccessor that we can use to
            convert the input to a RecordBatch.
        tf_transform_output: The transformation output.
        batch_size: The number of consecutive elements that we should
            combine in a single batch.

    Returns:
        A dataset that contains a tuple of (features, indices) where 
            features is a dictionary of Tensors, and indices is a single
            Tensor of label indices.
    """

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema=tf_transform_output.raw_metadata.schema,
    )

    tft_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        transformed_features = tft_layer(raw_features)
        transformed_label = transformed_features.pop(LABEL)
        return transformed_features, transformed_label
    
    return dataset.map(apply_transform).repeat()

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses a serialized tf.Example and applies
    the transformations during inference.
    Args:
        model: The model that we are serving.
        tf_transform_output: The transformation output that we want to 
            include with the model.
    """
    
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k != LABEL
        }

        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            required_feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn

def _model() -> tf.keras.Model:
    inputs = [
              layers.Input(shape=(1,), name="Age"),
              layers.Input(shape=(1,), name="EstimatedSalary"),
              layers.Input(shape=(1,), name="Gender")
    ]

    x = layers.concatenate(inputs)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(1e-2),
        loss="binary_crossentropy",
        metrics=[metrics.BinaryAccuracy()],
    )

    model.summary(print_fn=logging.info)
    return model

def run_fn(fn_args: tfx.components.FnArgs):
    """
    The callback function that will be called by the Trainer component
    to train the model using the suplied arguments.

    Args:
        fn_args: A collection of name/value pairs representing the 
            arguments to train the model.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=BATCH_SIZE,
    )

    model = _model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS
    )

    # We need to modify the default signature to include the transform layer in 
    # the computational graph.
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
