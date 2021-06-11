# Train the recommender model using a Gradient Workflow
#
# This script is minimally commented because it is called from and explained by
# the recommender project notebook, deep_learning_recommender_tf.ipynb
#
# Code is the minimal subset of the notebook to perform the steps
#
# Lines are the same as the notebook, except where they need to be changed to
# work in a .py script as opposed to a .ipynb notebook
#
# Last updated: Jun 11th 2021
#
# TODO
#
# Supply hyperparams as env: defining env var HP_FINAL_EPOCHS and HP_FINAL_LR in
# workflow-train-model.yaml hits PLA-278. Currently hardwire the params here.

# Setup

import subprocess
subprocess.run('pip install --upgrade pip', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run('pip install -q tensorflow-recommenders==0.4.0', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run('pip install -q --upgrade tensorflow-datasets==4.2.0', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

import os
import platform
import pprint
import tempfile

from typing import Dict, Text

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Get model hyperparameters

### TMP: Hardwire values; supply params as env vars when PLA-278 is resolved ###

#hp_final_epochs = int(os.environ.get('HP_FINAL_EPOCHS'))
#hp_final_lr = float(os.environ.get('HP_FINAL_LR'))
hp_final_epochs = 50
hp_final_lr = 0.1

# Data preparation

ratings_raw = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings_raw.map(lambda x: {
    "movie_title": x["movie_title"],
    "timestamp": x["timestamp"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_time = timestamps.max()
min_time = timestamps.min()

sixtieth_percentile = min_time + 0.6*(max_time - min_time)
eightieth_percentile = min_time + 0.8*(max_time - min_time)

train =      ratings.filter(lambda x: x["timestamp"] <= sixtieth_percentile)
validation = ratings.filter(lambda x: x["timestamp"] > sixtieth_percentile and x["timestamp"] <= eightieth_percentile)
test =       ratings.filter(lambda x: x["timestamp"] > eightieth_percentile)

ntimes_tr = 0
ntimes_va = 0
ntimes_te = 0

for x in train.take(-1).as_numpy_iterator():
  ntimes_tr += 1

for x in validation.take(-1).as_numpy_iterator():
  ntimes_va += 1

for x in test.take(-1).as_numpy_iterator():
  ntimes_te += 1

print("Number of rows in training set = {}".format(ntimes_tr))
print("Number of rows in validation set = {}".format(ntimes_va))
print("Number of rows in testing set = {}".format(ntimes_te))
print("Total number of rows = {}".format(ntimes_tr+ntimes_va+ntimes_te))

train = train.shuffle(ntimes_tr)
validation = validation.shuffle(ntimes_va)
test = test.shuffle(ntimes_te)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

cached_train = train.shuffle(ntimes_tr).batch(8192).cache()
cached_validation = validation.shuffle(ntimes_va).batch(8192).cache()
cached_test = test.batch(4096).cache()

# Define model

class MovielensModelTunedRanking(tfrs.models.Model):

    def __init__(self) -> None:
        super().__init__()
        embedding_dimension = 32

        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1)
        ])

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        return (
            user_embeddings,
            movie_embeddings,
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("user_rating")
        user_embeddings, movie_embeddings, rating_predictions = self(features)

        rating_loss = self.task(
            labels=ratings,
            predictions=rating_predictions,
        )

        return rating_loss

# Train model

model_tr = MovielensModelTunedRanking()
model_tr.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=hp_final_lr))
history_tr = model_tr.fit(cached_train, epochs=hp_final_epochs, validation_data=cached_validation)

rmse_tr = history_tr.history["root_mean_squared_error"][-1]
print(f"Root mean squared error in user rating from training: {rmse_tr:.2f}")

val_rmse_tr = history_tr.history["val_root_mean_squared_error"][-1]
print(f"Root mean squared error in user rating from validation: {val_rmse_tr:.2f}")

# Evaluate model

eval_tr = model_tr.evaluate(cached_test, return_dict=True)

rmse_eval_tr = eval_tr["root_mean_squared_error"]
print(f"Root mean squared error in user rating from evaluation: {rmse_eval_tr:.2f}")

# Model on testing set

model_tr.predict(cached_test)

# Save model

export_path = '/outputs/trainedRecommender'
print('export_path = {}\n'.format(export_path))

model_tr.save(export_path)

print('Done')
