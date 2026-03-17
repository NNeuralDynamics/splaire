import os
import sys
import time
import argparse
import random
import time
import pickle
import math
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from splaire import Splaire
import wandb
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbEvalCallback,
    WandbModelCheckpoint,
)


parser = argparse.ArgumentParser(
    description="train Splaire with w&b and tensorboard profiling (single GPU)"
)
parser.add_argument('--gpu', type=str, required=True,
                    help="GPU id for CUDA_VISIBLE_DEVICES")
parser.add_argument('-c','--context', type=int,
                    choices=[80,400,2000,10000], required=True)
parser.add_argument('-n','--name', type=str, required=True)
parser.add_argument('-d','--train_dataset', type=str, required=True)
parser.add_argument('-v','--valid_dataset', type=str, required=True)
parser.add_argument('--mode', choices=['classification','regression','both'], required=True)
parser.add_argument('--reset-training-tracking', action='store_true')
parser.add_argument('--mask_classification', action='store_true',
                    help="Mask classification loss where y_true[...,3]==777")
parser.add_argument('--mask_regression', action='store_true',
                    help="Mask regression loss where y_true[...,3]==777")
args = parser.parse_args()

# GPU setup
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


seed = int(time.time() * 1e6) % (2**32 - 1)  
print(f"Using random seed: {seed}")

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ.update({
    'TF_DETERMINISTIC_OPS': '1',
    'TF_CUDNN_DETERMINISTIC': '1'
})
print("Using GPU:", args.gpu)


# model directory setup
MODEL_DIR  = f'./Models/{args.name}'
os.makedirs(MODEL_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(MODEL_DIR, f"{args.name}_checkpoint.keras")
BEST_PATH  = os.path.join(MODEL_DIR, f"{args.name}_best.keras")
STATE_PATH = os.path.join(MODEL_DIR, f"{args.name}_state.pkl")
LR_PATH    = os.path.join(MODEL_DIR, f"{args.name}_lr.npy")

# context-specific hyperparameters
if args.context == 80:
    W, AR, BATCH = np.array([11]*4), np.array([1]*4), 18*28
elif args.context == 400:
    W, AR, BATCH = np.array([11]*8), np.array([1]*4 + [4]*4), 18*28
elif args.context == 2000:
    W, AR, BATCH = np.array([11]*8 + [21]*4), np.array([1]*4 + [4]*4 + [10]*4), 12*28
else:
    W, AR, BATCH = np.array([11]*8 + [21]*4 + [41]*4), np.array([1]*4 + [4]*4 + [10]*4 + [25]*4), 128
assert 2 * np.sum(AR * (W - 1)) == args.context, 'Context mismatch'
print(f"Context={args.context}, batch_size={BATCH}")

rf = int(np.sum(AR * (W - 1)))        
seq_len      = rf * 3                   
target_len   = rf                       

print(f"context={args.context}, receptive_field={rf}, "
      f"input_seq_len={seq_len}, target_len={target_len}, batch={BATCH}")

# count sequences helper
def count_seqs(path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        num_keys = 3 if any(k.startswith('GC') for k in keys) else 2
        nchunks = len(keys) // num_keys
        total = sum(f[f'X{i}'].shape[0] for i in range(nchunks))
    return total

# TensorBoard callback
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f'logs/{args.name}',
    histogram_freq=1,
    update_freq='batch',
    profile_batch=(600,1000)
)

# w&b init
wandb.tensorboard.patch(root_logdir=f"logs/{args.name}")
wandb.init(
    project='spliceDCNN',
    name=args.name,
    sync_tensorboard=True,
    config={
        'context': args.context,
        'mode': args.mode,
        'batch_size': BATCH,
        'mask_cls': args.mask_classification,
        'mask_reg': args.mask_regression,
        'initial_lr': 1e-4,
        'gpu': args.gpu,
    }
)


def masked_binary_crossentropy(y_reg, y_pred):

    labels = tf.squeeze(y_reg, axis=-1)
    mask   = tf.cast(tf.not_equal(labels, 777.0), tf.float32)
    logits = tf.squeeze(y_pred, axis=-1)
    
    bce    = tf.nn.sigmoid_cross_entropy_with_logits(
                 labels=tf.cast(labels, tf.float32),
                 logits=logits
             )
    
    masked_bce = bce * mask
    return tf.reduce_sum(masked_bce) / (tf.reduce_sum(mask) + 1e-6)


def masked_mae_metric(y_reg, y_pred):
    labels = y_reg[..., 0]
    mask   = tf.not_equal(labels, 777.0)
    y_pr   = tf.sigmoid(tf.squeeze(y_pred, axis=-1))

    return tf.reduce_mean(
        tf.abs(tf.boolean_mask(labels, mask) -
               tf.boolean_mask(y_pr,    mask))
    )


def masked_mse_metric(y_reg, y_pred):
    labels = y_reg[..., 0]
    mask   = tf.not_equal(labels, 777.0)
    y_pr   = tf.sigmoid(tf.squeeze(y_pred, axis=-1))

    return tf.reduce_mean(
        tf.square(tf.boolean_mask(labels, mask) -
                  tf.boolean_mask(y_pr,    mask))
    )


def masked_r2_metric(y_reg, y_pred):
    labels = y_reg[..., 0]
    mask   = tf.not_equal(labels, 777.0)
    y_pr   = tf.sigmoid(tf.squeeze(y_pred, axis=-1))
    y_true = tf.boolean_mask(labels, mask)
    y_pred = tf.boolean_mask(y_pr, mask)
    ss_res = tf.reduce_sum((y_true - y_pred) ** 2)
    ss_tot = tf.reduce_sum((y_true - tf.reduce_mean(y_true)) ** 2)

    return tf.cond(
        ss_tot > 0,
        lambda: 1 - ss_res / ss_tot,
        lambda: tf.constant(0.0)
    )

model = Splaire(L=32, W=W, AR=AR, dropout_rate=0.2)

# compile with additional metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        'classification_output': tf.keras.losses.CategoricalCrossentropy(),
        'regression_output':    masked_binary_crossentropy,
    },
    loss_weights={
        'classification_output': 1.0,
        'regression_output':    1.0,
    },
    metrics={
        'classification_output': [
            tf.keras.metrics.AUC(curve='PR', name='auprc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
        'regression_output': [
            masked_mae_metric,
            masked_mse_metric,
            masked_r2_metric
        ]
    }
)


num_params = model.count_params()
wandb.config.update({'num_parameters': num_params})
print(f'model has {num_params:,} parameters')
model.summary()

n_train = count_seqs(args.train_dataset)
steps_train = math.ceil(n_train / BATCH)
n_val = count_seqs(args.valid_dataset)
steps_val = math.ceil(n_val / BATCH)
mini_steps = math.ceil(steps_train / 10)
print(f"train steps: {steps_train}, mini_steps: {mini_steps}, validation steps: {steps_val}")


def seq_generator(path, buffer_size=5):
    with h5py.File(path, 'r') as f:
        keys     = list(f.keys())
        num_keys = 3 if any(k.startswith('GC') for k in keys) else 2
        nchunks  = len(keys) // num_keys

        chunk_ids = list(range(nchunks))
        random.shuffle(chunk_ids)

        while True:
            for start in range(0, nchunks, buffer_size):
                buf_chunks = chunk_ids[start:start + buffer_size]
                # load those chunks into memory
                X_buf = [f[f'X{c}'][:] for c in buf_chunks]
                Y_buf = [f[f'Y{c}'][:] for c in buf_chunks]
                order = [(bi, i)
                         for bi, arr in enumerate(X_buf)
                         for i in range(arr.shape[0])]
                random.shuffle(order)

                for buf_idx, seq_i in order:
                    yield X_buf[buf_idx][seq_i], Y_buf[buf_idx][seq_i]
                    
            random.shuffle(chunk_ids)

def to_multioutput(x, y_full):
    y_cls = y_full[..., :3]  
    y_reg = y_full[..., 3:4]
    
    return x, {
        'classification_output': y_cls,
        'regression_output':    y_reg
    }




AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset
      .from_generator(
         lambda: seq_generator(args.train_dataset),
         output_signature=(
            tf.TensorSpec((seq_len, 4), tf.float32),
            tf.TensorSpec((target_len, 4), tf.float32),
         )
      )
      .map(to_multioutput, num_parallel_calls=AUTOTUNE)
      .batch(BATCH)
      .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset
      .from_generator(
         lambda: seq_generator(args.valid_dataset),
         output_signature=(
            tf.TensorSpec((seq_len, 4), tf.float32),
            tf.TensorSpec((target_len, 4), tf.float32),
         )
      )
      .map(to_multioutput, num_parallel_calls=AUTOTUNE)
      .batch(BATCH)
      .prefetch(AUTOTUNE)
)




callbacks = [
    tb_callback,
    WandbMetricsLogger(log_freq= 'batch'),
    tf.keras.callbacks.ModelCheckpoint(CKPT_PATH, save_best_only=False),
    tf.keras.callbacks.ModelCheckpoint(BEST_PATH, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda ep, logs: (
        pickle.dump({'start_epoch': ep+1, 'best_val_loss': logs['val_loss']}, open(STATE_PATH,'wb')),
        np.save(LR_PATH, K.get_value(model.optimizer.learning_rate)
)
    )),
]



history = model.fit(
    train_ds,
    epochs= 100,
    steps_per_epoch= mini_steps,
    validation_data=valid_ds,
    validation_steps= steps_val,
    callbacks=callbacks,
    initial_epoch=0,
    verbose=1
)


wandb.finish()

print('all done')