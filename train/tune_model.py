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
    description="Fine-tune single head of pre-trained Splaire model"
)
parser.add_argument('--gpu', type=str, required=True,
                    help="GPU id for CUDA_VISIBLE_DEVICES")
parser.add_argument('-c','--context', type=int,
                    choices=[80,400,2000,10000], required=True)
parser.add_argument('-n','--name', type=str, required=True,
                    help="Name for the new fine-tuned model")
parser.add_argument('--pretrained_model', type=str, required=True,
                    help="Path to the pre-trained model (.keras file)")
parser.add_argument('-d','--train_dataset', type=str, required=True)
parser.add_argument('-v','--valid_dataset', type=str, required=True)
parser.add_argument('--head', choices=['classification','regression'], required=True,
                    help="Which head to fine-tune")
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help="Learning rate for fine-tuning (default: 1e-5)")
parser.add_argument('--mask_classification', action='store_true',
                    help="Mask classification loss where y_true[...,3]==777")
parser.add_argument('--mask_regression', action='store_true',
                    help="Mask regression loss where y_true[...,3]==777")
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

seed = int(time.time() * 1e6) % (2**32 - 1)  
print(f"using random seed: {seed}")

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ.update({
    'TF_DETERMINISTIC_OPS': '1',
    'TF_CUDNN_DETERMINISTIC': '1'
})
print("using gpu:", args.gpu)

MODEL_DIR  = f'./Models/{args.name}'
os.makedirs(MODEL_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(MODEL_DIR, f"{args.name}_checkpoint.keras")
BEST_PATH  = os.path.join(MODEL_DIR, f"{args.name}_best.keras")
STATE_PATH = os.path.join(MODEL_DIR, f"{args.name}_state.pkl")
LR_PATH    = os.path.join(MODEL_DIR, f"{args.name}_lr.npy")

if args.context == 80:
    W, AR, BATCH = np.array([11]*4), np.array([1]*4), 18*28
elif args.context == 400:
    W, AR, BATCH = np.array([11]*8), np.array([1]*4 + [4]*4), 18*28
elif args.context == 2000:
    W, AR, BATCH = np.array([11]*8 + [21]*4), np.array([1]*4 + [4]*4 + [10]*4), 12*28
else:
    W, AR, BATCH = np.array([11]*8 + [21]*4 + [41]*4), np.array([1]*4 + [4]*4 + [10]*4 + [25]*4), 128
assert 2 * np.sum(AR * (W - 1)) == args.context, 'context mismatch'
print(f"context={args.context}, batch_size={BATCH}")

def count_seqs(path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        num_keys = 3 if any(k.startswith('GC') for k in keys) else 2
        nchunks = len(keys) // num_keys
        total = sum(f[f'X{i}'].shape[0] for i in range(nchunks))
    return total

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f'logs/{args.name}',
    histogram_freq=1,
    update_freq='batch',
    profile_batch=(600,1000)
)

wandb.tensorboard.patch(root_logdir=f"logs/{args.name}")
wandb.init(
    project='spliceDCNN',
    name=args.name,
    sync_tensorboard=True,
    config={
        'context': args.context,
        'head': args.head,
        'batch_size': BATCH,
        'mask_cls': args.mask_classification,
        'mask_reg': args.mask_regression,
        'learning_rate': args.learning_rate,
        'pretrained_model': args.pretrained_model,
        'gpu': args.gpu,
        'training_type': 'finetune'
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
    y_pred = tf.boolean_mask(y_pr,    mask)
    ss_res = tf.reduce_sum((y_true - y_pred) ** 2)
    ss_tot = tf.reduce_sum((y_true - tf.reduce_mean(y_true)) ** 2)
    return tf.cond(
        ss_tot > 0,
        lambda: 1 - ss_res / ss_tot,
        lambda: tf.constant(0.0)
    )

print(f"loading pre-trained model from: {args.pretrained_model}")
model = tf.keras.models.load_model(
    args.pretrained_model,
    custom_objects={
        'masked_binary_crossentropy': masked_binary_crossentropy,
        'masked_mae_metric': masked_mae_metric,
        'masked_mse_metric': masked_mse_metric,
        'masked_r2_metric': masked_r2_metric
    }
)

print("model loaded")
print(f"model has {model.count_params():,} parameters")

print(f"freezing unused head and creating single-output model for {args.head}...")

if args.head == 'classification':
    for layer in model.layers:
        if layer.name == 'regression_output':
            layer.trainable = False
            print(f"frozen layer: {layer.name}")
    
    new_model = Model(
        inputs=model.input,
        outputs=model.get_layer('classification_output').output
    )
    
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='classification_output_auprc'),
            tf.keras.metrics.Precision(name='classification_output_precision'),
            tf.keras.metrics.Recall(name='classification_output_recall')
        ]
    )
    
elif args.head == 'regression':
    for layer in model.layers:
        if layer.name == 'classification_output':
            layer.trainable = False
            print(f"frozen layer: {layer.name}")
    
    new_model = Model(
        inputs=model.input,
        outputs=model.get_layer('regression_output').output
    )
    
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=masked_binary_crossentropy,
        metrics=[
            masked_mae_metric,
            masked_mse_metric,
            masked_r2_metric
        ]
    )

model = new_model

trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
total_params = model.count_params()
print(f"trainable parameters: {trainable_params:,} / {total_params:,}")
print(f"model now outputs only: {args.head}")
model.summary()

wandb.config.update({'trainable_parameters': trainable_params})

n_train = count_seqs(args.train_dataset)
steps_train = math.ceil(n_train / BATCH)
n_val = count_seqs(args.valid_dataset)
steps_val = math.ceil(n_val / BATCH)
mini_steps = math.ceil(steps_train / 10)
print(f"train steps: {steps_train}, mini_steps: {mini_steps}, val steps: {steps_val}")

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

                X_buf = [f[f'X{c}'][:] for c in buf_chunks]
                Y_buf = [f[f'Y{c}'][:] for c in buf_chunks]
                
                order = [(bi, i)
                         for bi, arr in enumerate(X_buf)
                         for i in range(arr.shape[0])]
                random.shuffle(order)

                for buf_idx, seq_i in order:
                    yield X_buf[buf_idx][seq_i], Y_buf[buf_idx][seq_i]
                    
            random.shuffle(chunk_ids)

def to_single_output(x, y_full):
    if args.head == 'classification':
        y_target = y_full[..., :3]
    elif args.head == 'regression':
        y_target = y_full[..., 3:4]
    
    return x, y_target

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset
      .from_generator(
         lambda: seq_generator(args.train_dataset),
         output_signature=(
            tf.TensorSpec((15000,4), tf.float32), 
            tf.TensorSpec((5000,4), tf.float32),
         )
      )
      .map(to_single_output, num_parallel_calls=AUTOTUNE)
      .batch(BATCH)
      .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset
      .from_generator(
         lambda: seq_generator(args.valid_dataset),
         output_signature=(
            tf.TensorSpec((15000,4), tf.float32),
            tf.TensorSpec((5000,4), tf.float32),
         )
      )
      .map(to_single_output, num_parallel_calls=AUTOTUNE)
      .batch(BATCH)
      .prefetch(AUTOTUNE)
)

if args.head == 'classification':
    monitor_metric = 'val_classification_output_auprc'
    mode = 'max'
elif args.head == 'regression':
    monitor_metric = 'val_masked_r2_metric'
    mode = 'max'

callbacks = [
    tb_callback,
    WandbMetricsLogger(log_freq='batch'),
    tf.keras.callbacks.ModelCheckpoint(CKPT_PATH, save_best_only=False),
    tf.keras.callbacks.ModelCheckpoint(
        BEST_PATH, 
        save_best_only=True, 
        monitor=monitor_metric, 
        mode=mode,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, 
        factor=0.5, 
        patience=5, 
        verbose=1,
        mode=mode,
        min_lr=1e-8
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric, 
        patience=10, 
        verbose=1,
        mode=mode,
        restore_best_weights=True
    ),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda ep, logs: (
        pickle.dump({
            'start_epoch': ep+1, 
            'best_metric': logs.get(monitor_metric, 0),
            'head': args.head
        }, open(STATE_PATH,'wb')),
        np.save(LR_PATH, K.get_value(model.optimizer.learning_rate))
    )),
]

print(f"starting fine-tuning of {args.head} head...")
print(f"monitoring: {monitor_metric}")

history = model.fit(
    train_ds,
    epochs=50,
    steps_per_epoch=mini_steps,
    validation_data=valid_ds,
    validation_steps=steps_val,
    callbacks=callbacks,
    initial_epoch=0,
    verbose=1
)

wandb.finish()

print(f'fine-tuning of {args.head} head completed.')