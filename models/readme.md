# models

trained SPLAIRE model weights in Keras format.

## naming

`{Dataset}_{Length}_v{Fold}_{Task}_best.keras`

- **Dataset**: `Ref` (reference sequences), `Var` (variant sequences)
- **Length**: 100 (context window in bp)
- **Fold**: v1–v5 (cross-validation fold)
- **Task**: `cls` (classification), `reg` (regression)

## contents

10 classification models: `*_cls_best.keras`
10 regression models: `*_reg_best.keras`

## usage

```python
import tensorflow as tf

model = tf.keras.models.load_model('Ref_100_v1_cls_best.keras')
# input shape: (batch, seq_len, 4) one-hot encoded pre-mRNA
predictions = model.predict(sequences)
```
