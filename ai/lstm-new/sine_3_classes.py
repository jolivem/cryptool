# ==========================================
# Classif 3 classes (DOWN/FLAT/UP) sur sinusoïde
# ==========================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
WINDOW      = 100        # longueur de fenêtre
HORIZON     = 1         # pas suivant
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15      # test = 1 - train - val
EPS         = 1e-6      # bande "FLAT" ; si None => auto (percentile)
SEED        = 1337
SAV_PATH    = "artifacts_sine_trend3"

os.makedirs(SAV_PATH, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# -------------------------
# 0) Génération d'une sinusoïde (vous pouvez remplacer ce bloc)
# -------------------------
T_TOTAL   = 10000
PERIOD    = 200          # période de la sinusoïde
AMP       = 1.0
NOISE_STD = 0.1         # bruit gaussien pour rendre la tâche non triviale

t = np.arange(T_TOTAL, dtype=np.float32)
close = AMP * np.sin(2*np.pi*t/PERIOD) + NOISE_STD * np.random.randn(T_TOTAL).astype(np.float32)

# DataFrame minimal avec une colonne 'close'
df = pd.DataFrame({'close': close})

# -------------------------
# 1) Features causales (pas d'OHLC ici)
# -------------------------
work = pd.DataFrame(index=df.index)
work['close']        = df['close'].astype(np.float32)
work['diff1']        = work['close'].diff()
work['diff2']        = work['diff1'].diff()
work['roll_mean_10'] = work['close'].rolling(10).mean()
work['roll_std_10']  = work['close'].rolling(10).std()
# pente approx sur 10 pas (différence / pas)
work['roll_slope_10']= (work['close'] - work['close'].shift(10)) / 10.0

# Nettoyage des NaN (dus à diff/rolling)
work = work.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

FEATURE_COLS = [
    'close', 'diff1', 'diff2',
    'roll_mean_10', 'roll_std_10', 'roll_slope_10'
]
F = len(FEATURE_COLS)

X_all = work[FEATURE_COLS].to_numpy(np.float32)  # (T, F)
T = len(work)

# -------------------------
# 2) Labels 3 classes (à partir de la variation suivante)
#    0 = DOWN, 1 = FLAT, 2 = UP
# -------------------------
# Variation simple (pas de log ici car la sinusoïde peut être négative)
ret1 = work['diff1'].to_numpy(np.float32)  # même longueur que work

# Bande FLAT : auto si EPS=None (ici on prend le 40e percentile des |ret1|)
if EPS is None:
    valid = np.isfinite(ret1)
    EPS = float(np.percentile(np.abs(ret1[valid]), 40))
    print(f"EPS auto (40e pct |ret1|) = {EPS:.6f}")

labels3 = np.where(ret1 >  EPS, 2,
          np.where(ret1 < -EPS, 0, 1)).astype(np.int64)

# -------------------------
# 3) Splits temporels
# -------------------------
n_train = int(T * TRAIN_SPLIT)
n_val   = int(T * VAL_SPLIT)
n_test  = T - n_train - n_val

train_slice = slice(0, n_train)
val_slice   = slice(n_train, n_train + n_val)
test_slice  = slice(n_train + n_val, T)

# -------------------------
# 4) Standardisation des features sur TRAIN uniquement
# -------------------------
mu_x  = X_all[train_slice].mean(axis=0)
std_x = X_all[train_slice].std(axis=0) + 1e-8
Z = (X_all - mu_x) / std_x

# -------------------------
# 5) Windowing (X multivarié, y classe)
# -------------------------
def make_windows_cls3(Z, y_classes, slc, window=50, horizon=1):
    """
    Retourne X,y pour classification 3 classes.
    X: (n, window, F), y: (n,)
    Label index j = i + window + horizon - 1 (prochain pas).
    """
    start, end = slc.start, slc.stop
    F = Z.shape[1]
    xs, ys = [], []
    for i in range(start, end - window - horizon + 1):
        j = i + window + horizon - 1
        xs.append(Z[i : i + window, :])
        ys.append(y_classes[j])

    if len(xs) == 0:
        # IMPORTANT : retourner des tenseurs vides aux bonnes dimensions
        X = np.empty((0, window, F), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)
    else:
        X = np.stack(xs).astype(np.float32)        # -> (n, window, F)
        y = np.asarray(ys, dtype=np.int32)         # int32 pour Keras
    return X, y

Xtr, ytr = make_windows_cls3(Z, labels3, train_slice, WINDOW, HORIZON)
Xva, yva = make_windows_cls3(Z, labels3, val_slice,   WINDOW, HORIZON)
Xte, yte = make_windows_cls3(Z, labels3, test_slice,  WINDOW, HORIZON)

print(f"Windows -> train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte)}  | features:{F}")

# -------------------------
# 6) Pondération de classes (souvent utile si FLAT domine)
# -------------------------
def compute_class_weights(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = {k: (total / (n_classes * counts[k])) if counts[k] > 0 else 1.0
               for k in range(n_classes)}
    return weights, counts

class_weight, train_counts = compute_class_weights(ytr, n_classes=3)
print("Répartition train (DOWN, FLAT, UP) :", train_counts.tolist())
print("Poids de classes :", class_weight)

# -------------------------
# 7) Modèle LSTM -> Dense -> Softmax(3)
# -------------------------
inputs = tf.keras.Input(shape=(WINDOW, F))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
logits = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs, logits)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
]

history = model.fit(Xtr, ytr,
                    validation_data=(Xva, yva),
                    epochs=100,
                    batch_size=128,
                    callbacks=cb,
                    class_weight=class_weight,
                    verbose=1)

# -------------------------
# 8) Évaluation + sauvegardes
# -------------------------
test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
print(f"[TEST] loss:{test_loss:.4f}  acc:{test_acc:.4f}")

model.save(os.path.join(SAV_PATH, "sine_lstm_trend3.keras"))
np.savez(os.path.join(SAV_PATH, "sine_trend3_artifacts.npz"),
         mu_x=mu_x, std_x=std_x,
         feature_cols=np.array(FEATURE_COLS),
         eps=EPS, window=WINDOW, horizon=HORIZON)
print("Modèle & scalers sauvegardés dans:", SAV_PATH)

# -------------------------
# 9) Prédictions test + graphiques
# -------------------------
probs_te = model.predict(Xte, verbose=0)   # (n, 3)
yhat = np.argmax(probs_te, axis=1)         # (n,)
class_names = np.array(["DOWN","FLAT","UP"])
idx = np.arange(len(yte))

# (a) Probabilités par classe
plt.figure(figsize=(11,4))
for k, name in enumerate(class_names):
    plt.plot(idx, probs_te[:, k], label=f"P({name})")
plt.ylim(0, 1)
plt.title("Sinusoïde — probabilités prédites (3 classes) sur test")
plt.xlabel("Index fenêtre test"); plt.ylabel("Probabilité")
plt.legend(); plt.tight_layout()
p1 = os.path.join(SAV_PATH, "sine_trend3_probabilities.png")
plt.savefig(p1, dpi=130)
print("Saved:", p1)

# (b) Classe prédite vs réelle (avec zones correctes)
plt.figure(figsize=(11,4))
plt.step(idx, yte,  where="post", label="Réel",      alpha=0.85)
plt.step(idx, yhat, where="post", label="Prédit",    alpha=0.85)
correct = (yhat == yte)
ax = plt.gca()
plt.fill_between(idx, -0.5, 2.5, where=correct, alpha=0.08, step="pre",
                 transform=ax.get_xaxis_transform())
plt.yticks([0,1,2], class_names)
plt.ylim(-0.5, 2.5)
plt.title("Sinusoïde — classe prédite vs réelle (test)")
plt.xlabel("Index fenêtre test"); plt.ylabel("Classe")
plt.legend(); plt.tight_layout()
p2 = os.path.join(SAV_PATH, "sine_trend3_pred_vs_actual.png")
plt.savefig(p2, dpi=130)
print("Saved:", p2)

# (c) Matrice de confusion
try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, yhat, labels=[0,1,2])
except Exception:
    cm = np.zeros((3,3), dtype=int)
    for t_, p_ in zip(yte, yhat):
        cm[int(t_), int(p_)] += 1

plt.figure(figsize=(4.8,4.8))
plt.imshow(cm)
plt.xticks([0,1,2], class_names); plt.yticks([0,1,2], class_names)
plt.title("Matrice de confusion (test)")
plt.xlabel("Prédit"); plt.ylabel("Réel")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center")
plt.tight_layout()
p3 = os.path.join(SAV_PATH, "sine_trend3_confusion_matrix.png")
plt.savefig(p3, dpi=130)
print("Saved:", p3)

# (d) Courbe d'accuracy cumulée
run_acc = np.cumsum(correct) / (np.arange(len(correct)) + 1)
plt.figure(figsize=(11,3.5))
plt.plot(idx, run_acc, label="Accuracy cumulée")
plt.ylim(0,1)
plt.title("Sinusoïde — accuracy cumulée (test)")
plt.xlabel("Index fenêtre test"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
p4 = os.path.join(SAV_PATH, "sine_trend3_running_accuracy.png")
plt.savefig(p4, dpi=130)
print("Saved:", p4)

# -------------------------
# 10) Inference 1 pas à partir de la dernière fenêtre
# -------------------------
last_win = Z[-WINDOW:, :]                      # (WINDOW, F)
probs_next = model.predict(last_win[None, ...], verbose=0)[0]  # (3,)
pred_next  = int(np.argmax(probs_next))
print(f"Prochaine classe prédite: {class_names[pred_next]}  "
      f"(P_DOWN={probs_next[0]:.3f}, P_FLAT={probs_next[1]:.3f}, P_UP={probs_next[2]:.3f})")
