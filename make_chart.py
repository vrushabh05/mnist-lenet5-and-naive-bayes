import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
nb = list(csv.DictReader(open("results/naive_bayes.csv")))
log = list(csv.DictReader(open("results/lenet5_training_log.csv")))
cm = np.array([[int(v) for v in r[1:]] for r in list(csv.reader(open("results/lenet5_confusion_matrix.csv")))[1:]])

fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
names = [r["model"] for r in nb] + ["LeNet-5 (CNN)"]
vals = [float(r["accuracy"]) for r in nb] + [0.9884]
bars = ax.barh(names[::-1], vals[::-1], color=["#1d4ed8"] + ["#9ca3af"] * 3, height=0.55)
for b, v in zip(bars, vals[::-1]): ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0, 1.1); ax.set_xlabel("Test accuracy"); ax.set_title("MNIST: Naive Bayes variants vs LeNet-5", fontsize=11)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.savefig("assets/model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
e = [int(r["epoch"]) for r in log]
axes[0].plot(e, [float(r["train_acc"]) for r in log], color="#c2410c", lw=2, label="Train")
axes[0].plot(e, [float(r["test_acc"]) for r in log], color="#1d4ed8", lw=2, label="Test")
axes[0].set_ylim(0.85, 1.0); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].set_title("LeNet-5 training", fontsize=11)
axes[0].legend(frameon=False); [axes[0].spines[s].set_visible(False) for s in ("top", "right")]
axes[1].imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        if cm[i, j]: axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > 500 else "black")
axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10)); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_title("LeNet-5 confusion matrix (10,000 test images)", fontsize=11)
plt.tight_layout(); plt.savefig("assets/lenet5_results.png")
