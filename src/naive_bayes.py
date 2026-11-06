"""Gaussian, Multinomial and Bernoulli Naive Bayes on MNIST (70,000 images, random 80/20 split, seed 42)."""
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

runs = {
    "GaussianNB": (GaussianNB(), X_tr / 255.0, X_te / 255.0),
    "MultinomialNB": (MultinomialNB(), X_tr, X_te),
    "BernoulliNB": (BernoulliNB(), (X_tr > 0).astype(int), (X_te > 0).astype(int)),
}
scores, preds = {}, {}
for name, (model, a, b) in runs.items():
    model.fit(a, y_tr)
    preds[name] = model.predict(b)
    scores[name] = accuracy_score(y_te, preds[name])
    print(f"{name}: {scores[name]:.4f}")

best = max(scores, key=scores.get)
print("Best:", best)
print(confusion_matrix(y_te, preds[best]))
print(classification_report(y_te, preds[best]))
