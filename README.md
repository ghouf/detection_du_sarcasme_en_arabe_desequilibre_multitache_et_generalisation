# TP 2 — Détection du sarcasme en arabe (ArSarcasm-v2)

Partie A : audit du texte brut, taux de sarcasme, classifieur trivial, baseline TF-IDF et raccourcis de surface. Partie B : PLM utilisé tel quel, sans entraînement (zero-shot NLI et heuristique de sentiment). Partie C : embeddings statiques (aléatoires ou AraVec) et réseaux PyTorch (BiLSTM, attention, mini-Transformer). Les parties D à G ne sont pas encore implémentées.

Le fichier de test officiel sert aux taux descriptifs, au taux d'OOV de la partie C et au retrait des fuites dans le train. Il ne sert ni à ajuster le TF-IDF, ni à entraîner un modèle, ni à choisir un seuil, ni à scorer les modèles des parties B et C. Les fichiers `predictions/<modèle>_seed<k>.csv` seront écrits en partie F.

## Installation

Python 3.10 à 3.13. Le fichier `.python-version` demande 3.12 : `gensim` n'a pas de wheel pour 3.14, et `uv sync` installe cet interpréteur s'il manque. `arabert` impose `emoji==1.4.2` ; le détecteur d'emojis accepte aussi l'API plus récente.

```bash
uv sync
```

## Exécution

Depuis la racine du dépôt :

```bash
uv run python scripts/part_a.py
uv run python scripts/part_b.py
uv run python scripts/part_c.py
uv run jupyter lab notebooks/part_a.ipynb
uv run jupyter lab notebooks/part_b.ipynb
uv run jupyter lab notebooks/part_c.ipynb
uv run pytest
```

Le script et le notebook d'une partie appellent les mêmes fonctions de `src/`. Relancer un script remplace les lignes de ses modèles dans `results/results.csv` et conserve les autres.

`uv run python scripts/part_b.py --limit 4` score un préfixe de la validation et n'écrit pas le journal. Le passage complet télécharge les deux modèles et ne touche pas au fichier de test officiel.

`uv run python scripts/part_c.py --limit 32 --skip-aravec` entraîne C1–C3 sur un préfixe et n'écrit pas le journal. Le passage complet utilise le même budget pour chaque architecture (Adam 1e-3, batch 64, 8 époques, patience 2) et l'arrêt anticipé sur le F1 sarcastique de val. AraVec n'est pas téléchargé : déposer `full_uni_cbow_100_twitter.mdl` dans `data/aravec/` pour les runs figés puis dégelés. `--char-cnn` et `--focal` ajoutent les variantes optionnelles.

## Dépôt

```
src/data/            chargement, split imposé, retrait des fuites
src/preprocess/      audit brut, clé de dédoublonnage, texte TF-IDF
src/models/          baseline TF-IDF ; PLM zero-shot ; vocabulaire, AraVec et réseaux de la partie C
src/train/           graines, baseline, budget commun et déséquilibre de la partie C
src/evaluate/        métriques, seuil, journal, export des prédictions
notebooks/           part_a.ipynb, part_b.ipynb, part_c.ipynb
scripts/part_a.py    même pipeline que le notebook A, sans interface
scripts/part_b.py    zero-shot NLI et heuristique de sentiment
scripts/part_c.py    embeddings statiques et réseaux profonds
tests/
results/results.csv  journal (colonnes de test vides jusqu'à la partie F)
predictions/         prédictions de test, partie F
```

Graine fixée : 42 (`random`, NumPy, et torch / transformers lorsqu'ils sont importés). Split imposé : 10 % de validation, `random_state=42`, stratifié sur `sarcasm`.

## Parties suivantes

- D — PLM gelé
- E — fine-tuning mono-tâche et multi-tâche
- F — tests de généralisation et export `predictions/<modèle>_seed<k>.csv` (`id`, `y_true`, `y_pred`, `proba_sarcasm`)
- G — comparaison et décision
