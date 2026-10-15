# TP 2 — Détection du sarcasme en arabe (ArSarcasm-v2)

Partie A : audit du texte brut, taux de sarcasme, classifieur trivial, baseline TF-IDF et raccourcis de surface. Les parties B à G ne sont pas encore implémentées.

Le fichier de test officiel sert aux taux descriptifs et au retrait des fuites dans le train. Il ne sert ni à ajuster le TF-IDF, ni à entraîner la régression logistique, ni à choisir le seuil. Les fichiers `predictions/<modèle>_seed<k>.csv` seront écrits en partie F.

## Installation

Python 3.10 à 3.13. Le fichier `.python-version` demande 3.12 : `gensim` n'a pas de wheel pour 3.14, et `uv sync` installe cet interpréteur s'il manque. `arabert` impose `emoji==1.4.2` ; le détecteur d'emojis accepte aussi l'API plus récente.

```bash
uv sync
```

## Exécution

Depuis la racine du dépôt :

```bash
uv run python scripts/part_a.py
uv run jupyter lab notebooks/part_a.ipynb
uv run pytest
```

Le script et le notebook appellent les mêmes fonctions de `src/`. Relancer le script remplace les lignes de la partie A dans `results/results.csv` et conserve les lignes des modèles ajoutés plus tard.

## Dépôt

```
src/data/            chargement, split imposé, retrait des fuites
src/preprocess/      audit brut, clé de dédoublonnage, texte TF-IDF
src/models/          baseline TF-IDF ; familles suivantes à ajouter ici
src/train/           graines et entraînement de la baseline
src/evaluate/        métriques, seuil, journal, export des prédictions
notebooks/           part_a.ipynb
scripts/part_a.py    même pipeline, sans interface
tests/
results/results.csv  journal (colonnes de test vides jusqu'à la partie F)
predictions/         prédictions de test, partie F
```

Graine fixée : 42 (`random`, NumPy, et torch / transformers lorsqu'ils sont importés). Split imposé : 10 % de validation, `random_state=42`, stratifié sur `sarcasm`.

## Parties suivantes

- B — PLM zero-shot (NLI) et heuristique de sentiment
- C — embeddings statiques et réseaux profonds
- D — PLM gelé
- E — fine-tuning mono-tâche et multi-tâche
- F — tests de généralisation et export `predictions/<modèle>_seed<k>.csv` (`id`, `y_true`, `y_pred`, `proba_sarcasm`)
- G — comparaison et décision
