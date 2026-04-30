# Archelec RAG — Élections législatives françaises de 1988

## Comment utiliser ce projet

Ce projet doit être exécuté **en local**.  
Pour l’utiliser, vous devez **cloner le dépôt** et suivre les étapes d’installation ci-dessous.

---

## Aperçu

Ce projet propose une application de question-réponse sur le corpus **Archelec**, constitué de professions de foi de candidats aux élections législatives françaises de 1988.

L’objectif est de permettre à un utilisateur de poser une question en langage naturel et d’obtenir une réponse courte, factuelle et fondée sur les documents du corpus.

Le système repose sur une architecture **Retrieval-Augmented Generation (RAG)** : il récupère d’abord les passages les plus pertinents dans les textes, puis utilise un modèle de langage pour générer une réponse contextualisée.

---

## Architecture

```text
Corpus texte (.txt)
        |
        v
Découpage en chunks
LangChain — 1000 caractères / overlap 200
        |
        v
Encodage sémantique
sentence-transformers/all-MiniLM-L6-v2
        |
        v
Index vectoriel FAISS
(sauvegardé localement)
        |
        v
Question utilisateur
        |
        v
Recherche des passages les plus proches
        |
        v
Génération de réponse
Llama-3.1-8B-Instruct via HuggingFace
        |
        v
Interface web FastAPI
```

---

## Structure du projet

```text
RAG-Project-on-the-Archelec-corpus/
│
├── app.py
├── requirements.txt
├── README.md
├── RAG_updated_with_data_analysis_1.ipynb
│
├── text_files/
│   └── fichiers .txt du corpus
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
└── .env   (à créer)
```

---

## Description des fichiers

| Fichier / dossier | Description |
|---|---|
| `app.py` | Application FastAPI contenant le pipeline RAG et l’interface web |
| `requirements.txt` | Dépendances Python |
| `RAG_updated_with_data_analysis_1.ipynb` | Notebook complet : analyse + pipeline + évaluation |
| `text_files/` | Corpus des professions de foi |
| `faiss_index/` | Index FAISS sauvegardé |
| `.env` | Token HuggingFace (non versionné) |

---

## Données

Le projet utilise le corpus Archelec pour l’année **1988**.

Statistiques principales :

- **3 628 documents**
- **2 002 003 mots**
- **16 815 chunks**
- **≈ 4.7 chunks par document**

Le choix de 1988 est motivé par des contraintes de calcul (embeddings + FAISS).

---

## Installation

```bash
git clone <url-du-repo>
cd RAG-Project-on-the-Archelec-corpus

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

---

## Configuration

Créer un fichier `.env` à la racine :

```text
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Obtenir le token ici :  
https://huggingface.co/settings/tokens

---

## Lancement

```bash
uvicorn app:app --reload
```

Puis ouvrir :

```text
http://127.0.0.1:8000
```

### Important

- Premier lancement : création de l’index FAISS (plusieurs minutes)
- Lancements suivants : chargement rapide

---

## Fonctionnement

1. L’utilisateur pose une question  
2. La question est transformée en embedding  
3. FAISS récupère les **5 passages les plus similaires**  
4. Ces passages sont envoyés au LLM  
5. Le modèle génère une réponse basée uniquement sur le contexte  
6. L’interface affiche la réponse + les sources  

---

## Notebook d’analyse

Le notebook contient :

- Analyse descriptive du corpus  
- Distribution des longueurs  
- Analyse par tour électoral  
- Analyse géographique  
- Qualité OCR (alpha ratio, digits, lignes courtes)  
- Fréquences lexicales  
- Bigrammes  
- Diversité lexicale (TTR)  
- Loi de Zipf  
- Similarité TF-IDF  
- Heatmap de similarité  
- Justification du chunking  
- Simulation du découpage  
- Construction du pipeline RAG  
- Évaluation  

---

## Évaluation

Deux approches sont comparées :

| Méthode | Description |
|---|---|
| RAG | Réponse avec documents récupérés |
| LLM seul | Réponse sans contexte |

### Métrique

```text
Groundedness = mots communs entre réponse et contexte / mots de la réponse
```

### Résultats

| Méthode | Score moyen |
|---|---:|
| RAG | 0.585 |
| LLM seul | 0.148 |

➡️ Le RAG produit des réponses beaucoup plus fiables et ancrées dans le corpus.

---

## Modèles utilisés

| Composant | Modèle |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | meta-llama/Llama-3.1-8B-Instruct |
| Recherche | FAISS |

---

## Limites

- Corpus limité à 1988  
- Présence de bruit OCR  
- Évaluation basée sur recouvrement lexical  
- Sensibilité aux hyperparamètres (chunk size, k, embeddings, prompt)  

---

## Améliorations possibles

- Ajouter les autres années (1973, 1978, 1981, 1993)  
- Tester d’autres modèles d’embeddings  
- Améliorer l’évaluation (annotation humaine)  
- Optimiser le chunking  
- Ajouter une interface plus avancée  
- Déployer en ligne  

---

## Auteur

Kevin Abe  
ENSAE Paris  
kevin.abe@ensae.fr
