# Archelec RAG — Élections législatives françaises de 1988

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
sauvegardé dans faiss_index/
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
└── .env
```

---

## Description des fichiers

| Fichier / dossier | Description |
|---|---|
| `app.py` | Application FastAPI, interface web et logique RAG |
| `requirements.txt` | Dépendances Python nécessaires |
| `RAG_updated_with_data_analysis_1.ipynb` | Notebook d’analyse et de construction du pipeline |
| `text_files/` | Corpus texte |
| `faiss_index/` | Index vectoriel FAISS |
| `.env` | Token HuggingFace |

---

## Données

Le projet utilise les fichiers texte du corpus Archelec pour l’année **1988**.

Statistiques principales :

- **3 628 documents**
- **2 002 003 mots**
- **16 815 chunks**
- **≈ 4.7 chunks par document**

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

Créer un fichier `.env` :

```text
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## Lancement

```bash
uvicorn app:app --reload
```

Ouvrir :

```text
http://127.0.0.1:8000
```

- **Premier lancement :** construction de l’index FAISS (plusieurs minutes)  
- **Lancements suivants :** chargement rapide depuis le disque  

---

## Fonctionnement

1. L’utilisateur pose une question  
2. La question est encodée en vecteur  
3. FAISS récupère les 5 chunks les plus proches  
4. Ces passages sont envoyés au modèle LLM  
5. Le modèle génère une réponse basée sur le contexte  
6. L’interface affiche la réponse + les sources  

---

## Notebook d’analyse

Le notebook `RAG_updated_with_data_analysis_1.ipynb` contient :

- Analyse descriptive du corpus  
- Distribution des longueurs  
- Analyse par tour électoral  
- Qualité OCR  
- Fréquences lexicales  
- Bigrammes  
- Diversité lexicale  
- Loi de Zipf  
- Similarité TF-IDF  
- Justification du chunking  
- Construction du pipeline RAG  
- Évaluation  

---

## Évaluation

Comparaison entre deux approches :

| Méthode | Description |
|---|---|
| RAG | Réponse avec contexte récupéré |
| LLM seul | Réponse sans contexte |

Métrique utilisée :

```text
Groundedness = mots communs entre réponse et contexte / mots de la réponse
```

Résultats :

| Méthode | Score moyen |
|---|---:|
| RAG | 0.585 |
| LLM seul | 0.148 |

➡️ Le RAG produit des réponses plus ancrées dans le corpus.

---

## Modèles utilisés

| Composant | Modèle |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | meta-llama/Llama-3.1-8B-Instruct |
| Recherche | FAISS |

---

## Limites

- Projet limité à l’année 1988  
- Présence de bruit OCR  
- Métrique d’évaluation simplifiée  
- Sensible aux paramètres (chunk size, embeddings, prompt)  

---

## Améliorations possibles

- Ajouter d’autres années (1973, 1978, 1981, 1993)  
- Tester d’autres modèles d’embeddings  
- Améliorer l’évaluation (annotations humaines)  
- Optimiser le chunking  
- Déployer l’application en ligne  

---

## Auteur

Kevin Abe  
ENSAE Paris  
kevin.abe@ensae.fr
