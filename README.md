# Archelec RAG — Élections législatives françaises de 1988

## Comment utiliser ce projet

Ce projet doit être exécuté **en local**.  
Pour l’utiliser, vous devez cloner le dépôt et suivre les étapes d’installation ci-dessous.

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
