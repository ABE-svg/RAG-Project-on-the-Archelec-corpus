# Archelec RAG - Elections legislatives françaises de 1988

## Apercu

Le corpus Archelec (CEVIPOF / Sciences Po) regroupe plus de **3 500 professions de foi** numerisees, couvrant les deux tours des legislatives de 1988. Ce projet permet d'interroger ce corpus en langage naturel via une interface web, en obtenant des reponses sourcees directement issues des documents.

---

## Architecture

```
Corpus texte (.txt)
        |
        v
Decoupage en chunks (LangChain - 1000 chars / overlap 200)
        |
        v
Encodage semantique (sentence-transformers/all-MiniLM-L6-v2)
        |
        v
Index vectoriel FAISS  <-- sauvegarde sur disque (faiss_index/)
        |
   Requete utilisateur
        |
        v
Recherche des 5 passages les plus proches (similarite cosinus)
        |
        v
Generation de reponse (Llama-3.1-8B-Instruct via HuggingFace API)
        |
        v
Interface web FastAPI
```

---

## Structure du projet

```
RAG-Project-on-the-Archelec-corpus/
|
|- RAG/
|   |- app.py                              # Serveur FastAPI + interface web
|   |- RAG_updated_with_data_analysis.ipynb  # Notebook d'exploration et de pipeline
|   |- faiss_index/                        # Index FAISS genere au premier lancement
|   |- text_files/                         # Corpus texte (professions de foi _PF_)
|   `- .env                                # Token HuggingFace (non versionne)
|
`- README.md
```

---

## Prerequis

- Python 3.10 ou superieur
- Un compte HuggingFace avec acces au modele `meta-llama/Llama-3.1-8B-Instruct`
- Le corpus texte place dans `text_files/` (fichiers `.txt` avec `_PF_` dans le nom)

---

## Installation

```bash
# 1. Cloner le depot
git clone <url-du-repo>
cd RAG-Project-on-the-Archelec-corpus/RAG

# 2. Creer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dependances
pip install fastapi uvicorn python-dotenv pydantic \
            langchain langchain-huggingface langchain-community \
            faiss-cpu sentence-transformers numpy
```

---

## Configuration

### Option 1 : fichier `.env` (recommande pour le proprietaire)

Creer un fichier `.env` a la racine du dossier `RAG/` :

```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Le token est charge automatiquement au demarrage. Aucune action supplementaire n'est requise.

### Option 2 : interface web (pour les autres utilisateurs)

Si vous n'avez pas de fichier `.env`, lancez quand meme le serveur. Au premier chargement, ouvrez l'onglet **Parametres** dans la sidebar, entrez votre token HuggingFace personnel, puis cliquez sur **Appliquer**. Le token est conserve dans votre navigateur pour les sessions suivantes.

> Votre token HuggingFace est disponible sur : https://huggingface.co/settings/tokens

---

## Lancement

```bash
cd RAG
uvicorn app:app --reload
```

Puis ouvrir http://localhost:8000 dans votre navigateur.

> **Premier lancement :** la construction de l'index FAISS prend plusieurs minutes (encodage de ~16 000 chunks sur CPU). Les lancements suivants chargent l'index depuis le disque en quelques secondes.

---

## Fonctionnement

1. L'utilisateur pose une question en langage naturel
2. La question est encodee en vecteur par `all-MiniLM-L6-v2`
3. Les 5 passages les plus similaires sont recuperes depuis l'index FAISS
4. Ces passages sont fournis comme contexte au modele Llama-3.1-8B
5. Le modele genere une reponse factuelle basée uniquement sur ce contexte
6. L'interface affiche la reponse et les passages sources (nom de fichier + extrait)

---

## Notebook d'analyse

Le fichier `RAG_updated_with_data_analysis.ipynb` documente l'ensemble de la demarche :

- Statistiques descriptives du corpus (longueur, vocabulaire, TTR)
- Analyse lexicale : bigrammes, TF-IDF par tour, loi de Zipf
- Justification du `chunk_size` par la distribution des longueurs
- Simulation du decoupage et heatmap de similarite inter-documents
- Evaluation de la qualite OCR (ratio alpha, lignes courtes, chiffres)
- Construction et test du pipeline RAG complet

---

## Modeles utilises

| Composant | Modele |
|---|---|
| Encodage semantique | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation de texte | `meta-llama/Llama-3.1-8B-Instruct` |

---

## Limites connues

- Le modele repond uniquement a partir des documents indexés. Si un theme n'est pas couvert dans le corpus, il l'indique explicitement.
- Les bulletins de vote (`_BV_`) sont exclus de l'index : seules les professions de foi (`_PF_`) sont utilisees.
- Les en-tetes CEVIPOF/Sciences Po presents dans tous les fichiers peuvent influencer marginalement la recherche semantique.

