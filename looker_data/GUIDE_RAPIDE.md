
# 🎯 GUIDE RAPIDE LOOKER STUDIO

## 📁 Fichiers créés
- `main_dataset.csv` : Dataset principal pour tous les graphiques
- `bank_summary.csv` : Résumé par banque pour les KPIs
- `temporal_trends.csv` : Évolution temporelle (si dates disponibles)
- `topic_analysis.csv` : Analyse par sujet (si topics disponibles)
- `summary_metrics.json` : Métriques clés

## 🚀 ÉTAPES LOOKER STUDIO

### 1. Connexion des données
1. Aller sur **studio.google.com**
2. Cliquer **"Créer" → "Source de données"**
3. Choisir **"Upload de fichier"**
4. Télécharger **main_dataset.csv**

### 2. Configuration des colonnes
- **bank_name** → Dimension (Texte)
- **star_rating** → Métrique (Nombre)
- **sentiment_score** → Métrique (Nombre)
- **review_date** → Dimension (Date)
- **rating_category** → Dimension (Texte)
- **sentiment_emoji** → Dimension (Texte)

### 3. Premiers graphiques à créer

#### KPIs (Cartes de score)
- **Total avis** : COUNT(bank_name)
- **Note moyenne** : AVG(star_rating)  
- **Sentiment moyen** : AVG(sentiment_score)

#### Graphique en barres
- **Dimension** : bank_name
- **Métrique** : AVG(star_rating)
- **Titre** : "Note moyenne par banque"

#### Graphique en secteurs  
- **Dimension** : sentiment_emoji
- **Métrique** : COUNT(bank_name)
- **Titre** : "Distribution des sentiments"

### 4. Filtres recommandés
- Sélecteur de banque (bank_name)
- Sélecteur de période (review_date)
- Sélecteur de sentiment (sentiment_emoji)

## 🎨 Couleurs recommandées
- Positif 😊 : #4CAF50 (Vert)
- Négatif 😞 : #F44336 (Rouge)
- Neutre 😐 : #FF9800 (Orange)
- Principal : #2196F3 (Bleu)

## ✅ C'est tout !
Avec ces étapes, vous aurez un dashboard fonctionnel en 10 minutes !
