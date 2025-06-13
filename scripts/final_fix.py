#!/usr/bin/env python3
"""
Solution finale simplifiée - Crée directement les datasets Looker depuis CSV
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def create_looker_datasets_from_csv():
    """Crée les datasets Looker directement depuis les fichiers CSV"""
    print("🚀 CRÉATION DIRECTE DES DATASETS LOOKER")
    print("="*50)
    
    # Créer le dossier de sortie
    os.makedirs('looker_data', exist_ok=True)
    
    # Trouver le fichier CSV le plus complet
    csv_files = []
    for file in os.listdir('data/'):
        if file.startswith('bank_reviews_') and file.endswith('.csv'):
            csv_files.append(f'data/{file}')
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé dans data/")
        return False
    
    # Prendre le plus récent (probablement le plus complet)
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"📄 Fichier source: {latest_file}")
    
    try:
        # Charger les données
        df = pd.read_csv(latest_file)
        print(f"📊 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Nettoyer les données de base
        df_clean = df.copy()
        
        # Standardiser les noms de colonnes
        column_mapping = {
            'rating': 'star_rating',
            'bank_name': 'bank_name',
            'branch_name': 'branch_name',
            'review_text': 'review_text',
            'sentiment_label': 'sentiment_label',
            'ensemble_score': 'sentiment_score',
            'topic_label': 'topic_name',
            'topic_probability': 'topic_confidence'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_clean.columns and old_col != new_col:
                df_clean = df_clean.rename(columns={old_col: new_col})
        
        # S'assurer que les colonnes essentielles existent
        essential_columns = {
            'bank_name': 'Banque Inconnue',
            'star_rating': 3,
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'topic_name': 'autres',
            'review_text': 'Pas de texte'
        }
        
        for col, default_value in essential_columns.items():
            if col not in df_clean.columns:
                df_clean[col] = default_value
            else:
                df_clean[col] = df_clean[col].fillna(default_value)
        
        # Nettoyer les valeurs numériques
        if 'star_rating' in df_clean.columns:
            df_clean['star_rating'] = pd.to_numeric(df_clean['star_rating'], errors='coerce').fillna(3)
        
        if 'sentiment_score' in df_clean.columns:
            df_clean['sentiment_score'] = pd.to_numeric(df_clean['sentiment_score'], errors='coerce').fillna(0)
        
        # Ajouter des colonnes calculées pour Looker
        df_clean['rating_category'] = df_clean['star_rating'].apply(
            lambda x: 'Excellent (5★)' if x == 5 
            else 'Bon (4★)' if x == 4
            else 'Moyen (3★)' if x == 3
            else 'Mauvais (1-2★)' if x in [1, 2]
            else 'Non évalué'
        )
        
        df_clean['sentiment_emoji'] = df_clean['sentiment_label'].apply(
            lambda x: '😊 Positif' if str(x).lower() == 'positive'
            else '😞 Négatif' if str(x).lower() == 'negative'
            else '😐 Neutre' if str(x).lower() == 'neutral'
            else '❓ Inconnu'
        )
        
        # Ajouter des métriques de texte si pas présentes
        if 'word_count' not in df_clean.columns:
            df_clean['word_count'] = df_clean['review_text'].astype(str).apply(lambda x: len(x.split()))
        
        if 'char_count' not in df_clean.columns:
            df_clean['char_count'] = df_clean['review_text'].astype(str).apply(len)
        
        # Ajouter une date par défaut si manquante
        if 'review_date' not in df_clean.columns:
            df_clean['review_date'] = '2024-01-01'
        
        # Ajouter des colonnes temporelles
        df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')
        df_clean['review_date'] = df_clean['review_date'].fillna(pd.to_datetime('2024-01-01'))
        
        df_clean['year'] = df_clean['review_date'].dt.year
        df_clean['month'] = df_clean['review_date'].dt.month
        df_clean['month_name'] = df_clean['review_date'].dt.strftime('%B')
        df_clean['quarter'] = df_clean['review_date'].dt.quarter
        
        # Sauvegarder le dataset principal
        main_columns = [
            'bank_name', 'branch_name', 'star_rating', 'rating_category',
            'sentiment_label', 'sentiment_emoji', 'sentiment_score',
            'topic_name', 'review_text', 'word_count', 'char_count',
            'review_date', 'year', 'month', 'month_name', 'quarter'
        ]
        
        # Garder seulement les colonnes qui existent
        available_columns = [col for col in main_columns if col in df_clean.columns]
        df_main = df_clean[available_columns].copy()
        
        df_main.to_csv('looker_data/main_dataset.csv', index=False, encoding='utf-8')
        print(f"✅ Dataset principal créé: {len(df_main)} lignes")
        
        # Créer le résumé par banque
        bank_summary = df_clean.groupby('bank_name').agg({
            'star_rating': ['count', 'mean'],
            'sentiment_score': 'mean',
            'word_count': 'mean'
        }).round(2)
        
        bank_summary.columns = ['total_reviews', 'avg_rating', 'avg_sentiment', 'avg_word_count']
        bank_summary = bank_summary.reset_index()
        
        # Calculer les pourcentages de sentiment
        sentiment_counts = df_clean.groupby(['bank_name', 'sentiment_label']).size().unstack(fill_value=0)
        
        for bank in bank_summary['bank_name']:
            bank_total = bank_summary[bank_summary['bank_name'] == bank]['total_reviews'].iloc[0]
            
            if 'positive' in sentiment_counts.columns:
                positive_count = sentiment_counts.loc[bank, 'positive'] if bank in sentiment_counts.index else 0
                bank_summary.loc[bank_summary['bank_name'] == bank, 'positive_percentage'] = round(positive_count / bank_total * 100, 2)
            
            if 'negative' in sentiment_counts.columns:
                negative_count = sentiment_counts.loc[bank, 'negative'] if bank in sentiment_counts.index else 0
                bank_summary.loc[bank_summary['bank_name'] == bank, 'negative_percentage'] = round(negative_count / bank_total * 100, 2)
        
        # Remplir les NaN avec 0
        bank_summary = bank_summary.fillna(0)
        
        bank_summary.to_csv('looker_data/bank_summary.csv', index=False, encoding='utf-8')
        print(f"✅ Résumé banques créé: {len(bank_summary)} banques")
        
        # Créer l'analyse temporelle
        if 'review_date' in df_clean.columns:
            temporal_analysis = df_clean.groupby(['year', 'month', 'month_name']).agg({
                'star_rating': ['count', 'mean'],
                'sentiment_score': 'mean'
            }).round(2)
            
            temporal_analysis.columns = ['review_count', 'avg_rating', 'avg_sentiment']
            temporal_analysis = temporal_analysis.reset_index()
            
            temporal_analysis.to_csv('looker_data/temporal_trends.csv', index=False, encoding='utf-8')
            print(f"✅ Tendances temporelles créées: {len(temporal_analysis)} périodes")
        
        # Créer l'analyse par topic (si disponible)
        if 'topic_name' in df_clean.columns:
            topic_analysis = df_clean.groupby('topic_name').agg({
                'star_rating': ['count', 'mean'],
                'sentiment_score': 'mean'
            }).round(2)
            
            topic_analysis.columns = ['review_count', 'avg_rating', 'avg_sentiment']
            topic_analysis = topic_analysis.reset_index()
            
            # Calculer le pourcentage positif par topic
            topic_sentiment = df_clean.groupby(['topic_name', 'sentiment_label']).size().unstack(fill_value=0)
            
            for topic in topic_analysis['topic_name']:
                topic_total = topic_analysis[topic_analysis['topic_name'] == topic]['review_count'].iloc[0]
                
                if 'positive' in topic_sentiment.columns:
                    positive_count = topic_sentiment.loc[topic, 'positive'] if topic in topic_sentiment.index else 0
                    topic_analysis.loc[topic_analysis['topic_name'] == topic, 'positive_percentage'] = round(positive_count / topic_total * 100, 2)
            
            topic_analysis = topic_analysis.fillna(0)
            topic_analysis.to_csv('looker_data/topic_analysis.csv', index=False, encoding='utf-8')
            print(f"✅ Analyse topics créée: {len(topic_analysis)} topics")
        
        # Créer les métriques générales
        metrics = {
            "total_reviews": int(len(df_clean)),
            "total_banks": int(df_clean['bank_name'].nunique()),
            "avg_rating": round(float(df_clean['star_rating'].mean()), 2),
            "avg_sentiment": round(float(df_clean['sentiment_score'].mean()), 3),
            "positive_percentage": round(float((df_clean['sentiment_label'] == 'positive').sum() / len(df_clean) * 100), 2),
            "data_source": latest_file,
            "created_at": datetime.now().isoformat()
        }
        
        with open('looker_data/summary_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Métriques générales créées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def create_looker_guide_simple():
    """Crée un guide simplifié pour Looker Studio"""
    
    guide = """
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
"""
    
    with open('looker_data/GUIDE_RAPIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Guide rapide créé")

def main():
    """Fonction principale simplifiée"""
    print("🎯 SOLUTION FINALE SIMPLIFIÉE")
    print("="*40)
    
    # Créer les datasets directement depuis CSV
    if create_looker_datasets_from_csv():
        create_looker_guide_simple()
        
        print("\n🎉 SUCCÈS COMPLET!")
        print("="*40)
        print("📁 Fichiers créés dans looker_data/:")
        
        files_created = []
        for file in os.listdir('looker_data/'):
            if file.endswith(('.csv', '.json', '.md')):
                files_created.append(f"  ✅ {file}")
        
        for file in files_created:
            print(file)
        
        print(f"\n📊 Prêt pour Looker Studio!")
        print("1. Aller sur studio.google.com")
        print("2. Upload main_dataset.csv")
        print("3. Créer vos graphiques")
        print("4. Suivre le guide: looker_data/GUIDE_RAPIDE.md")
        
    else:
        print("❌ Échec de la création des datasets")

if __name__ == "__main__":
    main()