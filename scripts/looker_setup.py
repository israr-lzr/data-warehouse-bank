#!/usr/bin/env python3
"""
Configuration et préparation des données pour Looker Studio
Étape 6 du projet Data Warehouse - Visualisation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Base de données et export
import sqlite3
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/looker_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LookerStudioDataPrep:
    """Classe pour préparer les données pour Looker Studio"""
    
    def __init__(self, db_path: str = 'data/bank_reviews_dw.db'):
        """
        Initialise la préparation des données Looker
        
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.output_dir = 'looker_data'
        self.charts_dir = 'charts'
        self.setup_directories()
        
    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        for directory in [self.output_dir, self.charts_dir]:
            os.makedirs(directory, exist_ok=True)
        logger.info("Répertoires créés pour Looker Studio")
    
    def extract_data_for_looker(self) -> Dict[str, pd.DataFrame]:
        """
        Extrait toutes les données nécessaires pour Looker Studio
        
        Returns:
            Dictionnaire avec les DataFrames pour chaque visualisation
        """
        logger.info("Extraction des données pour Looker Studio...")
        
        datasets = {}
        
        try:
            with self.engine.connect() as conn:
                
                # 1. DATASET PRINCIPAL - Vue d'ensemble
                query_main = """
                SELECT 
                    f.review_id,
                    b.bank_name,
                    br.branch_name,
                    br.city,
                    br.region,
                    d.full_date as review_date,
                    d.year,
                    d.month,
                    d.month_name,
                    d.quarter,
                    d.day_name,
                    d.is_weekend,
                    f.star_rating,
                    s.sentiment_label,
                    f.ensemble_score,
                    f.sentiment_confidence,
                    t.topic_name,
                    f.topic_probability,
                    f.word_count,
                    f.char_count,
                    f.positive_words_count,
                    f.negative_words_count,
                    f.has_exclamation,
                    f.has_question,
                    f.has_caps,
                    f.data_quality_score,
                    f.review_text,
                    f.textblob_polarity,
                    f.vader_compound,
                    f.custom_sentiment_score
                FROM fact_reviews f
                LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
                LEFT JOIN dim_branch br ON f.branch_id = br.branch_id
                LEFT JOIN dim_date d ON f.date_id = d.date_id
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                LEFT JOIN dim_topic t ON f.topic_id = t.topic_id
                WHERE f.star_rating IS NOT NULL
                ORDER BY d.full_date DESC
                """
                datasets['main_dataset'] = pd.read_sql(query_main, conn)
                
                # 2. AGRÉGATIONS PAR BANQUE
                query_bank_summary = """
                SELECT 
                    b.bank_name,
                    COUNT(f.review_id) as total_reviews,
                    AVG(f.star_rating) as avg_rating,
                    AVG(f.ensemble_score) as avg_sentiment_score,
                    COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) as positive_reviews,
                    COUNT(CASE WHEN s.sentiment_label = 'negative' THEN 1 END) as negative_reviews,
                    COUNT(CASE WHEN s.sentiment_label = 'neutral' THEN 1 END) as neutral_reviews,
                    ROUND(COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) * 100.0 / COUNT(*), 2) as positive_percentage,
                    ROUND(COUNT(CASE WHEN s.sentiment_label = 'negative' THEN 1 END) * 100.0 / COUNT(*), 2) as negative_percentage,
                    AVG(f.word_count) as avg_word_count,
                    AVG(f.data_quality_score) as avg_data_quality,
                    MIN(d.full_date) as first_review_date,
                    MAX(d.full_date) as last_review_date
                FROM fact_reviews f
                LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                LEFT JOIN dim_date d ON f.date_id = d.date_id
                GROUP BY b.bank_id, b.bank_name
                ORDER BY total_reviews DESC
                """
                datasets['bank_summary'] = pd.read_sql(query_bank_summary, conn)
                
                # 3. TENDANCES TEMPORELLES
                query_temporal = """
                SELECT 
                    d.year,
                    d.month,
                    d.month_name,
                    d.quarter,
                    COUNT(f.review_id) as review_count,
                    AVG(f.star_rating) as avg_rating,
                    AVG(f.ensemble_score) as avg_sentiment,
                    COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) as positive_count,
                    COUNT(CASE WHEN s.sentiment_label = 'negative' THEN 1 END) as negative_count,
                    COUNT(CASE WHEN s.sentiment_label = 'neutral' THEN 1 END) as neutral_count,
                    ROUND(COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) * 100.0 / COUNT(*), 2) as positive_percentage
                FROM fact_reviews f
                LEFT JOIN dim_date d ON f.date_id = d.date_id
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                GROUP BY d.year, d.month, d.month_name, d.quarter
                ORDER BY d.year, d.month
                """
                datasets['temporal_trends'] = pd.read_sql(query_temporal, conn)
                
                # 4. ANALYSE PAR TOPIC
                query_topics = """
                SELECT 
                    t.topic_name,
                    COUNT(f.review_id) as review_count,
                    AVG(f.star_rating) as avg_rating,
                    AVG(f.ensemble_score) as avg_sentiment,
                    AVG(f.topic_probability) as avg_topic_probability,
                    COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) as positive_reviews,
                    COUNT(CASE WHEN s.sentiment_label = 'negative' THEN 1 END) as negative_reviews,
                    ROUND(COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) * 100.0 / COUNT(*), 2) as positive_percentage,
                    AVG(f.word_count) as avg_word_count
                FROM fact_reviews f
                LEFT JOIN dim_topic t ON f.topic_id = t.topic_id
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                WHERE t.topic_name IS NOT NULL
                GROUP BY t.topic_id, t.topic_name
                ORDER BY review_count DESC
                """
                datasets['topic_analysis'] = pd.read_sql(query_topics, conn)
                
                # 5. PERFORMANCE DES AGENCES
                query_branches = """
                SELECT 
                    b.bank_name,
                    br.branch_name,
                    br.city,
                    br.region,
                    COUNT(f.review_id) as total_reviews,
                    AVG(f.star_rating) as avg_rating,
                    AVG(f.ensemble_score) as avg_sentiment_score,
                    COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) as positive_reviews,
                    COUNT(CASE WHEN s.sentiment_label = 'negative' THEN 1 END) as negative_reviews,
                    ROUND(COUNT(CASE WHEN s.sentiment_label = 'positive' THEN 1 END) * 100.0 / COUNT(*), 2) as positive_percentage,
                    AVG(f.data_quality_score) as avg_data_quality
                FROM fact_reviews f
                LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
                LEFT JOIN dim_branch br ON f.branch_id = br.branch_id
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                WHERE br.branch_name IS NOT NULL
                GROUP BY b.bank_id, b.bank_name, br.branch_id, br.branch_name, br.city, br.region
                HAVING COUNT(f.review_id) >= 3  -- Minimum 3 avis
                ORDER BY positive_percentage DESC, avg_rating DESC
                """
                datasets['branch_performance'] = pd.read_sql(query_branches, conn)
                
                # 6. MATRICE BANQUE-TOPIC
                query_bank_topic = """
                SELECT 
                    b.bank_name,
                    t.topic_name,
                    COUNT(f.review_id) as review_count,
                    AVG(f.ensemble_score) as avg_sentiment,
                    ROUND(COUNT(f.review_id) * 100.0 / 
                          (SELECT COUNT(*) FROM fact_reviews f2 
                           WHERE f2.bank_id = f.bank_id), 2) as topic_percentage_in_bank
                FROM fact_reviews f
                LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
                LEFT JOIN dim_topic t ON f.topic_id = t.topic_id
                WHERE b.bank_name IS NOT NULL AND t.topic_name IS NOT NULL
                GROUP BY b.bank_id, b.bank_name, t.topic_id, t.topic_name
                ORDER BY b.bank_name, review_count DESC
                """
                datasets['bank_topic_matrix'] = pd.read_sql(query_bank_topic, conn)
                
                # 7. DATASET POUR WORD CLOUDS
                query_wordcloud = """
                SELECT 
                    s.sentiment_label,
                    t.topic_name,
                    GROUP_CONCAT(f.review_text, ' ') as combined_text
                FROM fact_reviews f
                LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
                LEFT JOIN dim_topic t ON f.topic_id = t.topic_id
                WHERE f.review_text IS NOT NULL 
                  AND LENGTH(f.review_text) > 10
                GROUP BY s.sentiment_label, t.topic_name
                """
                datasets['wordcloud_data'] = pd.read_sql(query_wordcloud, conn)
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des données: {e}")
            raise
        
        logger.info(f"Extraction terminée: {len(datasets)} datasets créés")
        return datasets
    
    def create_looker_optimized_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """
        Optimise et sauvegarde les datasets pour Looker Studio
        
        Args:
            datasets: Dictionnaire des DataFrames
        """
        logger.info("Optimisation des datasets pour Looker Studio...")
        
        for name, df in datasets.items():
            if df.empty:
                logger.warning(f"Dataset {name} est vide")
                continue
            
            # Optimisations générales
            df_optimized = df.copy()
            
            # Convertir les dates au bon format
            date_columns = ['review_date', 'first_review_date', 'last_review_date']
            for col in date_columns:
                if col in df_optimized.columns:
                    df_optimized[col] = pd.to_datetime(df_optimized[col], errors='coerce')
            
            # Arrondir les valeurs numériques
            numeric_columns = df_optimized.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if 'percentage' in col or 'score' in col or 'rating' in col:
                    df_optimized[col] = df_optimized[col].round(2)
                elif 'count' in col:
                    df_optimized[col] = df_optimized[col].astype('Int64')  # Nullable integer
            
            # Nettoyer les valeurs nulles pour certaines colonnes critiques
            if 'bank_name' in df_optimized.columns:
                df_optimized['bank_name'] = df_optimized['bank_name'].fillna('Inconnu')
            if 'sentiment_label' in df_optimized.columns:
                df_optimized['sentiment_label'] = df_optimized['sentiment_label'].fillna('unknown')
            
            # Ajouter des colonnes calculées utiles pour Looker
            if name == 'main_dataset':
                # Catégories de rating
                df_optimized['rating_category'] = df_optimized['star_rating'].apply(
                    lambda x: 'Excellent (5★)' if x == 5 
                    else 'Bon (4★)' if x == 4
                    else 'Moyen (3★)' if x == 3
                    else 'Mauvais (1-2★)' if x in [1, 2]
                    else 'Non noté'
                )
                
                # Catégories de sentiment avec emojis
                df_optimized['sentiment_emoji'] = df_optimized['sentiment_label'].apply(
                    lambda x: '😊 Positif' if x == 'positive'
                    else '😞 Négatif' if x == 'negative'
                    else '😐 Neutre' if x == 'neutral'
                    else '❓ Inconnu'
                )
                
                # Indicateur de qualité
                df_optimized['quality_level'] = df_optimized['data_quality_score'].apply(
                    lambda x: 'Haute' if x >= 0.8
                    else 'Moyenne' if x >= 0.6
                    else 'Basse' if x >= 0.4
                    else 'Très basse'
                )
                
                # Longueur du texte
                df_optimized['text_length_category'] = df_optimized['word_count'].apply(
                    lambda x: 'Très détaillé (>50 mots)' if x > 50
                    else 'Détaillé (20-50 mots)' if x >= 20
                    else 'Moyen (10-19 mots)' if x >= 10
                    else 'Court (<10 mots)'
                )
            
            # Sauvegarder en CSV pour Looker Studio
            output_path = f"{self.output_dir}/{name}.csv"
            df_optimized.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Dataset sauvegardé: {output_path} ({len(df_optimized)} lignes)")
    
    def create_summary_metrics(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Crée des métriques de résumé pour les KPIs
        
        Args:
            datasets: Dictionnaire des DataFrames
            
        Returns:
            Dictionnaire des métriques clés
        """
        logger.info("Calcul des métriques de résumé...")
        
        metrics = {}
        
        if 'main_dataset' in datasets and not datasets['main_dataset'].empty:
            main_df = datasets['main_dataset']
            
            # Métriques générales
            metrics['total_reviews'] = len(main_df)
            metrics['total_banks'] = main_df['bank_name'].nunique()
            metrics['total_branches'] = main_df['branch_name'].nunique()
            metrics['total_cities'] = main_df['city'].nunique()
            
            # Métriques de qualité
            metrics['avg_rating'] = main_df['star_rating'].mean()
            metrics['avg_sentiment_score'] = main_df['ensemble_score'].mean()
            metrics['avg_data_quality'] = main_df['data_quality_score'].mean()
            
            # Distribution des sentiments
            sentiment_dist = main_df['sentiment_label'].value_counts()
            metrics['positive_percentage'] = (sentiment_dist.get('positive', 0) / len(main_df) * 100)
            metrics['negative_percentage'] = (sentiment_dist.get('negative', 0) / len(main_df) * 100)
            metrics['neutral_percentage'] = (sentiment_dist.get('neutral', 0) / len(main_df) * 100)
            
            # Métriques temporelles
            if 'review_date' in main_df.columns:
                main_df['review_date'] = pd.to_datetime(main_df['review_date'], errors='coerce')
                date_range = main_df['review_date'].max() - main_df['review_date'].min()
                metrics['date_range_days'] = date_range.days if not pd.isna(date_range) else 0
                metrics['reviews_per_month'] = len(main_df) / (metrics['date_range_days'] / 30) if metrics['date_range_days'] > 0 else 0
            
            # Top performers
            if 'bank_summary' in datasets:
                bank_summary = datasets['bank_summary']
                if not bank_summary.empty:
                    metrics['best_bank'] = bank_summary.loc[bank_summary['positive_percentage'].idxmax(), 'bank_name']
                    metrics['most_reviewed_bank'] = bank_summary.loc[bank_summary['total_reviews'].idxmax(), 'bank_name']
        
        # Arrondir les métriques numériques
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = round(value, 2)
        
        # Sauvegarder les métriques
        metrics_path = f"{self.output_dir}/summary_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Métriques sauvegardées: {metrics_path}")
        return metrics
    
    def create_charts_for_presentation(self, datasets: Dict[str, pd.DataFrame]):
        """
        Crée des graphiques statiques pour les présentations
        
        Args:
            datasets: Dictionnaire des DataFrames
        """
        logger.info("Création des graphiques pour présentation...")
        
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance par banque (Barres horizontales)
        if 'bank_summary' in datasets and not datasets['bank_summary'].empty:
            df = datasets['bank_summary'].head(8)  # Top 8 banques
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Graphique 1: Nombre d'avis
            bars1 = ax1.barh(df['bank_name'], df['total_reviews'], color='skyblue')
            ax1.set_xlabel('Nombre d\'avis')
            ax1.set_title('Volume d\'avis par banque')
            ax1.bar_label(bars1, label_type='edge')
            
            # Graphique 2: Pourcentage de sentiment positif
            bars2 = ax2.barh(df['bank_name'], df['positive_percentage'], color='lightgreen')
            ax2.set_xlabel('% d\'avis positifs')
            ax2.set_title('Sentiment positif par banque')
            ax2.bar_label(bars2, fmt='%.1f%%', label_type='edge')
            
            plt.tight_layout()
            plt.savefig(f'{self.charts_dir}/banks_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Évolution temporelle
        if 'temporal_trends' in datasets and not datasets['temporal_trends'].empty:
            df = datasets['temporal_trends']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Créer une colonne date pour l'axe x
            df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            
            # Graphique 1: Volume d'avis
            ax1.plot(df['period'], df['review_count'], marker='o', linewidth=2, markersize=6)
            ax1.set_title('Évolution du volume d\'avis par mois')
            ax1.set_ylabel('Nombre d\'avis')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2: Sentiment moyen
            ax2.plot(df['period'], df['avg_sentiment'], marker='s', color='orange', linewidth=2, markersize=6)
            ax2.set_title('Évolution du sentiment moyen par mois')
            ax2.set_ylabel('Score de sentiment')
            ax2.set_xlabel('Période')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f'{self.charts_dir}/temporal_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Analyse par topic (Graphique radar)
        if 'topic_analysis' in datasets and not datasets['topic_analysis'].empty:
            df = datasets['topic_analysis'].head(8)  # Top 8 topics
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Graphique en barres pour les topics
            bars = ax.bar(range(len(df)), df['positive_percentage'], color='mediumseagreen')
            ax.set_xlabel('Topics')
            ax.set_ylabel('% d\'avis positifs')
            ax.set_title('Sentiment positif par topic')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['topic_name'], rotation=45, ha='right')
            
            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, df['positive_percentage']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{self.charts_dir}/topics_sentiment.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Matrice de corrélation (Heatmap)
        if 'main_dataset' in datasets and not datasets['main_dataset'].empty:
            df = datasets['main_dataset']
            
            # Sélectionner les colonnes numériques pour la corrélation
            corr_columns = ['star_rating', 'ensemble_score', 'sentiment_confidence', 
                          'topic_probability', 'word_count', 'data_quality_score']
            corr_data = df[corr_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Corrélation entre les métriques')
            
            plt.tight_layout()
            plt.savefig(f'{self.charts_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Graphiques créés dans {self.charts_dir}/")
    
    def create_looker_guide(self):
        """Crée un guide pour configurer Looker Studio"""
        
        guide_content = """
# 🚀 GUIDE LOOKER STUDIO - AVIS BANCAIRES

## 📊 DATASETS DISPONIBLES

### 1. main_dataset.csv
**Usage**: Dashboard principal, graphiques détaillés
**Colonnes clés**:
- `bank_name`: Nom de la banque
- `review_date`: Date de l'avis 
- `star_rating`: Note sur 5 étoiles
- `sentiment_label`: Sentiment (positive/negative/neutral)
- `sentiment_emoji`: Sentiment avec emoji
- `topic_name`: Sujet principal
- `ensemble_score`: Score de sentiment (-1 à +1)
- `rating_category`: Catégorie de note
- `quality_level`: Niveau de qualité des données

### 2. bank_summary.csv
**Usage**: Tableaux de bord par banque, KPIs
**Colonnes clés**:
- `total_reviews`: Nombre total d'avis
- `avg_rating`: Note moyenne
- `positive_percentage`: % d'avis positifs
- `negative_percentage`: % d'avis négatifs

### 3. temporal_trends.csv
**Usage**: Graphiques temporels, évolution
**Colonnes clés**:
- `year`, `month`, `month_name`: Période
- `review_count`: Nombre d'avis par mois
- `avg_sentiment`: Sentiment moyen par mois
- `positive_percentage`: % positifs par mois

### 4. topic_analysis.csv
**Usage**: Analyse par sujet, graphiques radar
**Colonnes clés**:
- `topic_name`: Nom du sujet
- `review_count`: Nombre d'avis par sujet
- `positive_percentage`: % positifs par sujet
- `avg_sentiment`: Sentiment moyen par sujet

### 5. branch_performance.csv
**Usage**: Performance des agences, cartes géographiques
**Colonnes clés**:
- `branch_name`: Nom de l'agence
- `city`, `region`: Localisation
- `positive_percentage`: % d'avis positifs
- `avg_rating`: Note moyenne

### 6. bank_topic_matrix.csv
**Usage**: Heatmap banque x sujet
**Colonnes clés**:
- `bank_name`: Banque
- `topic_name`: Sujet
- `review_count`: Nombre d'avis
- `topic_percentage_in_bank`: % du sujet dans la banque

## 📈 DASHBOARDS RECOMMANDÉS

### Dashboard 1: Vue d'ensemble
**Source**: bank_summary.csv + summary_metrics.json
**Graphiques**:
- KPI cards: Total avis, Note moyenne, % Positif
- Barres horizontales: Performance par banque
- Graphique en barres: Distribution des sentiments
- Tableau: Top 10 banques

### Dashboard 2: Analyse temporelle
**Source**: temporal_trends.csv + main_dataset.csv
**Graphiques**:
- Graphique linéaire: Évolution mensuelle du sentiment
- Graphique en aires: Volume d'avis par mois
- Heatmap: Sentiment par jour de la semaine
- Filtre: Sélection de période

### Dashboard 3: Analyse par sujet
**Source**: topic_analysis.csv + bank_topic_matrix.csv
**Graphiques**:
- Graphique radar: Sentiment par topic
- Heatmap: Matrice banque x topic
- Graphique en barres: Volume par topic
- Tableau: Détail des topics

### Dashboard 4: Performance des agences
**Source**: branch_performance.csv + main_dataset.csv
**Graphiques**:
- Carte géographique: Agences par ville
- Tableau: Ranking des agences
- Scatter plot: Note vs % Positif
- Filtre: Banque, Ville, Région

### Dashboard 5: Analyse détaillée
**Source**: main_dataset.csv
**Graphiques**:
- Histogramme: Distribution des notes
- Box plot: Sentiment par banque
- Scatter plot: Longueur vs Sentiment
- Tableau: Avis individuels avec filtres

## 🎨 CONFIGURATION LOOKER STUDIO

### Étape 1: Connexion des données
1. Ouvrir Looker Studio (studio.google.com)
2. Créer une nouvelle source de données
3. Choisir "Upload de fichier" 
4. Télécharger chaque fichier CSV
5. Configurer les types de données:
   - Dates: `review_date`, `first_review_date`, `last_review_date`
   - Métriques: `total_reviews`, `avg_rating`, `positive_percentage`
   - Dimensions: `bank_name`, `sentiment_label`, `topic_name`

### Étape 2: Création des graphiques
**KPI Cards**:
- Métrique: SUM(total_reviews) pour le total d'avis
- Métrique: AVG(avg_rating) pour la note moyenne
- Métrique: AVG(positive_percentage) pour le % positif

**Graphiques en barres**:
- Dimension: bank_name
- Métrique: positive_percentage
- Trier par: Métrique décroissant

**Graphiques temporels**:
- Dimension: Date (review_date)
- Métrique: review_count, avg_sentiment
- Style: Ligne avec marqueurs

**Heatmap**:
- Dimension ligne: bank_name
- Dimension colonne: topic_name
- Métrique: review_count
- Couleur: avg_sentiment

### Étape 3: Formatage et style
**Couleurs recommandées**:
- Positif: #4CAF50 (Vert)
- Négatif: #F44336 (Rouge)
- Neutre: #FF9800 (Orange)
- Principal: #2196F3 (Bleu)

**Filtres à ajouter**:
- Sélecteur de banque
- Sélecteur de période
- Sélecteur de ville/région
- Sélecteur de sentiment

### Étape 4: Interactivité
**Contrôles recommandés**:
- Date range picker pour la période
- Multi-select pour les banques
- Slider pour les notes minimales
- Checkbox pour inclure/exclure certains topics

## 🔍 MÉTRIQUES CALCULÉES LOOKER

### Métriques personnalisées à créer:

```
# Score de satisfaction global
Satisfaction_Score = (Positive_Reviews * 2 + Neutral_Reviews) / Total_Reviews

# Taux de conversion positif
Conversion_Rate = Positive_Reviews / (Positive_Reviews + Negative_Reviews)

# Index de qualité
Quality_Index = (AVG(star_rating) * 20 + AVG(positive_percentage)) / 2

# Tendance sentiment (comparaison période)
Sentiment_Trend = (Current_Period_Sentiment - Previous_Period_Sentiment) / Previous_Period_Sentiment * 100
```

## 📋 CHECKLIST DASHBOARDS

### ✅ Dashboard Vue d'ensemble
- [ ] KPIs généraux (Total avis, Note moyenne, % Positif)
- [ ] Classement des banques
- [ ] Distribution des sentiments (Pie chart)
- [ ] Évolution récente (7 derniers jours)
- [ ] Filtres: Période, Banque

### ✅ Dashboard Analyse temporelle
- [ ] Graphique linéaire: Évolution mensuelle
- [ ] Graphique en aires: Volume par mois
- [ ] Comparaison année sur année
- [ ] Saisonnalité (par trimestre)
- [ ] Filtres: Période, Granularité

### ✅ Dashboard Topics & Sentiments
- [ ] Graphique radar: Sentiment par topic
- [ ] Heatmap: Banque x Topic
- [ ] Top topics positifs/négatifs
- [ ] Nuages de mots (si possible)
- [ ] Filtres: Banque, Topic

### ✅ Dashboard Géographique
- [ ] Carte: Performance par ville
- [ ] Tableau: Ranking des agences
- [ ] Comparaison régionale
- [ ] Détail par agence
- [ ] Filtres: Région, Ville, Banque

### ✅ Dashboard Qualité & Détails
- [ ] Distribution des notes détaillée
- [ ] Analyse de la longueur des avis
- [ ] Score de qualité des données
- [ ] Tableau des avis individuels
- [ ] Filtres: Qualité, Longueur, Sentiment

## 🎯 TIPS POUR LOOKER STUDIO

### Performance:
- Limiter les datasets à 100k lignes max
- Utiliser des agrégations pré-calculées
- Éviter trop de filtres simultanés
- Optimiser les jointures

### Design:
- Utiliser une palette cohérente
- Grouper les graphiques par thème
- Ajouter des titres descriptifs
- Utiliser des icônes pour les KPIs

### Interactivité:
- Connecter les graphiques avec des filtres
- Utiliser des drill-downs logiques
- Ajouter des tooltips informatifs
- Permettre l'export des données

## 📧 SHARING & COLLABORATION

### Permissions recommandées:
- **Viewers**: Équipe direction, managers
- **Editors**: Équipe data, analystes
- **Admin**: Data manager principal

### Planification des rapports:
- Rapport hebdomadaire: Vue d'ensemble
- Rapport mensuel: Analyse complète
- Alertes: Chute significative de sentiment
- Export: Données pour présentations

## 🔧 MAINTENANCE

### Actions régulières:
- [ ] Mise à jour des données (automatique si possible)
- [ ] Vérification de la qualité des données
- [ ] Ajustement des seuils et filtres
- [ ] Optimisation des performances
- [ ] Feedback utilisateurs et améliorations

### Évolutions prévues:
- [ ] Intégration temps réel (API)
- [ ] Machine Learning prédictif
- [ ] Alertes automatiques
- [ ] Export automatisé vers autres outils
"""
        
        guide_path = f"{self.output_dir}/LOOKER_STUDIO_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"Guide Looker Studio créé: {guide_path}")
    
    def generate_looker_config_json(self):
        """Génère un fichier de configuration JSON pour automatiser Looker"""
        
        config = {
            "project_info": {
                "name": "Analyse Avis Bancaires Maroc",
                "description": "Dashboard d'analyse des avis clients pour les banques marocaines",
                "version": "1.0",
                "created_date": datetime.now().isoformat(),
                "data_sources": [
                    "main_dataset.csv",
                    "bank_summary.csv", 
                    "temporal_trends.csv",
                    "topic_analysis.csv",
                    "branch_performance.csv",
                    "bank_topic_matrix.csv"
                ]
            },
            "dashboards": [
                {
                    "name": "Vue d'ensemble",
                    "description": "KPIs généraux et performance par banque",
                    "charts": [
                        {
                            "type": "scorecard",
                            "title": "Total Avis",
                            "metric": "total_reviews",
                            "source": "bank_summary"
                        },
                        {
                            "type": "scorecard", 
                            "title": "Note Moyenne",
                            "metric": "avg_rating",
                            "source": "bank_summary"
                        },
                        {
                            "type": "bar_chart",
                            "title": "Performance par Banque",
                            "dimension": "bank_name",
                            "metric": "positive_percentage",
                            "source": "bank_summary"
                        },
                        {
                            "type": "pie_chart",
                            "title": "Distribution des Sentiments",
                            "dimension": "sentiment_label",
                            "metric": "count",
                            "source": "main_dataset"
                        }
                    ]
                },
                {
                    "name": "Analyse Temporelle",
                    "description": "Évolution dans le temps",
                    "charts": [
                        {
                            "type": "time_series",
                            "title": "Évolution Mensuelle du Sentiment",
                            "date_dimension": "review_date",
                            "metric": "avg_sentiment",
                            "source": "temporal_trends"
                        },
                        {
                            "type": "area_chart",
                            "title": "Volume d'Avis par Mois",
                            "date_dimension": "review_date",
                            "metric": "review_count",
                            "source": "temporal_trends"
                        }
                    ]
                },
                {
                    "name": "Analyse par Topics",
                    "description": "Performance par sujet",
                    "charts": [
                        {
                            "type": "radar_chart",
                            "title": "Sentiment par Topic",
                            "dimension": "topic_name",
                            "metric": "positive_percentage",
                            "source": "topic_analysis"
                        },
                        {
                            "type": "heatmap",
                            "title": "Matrice Banque x Topic",
                            "row_dimension": "bank_name",
                            "column_dimension": "topic_name",
                            "metric": "review_count",
                            "color_metric": "avg_sentiment",
                            "source": "bank_topic_matrix"
                        }
                    ]
                }
            ],
            "filters": [
                {
                    "name": "Période",
                    "type": "date_range",
                    "field": "review_date",
                    "default": "last_6_months"
                },
                {
                    "name": "Banque",
                    "type": "multi_select",
                    "field": "bank_name",
                    "default": "all"
                },
                {
                    "name": "Sentiment",
                    "type": "multi_select", 
                    "field": "sentiment_label",
                    "default": "all"
                },
                {
                    "name": "Ville",
                    "type": "single_select",
                    "field": "city",
                    "default": "all"
                }
            ],
            "style": {
                "colors": {
                    "positive": "#4CAF50",
                    "negative": "#F44336", 
                    "neutral": "#FF9800",
                    "primary": "#2196F3",
                    "secondary": "#9C27B0"
                },
                "fonts": {
                    "header": "Roboto",
                    "body": "Open Sans"
                }
            }
        }
        
        config_path = f"{self.output_dir}/looker_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Configuration Looker générée: {config_path}")
    
    def create_sample_queries_for_looker(self):
        """Crée des requêtes SQL personnalisées pour Looker Studio"""
        
        queries = {
            "kpi_overview": """
            SELECT 
                COUNT(*) as total_reviews,
                AVG(star_rating) as avg_rating,
                AVG(ensemble_score) as avg_sentiment,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_percentage
            FROM main_dataset
            WHERE review_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            """,
            
            "monthly_trend": """
            SELECT 
                DATE_TRUNC(review_date, MONTH) as month,
                COUNT(*) as review_count,
                AVG(ensemble_score) as avg_sentiment,
                AVG(star_rating) as avg_rating
            FROM main_dataset
            GROUP BY month
            ORDER BY month
            """,
            
            "bank_comparison": """
            SELECT 
                bank_name,
                COUNT(*) as total_reviews,
                AVG(star_rating) as avg_rating,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_percentage,
                AVG(word_count) as avg_review_length
            FROM main_dataset
            GROUP BY bank_name
            HAVING COUNT(*) >= 10
            ORDER BY positive_percentage DESC
            """,
            
            "topic_sentiment_analysis": """
            SELECT 
                topic_name,
                sentiment_label,
                COUNT(*) as review_count,
                AVG(topic_probability) as avg_confidence
            FROM main_dataset
            WHERE topic_name IS NOT NULL
            GROUP BY topic_name, sentiment_label
            ORDER BY topic_name, sentiment_label
            """,
            
            "regional_performance": """
            SELECT 
                region,
                city,
                COUNT(*) as total_reviews,
                AVG(star_rating) as avg_rating,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_percentage
            FROM main_dataset
            WHERE region IS NOT NULL AND city IS NOT NULL
            GROUP BY region, city
            HAVING COUNT(*) >= 5
            ORDER BY region, positive_percentage DESC
            """
        }
        
        queries_path = f"{self.output_dir}/custom_queries.sql"
        with open(queries_path, 'w', encoding='utf-8') as f:
            f.write("-- REQUÊTES PERSONNALISÉES POUR LOOKER STUDIO\n")
            f.write("-- Analyse des Avis Bancaires\n\n")
            
            for name, query in queries.items():
                f.write(f"-- {name.upper()}\n")
                f.write(query)
                f.write("\n\n")
        
        logger.info(f"Requêtes personnalisées créées: {queries_path}")
    
    def run_complete_looker_prep(self) -> Dict:
        """
        Exécute la préparation complète pour Looker Studio
        
        Returns:
            Dictionnaire avec le résumé des fichiers créés
        """
        logger.info("=== PRÉPARATION COMPLÈTE LOOKER STUDIO ===")
        
        try:
            # 1. Extraire les données
            datasets = self.extract_data_for_looker()
            
            # 2. Optimiser et sauvegarder les datasets
            self.create_looker_optimized_datasets(datasets)
            
            # 3. Créer les métriques de résumé
            metrics = self.create_summary_metrics(datasets)
            
            # 4. Créer les graphiques pour présentation
            self.create_charts_for_presentation(datasets)
            
            # 5. Créer le guide Looker Studio
            self.create_looker_guide()
            
            # 6. Générer la configuration JSON
            self.generate_looker_config_json()
            
            # 7. Créer les requêtes personnalisées
            self.create_sample_queries_for_looker()
            
            # Résumé des fichiers créés
            summary = {
                "datasets_created": len(datasets),
                "total_reviews": metrics.get('total_reviews', 0),
                "total_banks": metrics.get('total_banks', 0),
                "avg_sentiment": metrics.get('avg_sentiment_score', 0),
                "files_created": [
                    f"{self.output_dir}/main_dataset.csv",
                    f"{self.output_dir}/bank_summary.csv",
                    f"{self.output_dir}/temporal_trends.csv",
                    f"{self.output_dir}/topic_analysis.csv",
                    f"{self.output_dir}/branch_performance.csv",
                    f"{self.output_dir}/bank_topic_matrix.csv",
                    f"{self.output_dir}/summary_metrics.json",
                    f"{self.output_dir}/looker_config.json",
                    f"{self.output_dir}/LOOKER_STUDIO_GUIDE.md",
                    f"{self.output_dir}/custom_queries.sql"
                ],
                "charts_created": [
                    f"{self.charts_dir}/banks_performance.png",
                    f"{self.charts_dir}/temporal_evolution.png", 
                    f"{self.charts_dir}/topics_sentiment.png",
                    f"{self.charts_dir}/correlation_matrix.png"
                ]
            }
            
            logger.info("=== PRÉPARATION LOOKER STUDIO TERMINÉE ===")
            return summary
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation Looker: {e}")
            raise

def main():
    """Fonction principale de préparation Looker Studio"""
    
    # Vérifier que la base de données existe
    db_path = 'data/bank_reviews_dw.db'
    if not os.path.exists(db_path):
        logger.error(f"Base de données non trouvée: {db_path}")
        logger.error("Veuillez d'abord exécuter l'étape 5 (data_warehouse.py)")
        return
    
    # Initialiser la préparation Looker
    looker_prep = LookerStudioDataPrep(db_path)
    
    try:
        # Exécuter la préparation complète
        summary = looker_prep.run_complete_looker_prep()
        
        # Afficher les résultats
        print(f"\n✅ Préparation Looker Studio terminée!")
        print(f"📊 Statistiques:")
        print(f"  - Datasets créés: {summary['datasets_created']}")
        print(f"  - Total avis: {summary['total_reviews']}")
        print(f"  - Banques analysées: {summary['total_banks']}")
        print(f"  - Sentiment moyen: {summary['avg_sentiment']:.3f}")
        
        print(f"\n💾 Fichiers de données créés:")
        for file_path in summary['files_created']:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path}")
        
        print(f"\n📈 Graphiques créés:")
        for chart_path in summary['charts_created']:
            if os.path.exists(chart_path):
                print(f"  ✓ {chart_path}")
            else:
                print(f"  ✗ {chart_path}")
        
        print(f"\n🚀 Prochaines étapes:")
        print(f"  1. Aller sur studio.google.com")
        print(f"  2. Créer une nouvelle source de données")
        print(f"  3. Télécharger les fichiers CSV du dossier looker_data/")
        print(f"  4. Suivre le guide: looker_data/LOOKER_STUDIO_GUIDE.md")
        print(f"  5. Utiliser la configuration: looker_data/looker_config.json")
        
        print(f"\n📋 URLs Looker Studio recommandées:")
        print(f"  - Créer un rapport: https://studio.google.com/reports/create")
        print(f"  - Galerie de templates: https://studio.google.com/gallery")
        print(f"  - Documentation: https://support.google.com/looker-studio/")
        
    except Exception as e:
        logger.error(f"Erreur dans main: {e}")
        raise

if __name__ == "__main__":
    main()