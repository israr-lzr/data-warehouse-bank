#!/usr/bin/env python3
"""
Script de correction complète de la base de données
Corrige tous les problèmes identifiés
"""

import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine, text
import json

def fix_fact_reviews_structure(db_path='data/bank_reviews_dw.db'):
    """Corrige la structure de fact_reviews en ajoutant les colonnes manquantes"""
    print("\n🔧 CORRECTION DE LA STRUCTURE fact_reviews")
    print("="*50)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier les colonnes existantes
        cursor.execute("PRAGMA table_info(fact_reviews)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Colonnes à ajouter si elles manquent
        columns_to_add = [
            ("topic_id", "INTEGER"),
            ("star_rating", "INTEGER"),
            ("review_text_cleaned", "TEXT"),
            ("review_date", "DATE"),
            ("textblob_polarity", "REAL"),
            ("textblob_subjectivity", "REAL"),
            ("vader_compound", "REAL"),
            ("vader_positive", "REAL"),
            ("vader_neutral", "REAL"),
            ("vader_negative", "REAL"),
            ("custom_sentiment_score", "REAL"),
            ("ensemble_score", "REAL"),
            ("sentiment_confidence", "REAL"),
            ("word_count", "INTEGER"),
            ("char_count", "INTEGER"),
            ("sentence_count", "INTEGER"),
            ("positive_words_count", "INTEGER"),
            ("negative_words_count", "INTEGER"),
            ("topic_probability", "REAL"),
            ("has_exclamation", "BOOLEAN DEFAULT 0"),
            ("has_question", "BOOLEAN DEFAULT 0"),
            ("has_caps", "BOOLEAN DEFAULT 0"),
            ("original_length", "INTEGER"),
            ("cleaned_length", "INTEGER"),
            ("scraped_at", "TIMESTAMP"),
            ("data_quality_score", "REAL")
        ]
        
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE fact_reviews ADD COLUMN {column_name} {column_type}")
                    print(f"  ✅ Ajouté: {column_name}")
                except Exception as e:
                    print(f"  ⚠️  Erreur {column_name}: {e}")
        
        conn.commit()
        conn.close()
        print("✅ Structure fact_reviews corrigée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction de structure: {e}")
        return False

def populate_missing_data_fixed(db_path='data/bank_reviews_dw.db'):
    """Peuple les tables avec des données de base (version corrigée)"""
    print("\n📊 PEUPLEMENT DES DONNÉES DE BASE")
    print("="*50)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier et peupler dim_sentiment
        cursor.execute("SELECT COUNT(*) FROM dim_sentiment")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📝 Peuplement de dim_sentiment...")
            sentiments = [
                ('positive', 'Avis exprimant une satisfaction', 0.1, 1.0),
                ('negative', 'Avis exprimant une insatisfaction', -1.0, -0.1),
                ('neutral', 'Avis neutres ou mitigés', -0.1, 0.1),
                ('unknown', 'Sentiment non déterminé', None, None)
            ]
            
            for sentiment, desc, min_score, max_score in sentiments:
                cursor.execute("""
                    INSERT OR IGNORE INTO dim_sentiment 
                    (sentiment_label, sentiment_description, sentiment_score_min, sentiment_score_max) 
                    VALUES (?, ?, ?, ?)
                """, (sentiment, desc, min_score, max_score))
            
            print("  ✅ dim_sentiment peuplée")
        
        # Vérifier et peupler dim_topic
        cursor.execute("SELECT COUNT(*) FROM dim_topic")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📝 Peuplement de dim_topic...")
            topics = [
                ('service_client', 'Qualité de l\'accueil et du service client'),
                ('attente_temps', 'Temps d\'attente et rapidité du service'),
                ('frais_tarifs', 'Frais bancaires et tarification'),
                ('services_bancaires', 'Services et produits bancaires'),
                ('technologie_digital', 'Services numériques et technologie'),
                ('localisation_agence', 'Emplacement et accessibilité des agences'),
                ('problemes_reclamations', 'Problèmes techniques et réclamations'),
                ('satisfaction_generale', 'Satisfaction générale et recommandations'),
                ('autres', 'Autres sujets non classifiés')
            ]
            
            for topic_name, description in topics:
                cursor.execute("""
                    INSERT OR IGNORE INTO dim_topic (topic_name, topic_description) 
                    VALUES (?, ?)
                """, (topic_name, description))
            
            print("  ✅ dim_topic peuplée")
        
        # Peupler dim_date avec quelques dates de base
        cursor.execute("SELECT COUNT(*) FROM dim_date")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📝 Peuplement de dim_date...")
            from datetime import datetime, timedelta, date
            
            # Créer des dates pour les 2 dernières années
            start_date = date(2023, 1, 1)
            end_date = date(2024, 12, 31)
            current_date = start_date
            
            french_months = [
                'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'
            ]
            
            french_days = [
                'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'
            ]
            
            while current_date <= end_date:
                year = current_date.year
                month = current_date.month
                day = current_date.day
                quarter = (month - 1) // 3 + 1
                day_of_week = current_date.weekday() + 1
                week_of_year = current_date.isocalendar()[1]
                is_weekend = day_of_week in [6, 7]
                
                month_name = french_months[month - 1]
                day_name = french_days[day_of_week - 1]
                
                cursor.execute("""
                    INSERT OR IGNORE INTO dim_date 
                    (full_date, year, quarter, month, month_name, day, 
                     day_of_week, day_name, week_of_year, is_weekend) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (current_date, year, quarter, month, month_name, day,
                      day_of_week, day_name, week_of_year, is_weekend))
                
                current_date += timedelta(days=1)
            
            print("  ✅ dim_date peuplée")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du peuplement: {e}")
        return False

def load_csv_data_to_warehouse(db_path='data/bank_reviews_dw.db'):
    """Charge les données CSV enrichies dans la base de données"""
    print("\n📥 CHARGEMENT DES DONNÉES CSV")
    print("="*50)
    
    try:
        # Trouver le fichier CSV le plus récent avec toutes les données
        data_files = []
        for file in os.listdir('data/'):
            if file.startswith('bank_reviews_topics_') and file.endswith('.csv'):
                data_files.append(f'data/{file}')
        
        if not data_files:
            # Essayer avec les autres fichiers
            for file in os.listdir('data/'):
                if file.startswith('bank_reviews_sentiment_') and file.endswith('.csv'):
                    data_files.append(f'data/{file}')
        
        if not data_files:
            print("❌ Aucun fichier CSV trouvé")
            return False
        
        # Prendre le plus récent
        latest_file = max(data_files, key=os.path.getmtime)
        print(f"📄 Chargement du fichier: {latest_file}")
        
        df = pd.read_csv(latest_file)
        print(f"📊 Données chargées: {len(df)} lignes")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Compter les lignes actuelles dans fact_reviews
        cursor.execute("SELECT COUNT(*) FROM fact_reviews")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️  fact_reviews contient déjà {existing_count} lignes")
            response = input("Voulez-vous les remplacer? (y/N): ")
            if response.lower() != 'y':
                print("Annulé par l'utilisateur")
                return False
            
            # Vider la table
            cursor.execute("DELETE FROM fact_reviews")
            print("🗑️  Anciennes données supprimées")
        
        # Obtenir les mappings des dimensions
        bank_mapping = {}
        cursor.execute("SELECT bank_id, bank_name FROM dim_bank")
        for row in cursor.fetchall():
            bank_mapping[row[1]] = row[0]
        
        sentiment_mapping = {}
        cursor.execute("SELECT sentiment_id, sentiment_label FROM dim_sentiment")
        for row in cursor.fetchall():
            sentiment_mapping[row[1]] = row[0]
        
        topic_mapping = {}
        cursor.execute("SELECT topic_id, topic_name FROM dim_topic")
        for row in cursor.fetchall():
            topic_mapping[row[1]] = row[0]
        
        date_mapping = {}
        cursor.execute("SELECT date_id, full_date FROM dim_date")
        for row in cursor.fetchall():
            date_mapping[str(row[1])] = row[0]
        
        # Insérer les données ligne par ligne
        success_count = 0
        for idx, row in df.iterrows():
            try:
                # Récupérer les IDs
                bank_id = bank_mapping.get(row.get('bank_name'))
                sentiment_id = sentiment_mapping.get(row.get('sentiment_label', 'unknown'), sentiment_mapping.get('unknown'))
                topic_id = topic_mapping.get(row.get('topic_label'), topic_mapping.get('autres'))
                
                # Date de l'avis (essayer plusieurs colonnes)
                review_date = None
                for date_col in ['review_date', 'scraped_at']:
                    if date_col in row and pd.notna(row[date_col]):
                        try:
                            review_date = pd.to_datetime(row[date_col]).date()
                            break
                        except:
                            continue
                
                date_id = None
                if review_date:
                    date_id = date_mapping.get(str(review_date))
                
                # Calculer un score de qualité basique
                quality_score = 0.5
                if pd.notna(row.get('review_text')) and len(str(row.get('review_text', ''))) > 10:
                    quality_score += 0.3
                if pd.notna(row.get('rating')) and row.get('rating') > 0:
                    quality_score += 0.2
                
                # Insérer dans fact_reviews
                cursor.execute("""
                    INSERT INTO fact_reviews (
                        bank_id, sentiment_id, topic_id, date_id,
                        star_rating, review_text, review_text_cleaned,
                        review_date, ensemble_score, sentiment_confidence,
                        word_count, char_count, positive_words_count, negative_words_count,
                        topic_probability, has_exclamation, has_question, has_caps,
                        data_quality_score, textblob_polarity, vader_compound, custom_sentiment_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bank_id,
                    sentiment_id, 
                    topic_id,
                    date_id,
                    int(row.get('rating', 0)) if pd.notna(row.get('rating')) else None,
                    str(row.get('review_text', '')),
                    str(row.get('cleaned_text', '')),
                    review_date,
                    float(row.get('ensemble_score', 0)) if pd.notna(row.get('ensemble_score')) else None,
                    float(row.get('sentiment_confidence', 0)) if pd.notna(row.get('sentiment_confidence')) else None,
                    int(row.get('word_count', 0)) if pd.notna(row.get('word_count')) else None,
                    int(row.get('char_count', 0)) if pd.notna(row.get('char_count')) else None,
                    int(row.get('positive_words_count', 0)) if pd.notna(row.get('positive_words_count')) else None,
                    int(row.get('negative_words_count', 0)) if pd.notna(row.get('negative_words_count')) else None,
                    float(row.get('topic_probability', 0)) if pd.notna(row.get('topic_probability')) else None,
                    bool(row.get('has_exclamation', False)),
                    bool(row.get('has_question', False)),
                    bool(row.get('has_caps', False)),
                    quality_score,
                    float(row.get('textblob_polarity', 0)) if pd.notna(row.get('textblob_polarity')) else None,
                    float(row.get('vader_compound', 0)) if pd.notna(row.get('vader_compound')) else None,
                    float(row.get('custom_sentiment_score', 0)) if pd.notna(row.get('custom_sentiment_score')) else None
                ))
                
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"  📊 Traité: {success_count}/{len(df)}")
                
            except Exception as e:
                print(f"  ⚠️  Erreur ligne {idx}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        print(f"✅ Données chargées: {success_count}/{len(df)} lignes")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False

def create_looker_datasets_direct(db_path='data/bank_reviews_dw.db'):
    """Crée les datasets Looker directement depuis la base corrigée"""
    print("\n📁 CRÉATION DES DATASETS LOOKER")
    print("="*50)
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        os.makedirs('looker_data', exist_ok=True)
        
        # Dataset principal
        query_main = """
        SELECT 
            f.review_id,
            b.bank_name,
            f.star_rating,
            s.sentiment_label,
            t.topic_name,
            f.ensemble_score,
            f.sentiment_confidence,
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
            f.review_date,
            d.year,
            d.month,
            d.month_name,
            d.quarter,
            f.textblob_polarity,
            f.vader_compound,
            f.custom_sentiment_score
        FROM fact_reviews f
        LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
        LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
        LEFT JOIN dim_topic t ON f.topic_id = t.topic_id
        LEFT JOIN dim_date d ON f.date_id = d.date_id
        WHERE f.review_text IS NOT NULL
        ORDER BY f.review_id
        """
        
        df_main = pd.read_sql(query_main, engine)
        
        if not df_main.empty:
            # Ajouter des colonnes calculées
            df_main['rating_category'] = df_main['star_rating'].apply(
                lambda x: 'Excellent (5★)' if x == 5 
                else 'Bon (4★)' if x == 4
                else 'Moyen (3★)' if x == 3
                else 'Mauvais (1-2★)' if pd.notna(x) and x in [1, 2]
                else 'Non noté'
            )
            
            df_main['sentiment_emoji'] = df_main['sentiment_label'].apply(
                lambda x: '😊 Positif' if x == 'positive'
                else '😞 Négatif' if x == 'negative'  
                else '😐 Neutre' if x == 'neutral'
                else '❓ Inconnu'
            )
            
            df_main.to_csv('looker_data/main_dataset.csv', index=False, encoding='utf-8')
            print(f"✅ Dataset principal: {len(df_main)} lignes")
        
        # Dataset banques
        query_banks = """
        SELECT 
            b.bank_name,
            COUNT(f.review_id) as total_reviews,
            AVG(f.star_rating) as avg_rating,
            AVG(f.ensemble_score) as avg_sentiment_score,
            SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_reviews,
            SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_reviews,
            SUM(CASE WHEN s.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_reviews,
            ROUND(SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as positive_percentage,
            AVG(f.word_count) as avg_word_count,
            AVG(f.data_quality_score) as avg_data_quality
        FROM fact_reviews f
        LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
        LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
        WHERE b.bank_name IS NOT NULL
        GROUP BY b.bank_id, b.bank_name
        ORDER BY total_reviews DESC
        """
        
        df_banks = pd.read_sql(query_banks, engine)
        if not df_banks.empty:
            df_banks.to_csv('looker_data/bank_summary.csv', index=False, encoding='utf-8')
            print(f"✅ Résumé banques: {len(df_banks)} banques")
        
        # Métriques générales
        metrics = {
            "total_reviews": len(df_main),
            "total_banks": df_main['bank_name'].nunique() if not df_main.empty else 0,
            "avg_rating": round(df_main['star_rating'].mean(), 2) if not df_main.empty else 0,
            "avg_sentiment": round(df_main['ensemble_score'].mean(), 3) if not df_main.empty else 0,
            "positive_percentage": round((df_main['sentiment_label'] == 'positive').sum() / len(df_main) * 100, 2) if not df_main.empty else 0
        }
        
        with open('looker_data/summary_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Métriques générées")
        return True
        
    except Exception as e:
        print(f"❌ Erreur création datasets: {e}")
        return False

def main():
    """Fonction principale de correction complète"""
    print("🚀 CORRECTION COMPLÈTE DE LA BASE DE DONNÉES")
    print("="*60)
    
    db_path = 'data/bank_reviews_dw.db'
    
    # 1. Créer les tables manquantes
    from fix_database import create_missing_tables
    create_missing_tables(db_path)
    
    # 2. Corriger la structure de fact_reviews  
    fix_fact_reviews_structure(db_path)
    
    # 3. Peupler les données de base
    populate_missing_data_fixed(db_path)
    
    # 4. Charger les données CSV dans la base
    load_csv_data_to_warehouse(db_path)
    
    # 5. Créer les datasets Looker
    create_looker_datasets_direct(db_path)
    
    print("\n✅ CORRECTION TERMINÉE!")
    print("="*60)
    print("📁 Fichiers créés dans looker_data/:")
    print("  - main_dataset.csv") 
    print("  - bank_summary.csv")
    print("  - summary_metrics.json")
    print("\n🚀 Vous pouvez maintenant utiliser Looker Studio!")

if __name__ == "__main__":
    main()