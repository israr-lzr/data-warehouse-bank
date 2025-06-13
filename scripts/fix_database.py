#!/usr/bin/env python3
"""
Script de diagnostic et correction de la base de données
Corrige les problèmes de tables manquantes
"""

import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine, text

def check_database_structure(db_path='data/bank_reviews_dw.db'):
    """Vérifie la structure de la base de données"""
    print("🔍 DIAGNOSTIC DE LA BASE DE DONNÉES")
    print("="*50)
    
    if not os.path.exists(db_path):
        print(f"❌ Base de données non trouvée: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Lister toutes les tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"📊 Tables trouvées: {len(tables)}")
        
        expected_tables = [
            'fact_reviews', 'dim_bank', 'dim_branch', 
            'dim_date', 'dim_sentiment', 'dim_topic', 'dim_reviewer'
        ]
        
        existing_tables = [table[0] for table in tables]
        
        for table in expected_tables:
            if table in existing_tables:
                # Compter les lignes
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✅ {table}: {count} lignes")
            else:
                print(f"  ❌ {table}: MANQUANTE")
        
        # Vérifier la structure de fact_reviews
        if 'fact_reviews' in existing_tables:
            cursor.execute("PRAGMA table_info(fact_reviews)")
            columns = cursor.fetchall()
            print(f"\n📋 Colonnes de fact_reviews: {len(columns)}")
            
            expected_columns = [
                'review_id', 'bank_id', 'branch_id', 'date_id', 
                'sentiment_id', 'topic_id', 'star_rating', 'review_text'
            ]
            
            existing_columns = [col[1] for col in columns]
            
            for col in expected_columns:
                if col in existing_columns:
                    print(f"  ✅ {col}")
                else:
                    print(f"  ❌ {col}: MANQUANTE")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du diagnostic: {e}")
        return False

def create_missing_tables(db_path='data/bank_reviews_dw.db'):
    """Crée les tables manquantes"""
    print("\n🔧 CRÉATION DES TABLES MANQUANTES")
    print("="*50)
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # SQL pour créer dim_topic si elle n'existe pas
        create_dim_topic = """
        CREATE TABLE IF NOT EXISTS dim_topic (
            topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_name VARCHAR(100) NOT NULL UNIQUE,
            topic_description TEXT,
            topic_keywords TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now'))
        );
        """
        
        # SQL pour créer dim_reviewer si elle n'existe pas
        create_dim_reviewer = """
        CREATE TABLE IF NOT EXISTS dim_reviewer (
            reviewer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            reviewer_name VARCHAR(100),
            reviewer_hash VARCHAR(64) UNIQUE,
            review_count INTEGER DEFAULT 1,
            first_review_date DATE,
            last_review_date DATE,
            avg_rating DECIMAL(3, 2),
            created_at TIMESTAMP DEFAULT (datetime('now')),
            updated_at TIMESTAMP DEFAULT (datetime('now'))
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_dim_topic))
            conn.execute(text(create_dim_reviewer))
            conn.commit()
            
        print("✅ Tables créées avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables: {e}")
        return False

def populate_missing_data(db_path='data/bank_reviews_dw.db'):
    """Peuple les tables avec des données de base si elles sont vides"""
    print("\n📊 PEUPLEMENT DES DONNÉES DE BASE")
    print("="*50)
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        with engine.connect() as conn:
            # Vérifier et peupler dim_sentiment
            result = conn.execute(text("SELECT COUNT(*) FROM dim_sentiment"))
            count = result.fetchone()[0]
            
            if count == 0:
                print("📝 Peuplement de dim_sentiment...")
                sentiments = [
                    ('positive', 'Avis exprimant une satisfaction', 0.1, 1.0),
                    ('negative', 'Avis exprimant une insatisfaction', -1.0, -0.1),
                    ('neutral', 'Avis neutres ou mitigés', -0.1, 0.1),
                    ('unknown', 'Sentiment non déterminé', None, None)
                ]
                
                for sentiment, desc, min_score, max_score in sentiments:
                    conn.execute(text("""
                        INSERT OR IGNORE INTO dim_sentiment 
                        (sentiment_label, sentiment_description, sentiment_score_min, sentiment_score_max) 
                        VALUES (?, ?, ?, ?)
                    """), (sentiment, desc, min_score, max_score))
                
                print("  ✅ dim_sentiment peuplée")
            
            # Vérifier et peupler dim_topic avec des topics par défaut
            result = conn.execute(text("SELECT COUNT(*) FROM dim_topic"))
            count = result.fetchone()[0]
            
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
                    conn.execute(text("""
                        INSERT OR IGNORE INTO dim_topic (topic_name, topic_description) 
                        VALUES (?, ?)
                    """), (topic_name, description))
                
                print("  ✅ dim_topic peuplée")
            
            conn.commit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du peuplement: {e}")
        return False

def create_simple_looker_datasets(db_path='data/bank_reviews_dw.db'):
    """Crée des datasets simplifiés pour Looker en cas de problème avec les jointures"""
    print("\n📁 CRÉATION DE DATASETS SIMPLIFIÉS")
    print("="*50)
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Créer le répertoire de sortie
        os.makedirs('looker_data', exist_ok=True)
        
        # Dataset principal simplifié (sans jointures complexes)
        query_simple = """
        SELECT 
            f.review_id,
            f.star_rating,
            f.review_text,
            f.ensemble_score,
            f.sentiment_confidence,
            f.word_count,
            f.char_count,
            f.positive_words_count,
            f.negative_words_count,
            f.has_exclamation,
            f.has_question,
            f.has_caps,
            f.data_quality_score,
            f.textblob_polarity,
            f.vader_compound,
            f.custom_sentiment_score,
            f.topic_probability,
            f.review_date,
            b.bank_name,
            s.sentiment_label
        FROM fact_reviews f
        LEFT JOIN dim_bank b ON f.bank_id = b.bank_id
        LEFT JOIN dim_sentiment s ON f.sentiment_id = s.sentiment_id
        WHERE f.star_rating IS NOT NULL
        ORDER BY f.review_id
        """
        
        df_main = pd.read_sql(query_simple, engine)
        
        if not df_main.empty:
            # Ajouter des colonnes calculées
            df_main['rating_category'] = df_main['star_rating'].apply(
                lambda x: 'Excellent (5★)' if x == 5 
                else 'Bon (4★)' if x == 4
                else 'Moyen (3★)' if x == 3
                else 'Mauvais (1-2★)' if x in [1, 2]
                else 'Non noté'
            )
            
            df_main['sentiment_emoji'] = df_main['sentiment_label'].apply(
                lambda x: '😊 Positif' if x == 'positive'
                else '😞 Négatif' if x == 'negative'
                else '😐 Neutre' if x == 'neutral'
                else '❓ Inconnu'
            )
            
            # Sauvegarder
            df_main.to_csv('looker_data/main_dataset_simple.csv', index=False, encoding='utf-8')
            print(f"✅ Dataset principal créé: {len(df_main)} lignes")
        
        # Dataset résumé par banque
        query_bank_summary = """
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
        
        df_banks = pd.read_sql(query_bank_summary, engine)
        
        if not df_banks.empty:
            df_banks.to_csv('looker_data/bank_summary_simple.csv', index=False, encoding='utf-8')
            print(f"✅ Résumé banques créé: {len(df_banks)} banques")
        
        # Créer des métriques de base
        metrics = {
            "total_reviews": len(df_main),
            "total_banks": df_main['bank_name'].nunique() if not df_main.empty else 0,
            "avg_rating": df_main['star_rating'].mean() if not df_main.empty else 0,
            "avg_sentiment": df_main['ensemble_score'].mean() if not df_main.empty else 0,
            "positive_percentage": (df_main['sentiment_label'] == 'positive').sum() / len(df_main) * 100 if not df_main.empty else 0
        }
        
        import json
        with open('looker_data/summary_metrics_simple.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print("✅ Métriques de base créées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des datasets: {e}")
        return False

def main():
    """Fonction principale de diagnostic et correction"""
    print("🚀 DIAGNOSTIC ET CORRECTION DE LA BASE DE DONNÉES")
    print("="*60)
    
    db_path = 'data/bank_reviews_dw.db'
    
    # 1. Diagnostic
    if not check_database_structure(db_path):
        print("❌ Échec du diagnostic")
        return
    
    # 2. Créer les tables manquantes
    if not create_missing_tables(db_path):
        print("❌ Échec de la création des tables")
        return
    
    # 3. Peupler les données de base
    if not populate_missing_data(db_path):
        print("❌ Échec du peuplement des données")
        return
    
    # 4. Vérification finale
    print("\n🔍 VÉRIFICATION FINALE")
    print("="*50)
    check_database_structure(db_path)
    
    # 5. Créer des datasets simplifiés pour Looker
    if not create_simple_looker_datasets(db_path):
        print("❌ Échec de la création des datasets")
        return
    
    print("\n✅ CORRECTION TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    print("📁 Fichiers créés dans looker_data/:")
    print("  - main_dataset_simple.csv")
    print("  - bank_summary_simple.csv") 
    print("  - summary_metrics_simple.json")
    print("\n🚀 Vous pouvez maintenant utiliser ces fichiers dans Looker Studio")
    print("📋 Ou réessayer: python scripts/looker_setup.py")

if __name__ == "__main__":
    main()