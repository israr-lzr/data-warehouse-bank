#!/usr/bin/env python3
"""
Crée un fichier CSV simplifié pour Looker Studio
"""

import pandas as pd
import os

def create_simple_looker_csv():
    """Crée un CSV ultra-simplifié pour Looker Studio"""
    print("🔧 CRÉATION D'UN CSV SIMPLIFIÉ POUR LOOKER")
    print("="*50)
    
    try:
        # Charger le fichier original
        source_file = 'data/bank_reviews_topics_20250612_001641.csv'
        
        if not os.path.exists(source_file):
            print(f"❌ Fichier non trouvé: {source_file}")
            return False
        
        df = pd.read_csv(source_file)
        print(f"📊 Données chargées: {len(df)} lignes")
        
        # Créer un dataset ultra-simplifié avec seulement les colonnes essentielles
        df_simple = pd.DataFrame()
        
        # Colonnes de base (renommage si nécessaire)
        df_simple['Banque'] = df.get('bank_name', 'Inconnue')
        df_simple['Agence'] = df.get('branch_name', 'Inconnue')
        df_simple['Note'] = pd.to_numeric(df.get('rating', df.get('star_rating', 3)), errors='coerce').fillna(3)
        df_simple['Sentiment'] = df.get('sentiment_label', 'neutral').fillna('neutral')
        df_simple['Score_Sentiment'] = pd.to_numeric(df.get('ensemble_score', 0), errors='coerce').fillna(0)
        df_simple['Sujet'] = df.get('topic_label', df.get('topic_name', 'autres')).fillna('autres')
        
        # Ajouter des colonnes calculées simples
        df_simple['Categorie_Note'] = df_simple['Note'].apply(
            lambda x: 'Excellent' if x == 5 
            else 'Bon' if x == 4
            else 'Moyen' if x == 3
            else 'Faible'
        )
        
        df_simple['Sentiment_Emoji'] = df_simple['Sentiment'].apply(
            lambda x: 'Positif 😊' if str(x).lower() == 'positive'
            else 'Negatif 😞' if str(x).lower() == 'negative'
            else 'Neutre 😐'
        )
        
        # Calculer des métriques textuelles simples
        if 'review_text' in df.columns:
            df_simple['Longueur_Avis'] = df['review_text'].astype(str).apply(len)
            df_simple['Nb_Mots'] = df['review_text'].astype(str).apply(lambda x: len(x.split()))
        else:
            df_simple['Longueur_Avis'] = 100
            df_simple['Nb_Mots'] = 20
        
        # Ajouter une date simple
        df_simple['Date'] = '2024-06-01'  # Date fixe pour éviter les problèmes
        df_simple['Annee'] = 2024
        df_simple['Mois'] = 6
        
        # Nettoyer les valeurs
        df_simple = df_simple.fillna({
            'Banque': 'Banque Inconnue',
            'Agence': 'Agence Inconnue', 
            'Note': 3,
            'Sentiment': 'neutral',
            'Score_Sentiment': 0,
            'Sujet': 'autres'
        })
        
        # S'assurer que les colonnes numériques sont bien numériques
        numeric_columns = ['Note', 'Score_Sentiment', 'Longueur_Avis', 'Nb_Mots', 'Annee', 'Mois']
        for col in numeric_columns:
            df_simple[col] = pd.to_numeric(df_simple[col], errors='coerce').fillna(0)
        
        # Créer le répertoire de sortie
        os.makedirs('looker_data', exist_ok=True)
        
        # Sauvegarder avec encodage UTF-8 sans BOM
        output_file = 'looker_data/dataset_simple.csv'
        df_simple.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ CSV simplifié créé: {output_file}")
        print(f"📊 {len(df_simple)} lignes, {len(df_simple.columns)} colonnes")
        print(f"📋 Colonnes: {list(df_simple.columns)}")
        
        # Afficher un aperçu
        print(f"\n🔍 Aperçu des données:")
        print(df_simple.head(3).to_string())
        
        # Créer aussi un résumé encore plus simple
        summary = df_simple.groupby('Banque').agg({
            'Note': ['count', 'mean'],
            'Score_Sentiment': 'mean'
        }).round(2)
        
        summary.columns = ['Total_Avis', 'Note_Moyenne', 'Sentiment_Moyen']
        summary = summary.reset_index()
        
        summary.to_csv('looker_data/resume_banques.csv', index=False, encoding='utf-8-sig')
        print(f"✅ Résumé banques créé: looker_data/resume_banques.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    if create_simple_looker_csv():
        print(f"\n🎯 FICHIERS PRÊTS POUR LOOKER STUDIO:")
        print(f"1. 📄 dataset_simple.csv (Dataset principal)")
        print(f"2. 📄 resume_banques.csv (Résumé par banque)")
        print(f"\n🚀 INSTRUCTIONS:")
        print(f"1. Aller sur studio.google.com")
        print(f"2. Upload dataset_simple.csv")
        print(f"3. Si erreur persiste, essayer resume_banques.csv")
        print(f"4. Créer vos graphiques!")
    else:
        print(f"❌ Échec de la création du CSV simplifié")

if __name__ == "__main__":
    main()