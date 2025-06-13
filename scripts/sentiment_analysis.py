#!/usr/bin/env python3
"""
Analyse de sentiment des avis bancaires
Étape 3 du projet Data Warehouse
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Bibliothèques pour l'analyse de sentiment
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankSentimentAnalyzer:
    """Classe pour l'analyse de sentiment des avis bancaires"""
    
    def __init__(self):
        """Initialise l'analyseur de sentiment"""
        self.setup_sentiment_tools()
        self.french_sentiment_lexicon = self.create_french_banking_lexicon()
        
    def setup_sentiment_tools(self):
        """Configure les outils d'analyse de sentiment"""
        try:
            # Télécharger VADER si nécessaire
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            # Initialiser VADER (fonctionne bien même pour le français)
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("Outils d'analyse de sentiment initialisés")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    def create_french_banking_lexicon(self) -> Dict[str, float]:
        """
        Crée un lexique de sentiment spécifique au domaine bancaire français
        
        Returns:
            Dictionnaire avec mots et scores de sentiment (-1 à +1)
        """
        lexicon = {
            # Mots très positifs
            'excellent': 0.9, 'parfait': 0.9, 'formidable': 0.8, 'magnifique': 0.8,
            'extraordinaire': 0.8, 'fantastique': 0.8, 'merveilleux': 0.8,
            
            # Mots positifs
            'bon': 0.6, 'bien': 0.6, 'super': 0.7, 'genial': 0.7, 'top': 0.7,
            'efficace': 0.6, 'rapide': 0.5, 'professionnel': 0.6, 'competent': 0.6,
            'aimable': 0.5, 'souriant': 0.5, 'accueillant': 0.5, 'sympathique': 0.5,
            'satisfait': 0.6, 'content': 0.6, 'heureux': 0.7, 'ravi': 0.7,
            'recommande': 0.6, 'conseille': 0.5, 'qualite': 0.5,
            
            # Mots neutres légèrement positifs
            'correct': 0.2, 'normal': 0.1, 'standard': 0.1, 'classique': 0.1,
            
            # Mots neutres légèrement négatifs  
            'moyen': -0.2, 'ordinaire': -0.1, 'banal': -0.2,
            
            # Mots négatifs
            'mauvais': -0.6, 'mal': -0.6, 'nul': -0.8, 'horrible': -0.8,
            'catastrophique': -0.9, 'deplorable': -0.8, 'inadmissible': -0.7,
            'lent': -0.5, 'long': -0.4, 'attente': -0.4, 'queue': -0.4,
            'incompetent': -0.7, 'desagreable': -0.6, 'impoli': -0.6,
            'mecontentement': -0.6, 'insatisfait': -0.6, 'decu': -0.6,
            'probleme': -0.5, 'difficulte': -0.4, 'complique': -0.4,
            'refuse': -0.5, 'rejete': -0.6, 'bloque': -0.5,
            
            # Mots très négatifs
            'scandaleux': -0.9, 'inacceptable': -0.8, 'honteux': -0.8,
            'arnaque': -0.9, 'escroquerie': -0.9, 'vol': -0.9,
            'fuite': -0.7, 'eviter': -0.7, 'deconsille': -0.7,
            
            # Termes bancaires spécifiques
            'frais': -0.3, 'commission': -0.2, 'cout': -0.2, 'cher': -0.4,
            'gratuit': 0.4, 'avantageux': 0.5, 'interessant': 0.4,
            'pret': 0.2, 'credit': 0.1, 'compte': 0.1, 'carte': 0.1,
            'virement': 0.1, 'depot': 0.1, 'retrait': 0.1,
            'conseiller': 0.3, 'directeur': 0.2, 'guichet': 0.1,
            'agence': 0.1, 'bureau': 0.1, 'accueil': 0.2,
            
            # Émotions
            'colere': -0.7, 'frustration': -0.6, 'enervement': -0.6,
            'stress': -0.5, 'anxiete': -0.5, 'peur': -0.6,
            'confiance': 0.6, 'securite': 0.5, 'tranquillite': 0.5,
            'satisfaction': 0.6, 'plaisir': 0.6, 'joie': 0.7,
            
            # Négations (pour ajustement contextuel)
            'pas': 0.0, 'aucun': 0.0, 'jamais': 0.0, 'personne': 0.0,
            'rien': 0.0, 'ni': 0.0, 'sans': 0.0, 'non': 0.0
        }
        
        logger.info(f"Lexique bancaire créé avec {len(lexicon)} termes")
        return lexicon
    
    def load_cleaned_data(self, file_path: str) -> pd.DataFrame:
        """Charge les données nettoyées"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Données chargées: {len(df)} avis")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyse de sentiment avec TextBlob
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire avec polarité et subjectivité
        """
        if pd.isna(text) or not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyse de sentiment avec VADER
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire avec scores VADER
        """
        if pd.isna(text) or not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        except Exception:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    def custom_french_sentiment(self, text: str, tokens: List[str] = None) -> Dict[str, float]:
        """
        Analyse de sentiment personnalisée pour le français bancaire
        
        Args:
            text: Texte original
            tokens: Tokens prétraités (optionnel)
            
        Returns:
            Dictionnaire avec score personnalisé
        """
        if pd.isna(text) or not text:
            return {'custom_score': 0.0, 'positive_words': 0, 'negative_words': 0}
        
        # Utiliser les tokens si fournis, sinon tokeniser simplement
        if tokens and isinstance(tokens, str):
            words = tokens.split()
        elif tokens and isinstance(tokens, list):
            words = tokens
        else:
            words = text.lower().split()
        
        positive_score = 0.0
        negative_score = 0.0
        positive_count = 0
        negative_count = 0
        
        # Analyser chaque mot
        for i, word in enumerate(words):
            if word in self.french_sentiment_lexicon:
                score = self.french_sentiment_lexicon[word]
                
                # Gestion des négations (mot précédent)
                negation = False
                if i > 0 and words[i-1] in ['pas', 'non', 'aucun', 'jamais', 'sans']:
                    negation = True
                
                # Inverser le score si négation
                if negation:
                    score = -score
                
                # Accumuler les scores
                if score > 0:
                    positive_score += score
                    positive_count += 1
                elif score < 0:
                    negative_score += abs(score)
                    negative_count += 1
        
        # Calculer le score final
        total_words = positive_count + negative_count
        if total_words == 0:
            custom_score = 0.0
        else:
            custom_score = (positive_score - negative_score) / total_words
        
        return {
            'custom_score': custom_score,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def rating_based_sentiment(self, rating: float) -> str:
        """
        Convertit la note étoile en sentiment
        
        Args:
            rating: Note sur 5 étoiles
            
        Returns:
            Label de sentiment
        """
        if pd.isna(rating) or rating == 0:
            return 'unknown'
        elif rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'
    
    def classify_sentiment(self, polarity: float, threshold_pos: float = 0.1, 
                         threshold_neg: float = -0.1) -> str:
        """
        Classifie le sentiment basé sur un score de polarité
        
        Args:
            polarity: Score de polarité
            threshold_pos: Seuil pour positif
            threshold_neg: Seuil pour négatif
            
        Returns:
            Label de sentiment
        """
        if polarity >= threshold_pos:
            return 'positive'
        elif polarity <= threshold_neg:
            return 'negative'
        else:
            return 'neutral'
    
    def ensemble_sentiment(self, textblob_pol: float, vader_comp: float, 
                          custom_score: float, rating: float = None) -> Dict[str, any]:
        """
        Combine plusieurs scores pour un sentiment final
        
        Args:
            textblob_pol: Score TextBlob
            vader_comp: Score VADER compound
            custom_score: Score personnalisé
            rating: Note étoile (optionnel)
            
        Returns:
            Dictionnaire avec sentiment final et scores
        """
        # Pondération des différents scores
        weights = {
            'textblob': 0.25,
            'vader': 0.35,
            'custom': 0.40
        }
        
        # Score ensemble
        ensemble_score = (
            weights['textblob'] * textblob_pol +
            weights['vader'] * vader_comp +
            weights['custom'] * custom_score
        )
        
        # Classification avec seuils ajustés
        ensemble_label = self.classify_sentiment(ensemble_score, 0.15, -0.15)
        
        # Vérification avec la note si disponible
        if rating and not pd.isna(rating):
            rating_label = self.rating_based_sentiment(rating)
            
            # Si forte contradiction, ajuster
            if (ensemble_label == 'positive' and rating_label == 'negative') or \
               (ensemble_label == 'negative' and rating_label == 'positive'):
                # Donner plus de poids à la note
                if rating >= 4:
                    ensemble_label = 'positive'
                elif rating <= 2:
                    ensemble_label = 'negative'
        
        # Calcul de la confiance
        confidence = min(abs(ensemble_score) * 2, 1.0)
        
        return {
            'sentiment': ensemble_label,
            'ensemble_score': ensemble_score,
            'confidence': confidence
        }
    
    def analyze_dataset_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Début de l'analyse de sentiment...")
        
        df_sentiment = df.copy()
        
        # Listes pour stocker les résultats
        textblob_results = []
        vader_results = []
        custom_results = []
        final_sentiments = []
        
        # Analyser chaque avis
        for idx, row in df_sentiment.iterrows():
            if idx % 100 == 0:
                logger.info(f"Analyse: {idx}/{len(df_sentiment)} avis")
            
            text = row.get('cleaned_text', '')
            tokens = row.get('tokens', '')
            rating = row.get('rating', None)
            
            # TextBlob
            tb_result = self.textblob_sentiment(text)
            textblob_results.append(tb_result)
            
            # VADER
            vader_result = self.vader_sentiment(text)
            vader_results.append(vader_result)
            
            # Sentiment personnalisé
            custom_result = self.custom_french_sentiment(text, tokens)
            custom_results.append(custom_result)
            
            # Sentiment final ensemble
            final_result = self.ensemble_sentiment(
                tb_result['polarity'],
                vader_result['compound'],
                custom_result['custom_score'],
                rating
            )
            final_sentiments.append(final_result)
        
        # Ajouter les résultats au DataFrame
        # TextBlob
        df_sentiment['textblob_polarity'] = [r['polarity'] for r in textblob_results]
        df_sentiment['textblob_subjectivity'] = [r['subjectivity'] for r in textblob_results]
        
        # VADER
        df_sentiment['vader_compound'] = [r['compound'] for r in vader_results]
        df_sentiment['vader_positive'] = [r['pos'] for r in vader_results]
        df_sentiment['vader_neutral'] = [r['neu'] for r in vader_results]
        df_sentiment['vader_negative'] = [r['neg'] for r in vader_results]
        
        # Sentiment personnalisé
        df_sentiment['custom_sentiment_score'] = [r['custom_score'] for r in custom_results]
        df_sentiment['positive_words_count'] = [r['positive_words'] for r in custom_results]
        df_sentiment['negative_words_count'] = [r['negative_words'] for r in custom_results]
        
        # Sentiment final
        df_sentiment['sentiment_label'] = [r['sentiment'] for r in final_sentiments]
        df_sentiment['ensemble_score'] = [r['ensemble_score'] for r in final_sentiments]
        df_sentiment['sentiment_confidence'] = [r['confidence'] for r in final_sentiments]
        
        # Ajouter des métriques basées sur les notes
        if 'rating' in df_sentiment.columns:
            df_sentiment['rating_sentiment'] = df_sentiment['rating'].apply(self.rating_based_sentiment)
        
        logger.info("Analyse de sentiment terminée")
        
        return df_sentiment
    
    def generate_sentiment_statistics(self, df: pd.DataFrame) -> Dict:
        """Génère des statistiques sur l'analyse de sentiment"""
        
        stats = {}
        
        # Distribution des sentiments
        sentiment_dist = df['sentiment_label'].value_counts()
        stats['sentiment_distribution'] = sentiment_dist.to_dict()
        stats['sentiment_percentages'] = (sentiment_dist / len(df) * 100).round(2).to_dict()
        
        # Scores moyens par sentiment
        stats['avg_scores'] = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            subset = df[df['sentiment_label'] == sentiment]
            if len(subset) > 0:
                stats['avg_scores'][sentiment] = {
                    'ensemble_score': subset['ensemble_score'].mean(),
                    'confidence': subset['sentiment_confidence'].mean(),
                    'textblob': subset['textblob_polarity'].mean(),
                    'vader': subset['vader_compound'].mean(),
                    'custom': subset['custom_sentiment_score'].mean()
                }
        
        # Corrélation avec les notes
        if 'rating' in df.columns:
            correlations = {}
            score_cols = ['ensemble_score', 'textblob_polarity', 'vader_compound', 'custom_sentiment_score']
            for col in score_cols:
                correlations[col] = df[col].corr(df['rating'])
            stats['rating_correlations'] = correlations
        
        # Statistiques par banque
        stats['by_bank'] = {}
        for bank in df['bank_name'].unique():
            bank_data = df[df['bank_name'] == bank]
            bank_sentiment = bank_data['sentiment_label'].value_counts()
            stats['by_bank'][bank] = {
                'total_reviews': len(bank_data),
                'sentiment_distribution': bank_sentiment.to_dict(),
                'avg_sentiment_score': bank_data['ensemble_score'].mean(),
                'avg_rating': bank_data['rating'].mean() if 'rating' in bank_data.columns else None
            }
        
        return stats
    
    def save_sentiment_data(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Sauvegarde les données avec analyse de sentiment"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/bank_reviews_sentiment_{timestamp}.csv"
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Données avec sentiment sauvegardées: {output_path}")
        
        return output_path
    
    def print_sentiment_report(self, stats: Dict):
        """Affiche un rapport détaillé de l'analyse de sentiment"""
        
        logger.info("=== RAPPORT D'ANALYSE DE SENTIMENT ===")
        
        # Distribution générale
        logger.info("Distribution des sentiments:")
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = stats['sentiment_percentages'][sentiment]
            logger.info(f"  {sentiment}: {count} avis ({percentage}%)")
        
        # Corrélations avec les notes
        if 'rating_correlations' in stats:
            logger.info("\nCorrection avec les notes étoiles:")
            for method, corr in stats['rating_correlations'].items():
                logger.info(f"  {method}: {corr:.3f}")
        
        # Performance par banque
        logger.info("\nSentiment moyen par banque:")
        for bank, data in stats['by_bank'].items():
            avg_score = data['avg_sentiment_score']
            total = data['total_reviews']
            pos_pct = data['sentiment_distribution'].get('positive', 0) / total * 100
            logger.info(f"  {bank}: {avg_score:.3f} ({pos_pct:.1f}% positifs, {total} avis)")

def main():
    """Fonction principale d'analyse de sentiment"""
    
    analyzer = BankSentimentAnalyzer()
    
    try:
        # ✅ Fichier généré automatiquement le 11 juin 2025
        input_file = "data/bank_reviews_cleaned_20250611_234258.csv"

        
        # Charger les données nettoyées
        df = analyzer.load_cleaned_data(input_file)
        
        # Analyser le sentiment
        df_sentiment = analyzer.analyze_dataset_sentiment(df)
        
        # Générer les statistiques
        stats = analyzer.generate_sentiment_statistics(df_sentiment)
        
        # Sauvegarder
        output_file = analyzer.save_sentiment_data(df_sentiment)
        
        # Afficher le rapport
        analyzer.print_sentiment_report(stats)
        
        print(f"\n✅ Analyse de sentiment terminée!")
        print(f"📊 {len(df_sentiment)} avis analysés")
        print(f"💾 Fichier sauvegardé: {output_file}")
        
        # Aperçu des résultats
        print(f"\n🔎 Aperçu des résultats:")
        sentiment_cols = ['bank_name', 'rating', 'sentiment_label', 'ensemble_score', 'sentiment_confidence']
        print(df_sentiment[sentiment_cols].head(10))
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {e}")
        raise
if __name__ == "__main__":
    print("▶️ Lancement du script d'analyse de sentiment...")  # DEBUG
    main()

    
    