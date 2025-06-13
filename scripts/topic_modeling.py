#!/usr/bin/env python3
"""
Topic Modeling (extraction de sujets) des avis bancaires
Étape 4 du projet Data Warehouse
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Machine Learning et Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Traitement de texte
import nltk
from wordcloud import WordCloud
from collections import Counter
import re

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/topic_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankTopicModeling:
    """Classe pour l'extraction de sujets des avis bancaires"""
    
    def __init__(self, n_topics: int = 8, random_state: int = 42):
        """
        Initialise le topic modeling
        
        Args:
            n_topics: Nombre de sujets à extraire
            random_state: Graine pour la reproductibilité
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.banking_topics = self.define_banking_topics()
        
    def define_banking_topics(self) -> Dict[str, List[str]]:
        """
        Définit les thèmes bancaires attendus avec mots-clés
        
        Returns:
            Dictionnaire des thèmes et mots-clés associés
        """
        topics = {
            'service_client': [
                'accueil', 'personnel', 'conseiller', 'equipe', 'staff', 'employe',
                'service', 'aide', 'assistance', 'conseil', 'information', 'explication',
                'competent', 'professionnel', 'aimable', 'souriant', 'sympathique',
                'impoli', 'desagreable', 'incompetent'
            ],
            'attente_temps': [
                'attente', 'queue', 'file', 'long', 'lent', 'rapide', 'vite',
                'temps', 'minute', 'heure', 'patience', 'delai', 'duree',
                'urgent', 'immediat', 'retard'
            ],
            'frais_tarifs': [
                'frais', 'tarif', 'prix', 'cout', 'cher', 'expensive', 'gratuit',
                'commission', 'fee', 'facturation', 'montant', 'argent',
                'economique', 'avantageux', 'raisonnable', 'abusif'
            ],
            'services_bancaires': [
                'compte', 'carte', 'credit', 'pret', 'virement', 'depot', 'retrait',
                'banque', 'banking', 'financier', 'transaction', 'operation',
                'ouverture', 'fermeture', 'activation', 'blocage'
            ],
            'technologie_digital': [
                'application', 'app', 'site', 'internet', 'online', 'digital',
                'mobile', 'telephone', 'ordinateur', 'electronique', 'automatique',
                'distributeur', 'atm', 'terminal', 'technologie', 'systeme'
            ],
            'localisation_agence': [
                'agence', 'bureau', 'succursale', 'local', 'batiment', 'lieu',
                'adresse', 'quartier', 'ville', 'proche', 'loin', 'accessible',
                'parking', 'transport', 'metro', 'bus', 'voiture'
            ],
            'problemes_reclamations': [
                'probleme', 'erreur', 'bug', 'panne', 'dysfonctionnement',
                'reclamation', 'plainte', 'litige', 'conflit', 'resolution',
                'solution', 'reparation', 'correction', 'amelioration'
            ],
            'satisfaction_generale': [
                'satisfait', 'content', 'heureux', 'ravi', 'plaisir', 'recommande',
                'excellent', 'parfait', 'super', 'top', 'genial', 'formidable',
                'decevant', 'decu', 'insatisfait', 'mauvais', 'horrible', 'nul'
            ]
        }
        
        logger.info(f"Thèmes bancaires définis: {list(topics.keys())}")
        return topics
    
    def load_sentiment_data(self, file_path: str) -> pd.DataFrame:
        """Charge les données avec sentiment"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Données chargées: {len(df)} avis avec sentiment")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def prepare_text_for_modeling(self, df: pd.DataFrame, 
                                  text_column: str = 'processed_text') -> List[str]:
        """
        Prépare le texte pour le topic modeling
        
        Args:
            df: DataFrame avec les avis
            text_column: Colonne contenant le texte traité
            
        Returns:
            Liste des textes nettoyés
        """
        documents = []
        
        for text in df[text_column]:
            if pd.isna(text) or not text:
                documents.append("")
            else:
                # Nettoyer davantage si nécessaire
                clean_text = str(text).strip()
                # Filtrer les documents trop courts
                if len(clean_text.split()) >= 3:
                    documents.append(clean_text)
                else:
                    documents.append("")
        
        # Filtrer les documents vides
        non_empty_docs = [doc for doc in documents if doc]
        logger.info(f"Documents préparés: {len(non_empty_docs)} sur {len(documents)}")
        
        return documents, non_empty_docs
    
    def create_vectorizers(self, documents: List[str]) -> Tuple[TfidfVectorizer, CountVectorizer]:
        """
        Crée les vectoriseurs TF-IDF et Count
        
        Args:
            documents: Liste des documents
            
        Returns:
            Tuple des vectoriseurs TF-IDF et Count
        """
        # Mots vides bancaires spécifiques
        banking_stopwords = [
            'banque', 'bank', 'agence', 'branch', 'client', 'customer',
            'service', 'avis', 'review', 'experience', 'fois', 'jour',
            'toujours', 'souvent', 'parfois', 'jamais', 'aller', 'venir'
        ]
        
        # TF-IDF pour LDA et analyse générale
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=banking_stopwords
        )
        
        # Count Vectorizer pour LDA (recommandé)
        count_vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=banking_stopwords
        )
        
        logger.info("Vectoriseurs créés")
        return tfidf_vectorizer, count_vectorizer
    
    def find_optimal_topics(self, count_matrix, max_topics: int = 15) -> int:
        """
        Trouve le nombre optimal de topics avec la perplexité
        
        Args:
            count_matrix: Matrice de comptage
            max_topics: Nombre maximum de topics à tester
            
        Returns:
            Nombre optimal de topics
        """
        logger.info("Recherche du nombre optimal de topics...")
        
        perplexities = []
        topic_range = range(2, max_topics + 1)
        
        for n_topics in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.random_state,
                max_iter=10,
                learning_method='batch'
            )
            lda.fit(count_matrix)
            perplexity = lda.perplexity(count_matrix)
            perplexities.append(perplexity)
            logger.info(f"Topics: {n_topics}, Perplexité: {perplexity:.2f}")
        
        # Trouver le coude dans la courbe de perplexité
        # Utiliser la différence seconde pour détecter le changement de courbure
        if len(perplexities) >= 3:
            second_diffs = []
            for i in range(1, len(perplexities) - 1):
                second_diff = perplexities[i-1] - 2*perplexities[i] + perplexities[i+1]
                second_diffs.append(second_diff)
            
            if second_diffs:
                optimal_idx = np.argmax(second_diffs) + 2  # +2 car on commence à 2
                optimal_topics = list(topic_range)[optimal_idx]
            else:
                optimal_topics = self.n_topics
        else:
            optimal_topics = self.n_topics
        
        logger.info(f"Nombre optimal de topics: {optimal_topics}")
        return optimal_topics
    
    def run_lda_analysis(self, count_matrix, count_vectorizer, 
                        optimal_topics: int = None) -> LatentDirichletAllocation:
        """
        Exécute l'analyse LDA
        
        Args:
            count_matrix: Matrice de comptage
            count_vectorizer: Vectoriseur utilisé
            optimal_topics: Nombre de topics (optionnel)
            
        Returns:
            Modèle LDA entraîné
        """
        n_topics = optimal_topics or self.n_topics
        
        logger.info(f"Exécution LDA avec {n_topics} topics...")
        
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=20,
            learning_method='batch',
            learning_offset=50.0,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        lda_model.fit(count_matrix)
        
        # Calculer la perplexité finale
        perplexity = lda_model.perplexity(count_matrix)
        logger.info(f"LDA terminé. Perplexité finale: {perplexity:.2f}")
        
        return lda_model
    
    def run_nmf_analysis(self, tfidf_matrix, tfidf_vectorizer,
                        optimal_topics: int = None) -> NMF:
        """
        Exécute l'analyse NMF (alternative à LDA)
        
        Args:
            tfidf_matrix: Matrice TF-IDF
            tfidf_vectorizer: Vectoriseur TF-IDF
            optimal_topics: Nombre de topics
            
        Returns:
            Modèle NMF entraîné
        """
        n_topics = optimal_topics or self.n_topics
        
        logger.info(f"Exécution NMF avec {n_topics} topics...")
        
        nmf_model = NMF(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=200,
            alpha=0.1,
            l1_ratio=0.5
        )
        
        nmf_model.fit(tfidf_matrix)
        
        logger.info("NMF terminé")
        return nmf_model
    
    def extract_topic_words(self, model, vectorizer, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Extrait les mots principaux de chaque topic
        
        Args:
            model: Modèle LDA ou NMF
            vectorizer: Vectoriseur utilisé
            n_words: Nombre de mots par topic
            
        Returns:
            Dictionnaire {topic_id: [mots]}
        """
        feature_names = vectorizer.get_feature_names_out()
        topic_words = {}
        
        for topic_idx, topic in enumerate(model.components_):
            # Indices des mots avec les plus hauts scores
            top_word_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_words[topic_idx] = top_words
        
        logger.info(f"Mots extraits pour {len(topic_words)} topics")
        return topic_words
    
    def assign_topic_labels(self, topic_words: Dict[int, List[str]]) -> Dict[int, str]:
        """
        Assigne des labels sémantiques aux topics basés sur les mots-clés
        
        Args:
            topic_words: Dictionnaire des mots par topic
            
        Returns:
            Dictionnaire {topic_id: label}
        """
        topic_labels = {}
        
        for topic_id, words in topic_words.items():
            # Calculer la similarité avec les thèmes prédéfinis
            best_theme = "autres"
            best_score = 0
            
            for theme_name, theme_keywords in self.banking_topics.items():
                # Compter les intersections
                intersection = set(words) & set(theme_keywords)
                score = len(intersection)
                
                if score > best_score:
                    best_score = score
                    best_theme = theme_name
            
            # Si pas assez de correspondance, utiliser les mots principaux
            if best_score < 2:
                best_theme = f"theme_{' '.join(words[:2])}"
            
            topic_labels[topic_id] = best_theme
        
        logger.info(f"Labels assignés: {topic_labels}")
        return topic_labels
    
    def assign_documents_to_topics(self, model, document_matrix,
                                  threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        Assigne chaque document à son topic principal
        
        Args:
            model: Modèle LDA/NMF
            document_matrix: Matrice des documents
            threshold: Seuil minimum de probabilité
            
        Returns:
            Liste de (topic_id, probabilité) pour chaque document
        """
        # Calculer la distribution des topics pour chaque document
        doc_topic_matrix = model.transform(document_matrix)
        
        assignments = []
        for doc_topics in doc_topic_matrix:
            if doc_topics is None or len(doc_topics) == 0:
                assignments.append((-1, 0.0))
                continue
            max_topic_id = np.argmax(doc_topics)
            max_prob = doc_topics[max_topic_id]
            
            # Assigner seulement si au-dessus du seuil
            if max_prob >= threshold:
                assignments.append((max_topic_id, max_prob))
            else:
                assignments.append((-1, max_prob))  # -1 = topic mixte/incertain
        
        logger.info(f"Documents assignés aux topics")
        return assignments
    
    def analyze_topics_by_sentiment(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la distribution des topics par sentiment
        
        Args:
            df: DataFrame avec topics et sentiments
            
        Returns:
            Statistiques croisées
        """
        if 'topic_id' not in df.columns or 'sentiment_label' not in df.columns:
            logger.warning("Colonnes topic_id ou sentiment_label manquantes")
            return {}
        
        cross_analysis = {}
        
        # Distribution des sentiments par topic
        for topic_id in df['topic_id'].unique():
            if topic_id == -1:  # Skip topics incertains
                continue
                
            topic_data = df[df['topic_id'] == topic_id]
            sentiment_dist = topic_data['sentiment_label'].value_counts()
            
            cross_analysis[topic_id] = {
                'total_reviews': len(topic_data),
                'sentiment_distribution': sentiment_dist.to_dict(),
                'avg_sentiment_score': topic_data['ensemble_score'].mean(),
                'avg_rating': topic_data['rating'].mean() if 'rating' in topic_data.columns else None
            }
        
        return cross_analysis
    
    def analyze_topics_by_bank(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la distribution des topics par banque
        
        Args:
            df: DataFrame avec topics et banques
            
        Returns:
            Statistiques par banque
        """
        bank_analysis = {}
        
        for bank in df['bank_name'].unique():
            bank_data = df[df['bank_name'] == bank]
            topic_dist = bank_data['topic_id'].value_counts()
            
            bank_analysis[bank] = {
                'total_reviews': len(bank_data),
                'topic_distribution': topic_dist.to_dict(),
                'main_topics': topic_dist.head(3).index.tolist()
            }
        
        return bank_analysis
    
    def create_wordclouds(self, topic_words: Dict[int, List[str]], 
                         topic_labels: Dict[int, str], output_dir: str = "plots/"):
        """
        Crée des nuages de mots pour chaque topic
        
        Args:
            topic_words: Mots par topic
            topic_labels: Labels des topics
            output_dir: Répertoire de sortie
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for topic_id, words in topic_words.items():
            if len(words) < 3:
                continue
                
            # Créer le texte pour le wordcloud
            text = ' '.join(words * 3)  # Répéter pour plus de visibilité
            
            # Générer le wordcloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=50,
                colormap='viridis'
            ).generate(text)
            
            # Sauvegarder
            label = topic_labels.get(topic_id, f"topic_{topic_id}")
            filename = f"{output_dir}wordcloud_topic_{topic_id}_{label}.png"
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Topic {topic_id}: {label}")
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Nuages de mots sauvegardés dans {output_dir}")
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Exécute l'analyse complète de topic modeling
        
        Args:
            df: DataFrame avec les avis et sentiments
            
        Returns:
            Tuple (DataFrame enrichi, statistiques)
        """
        logger.info("Début de l'analyse complète de topic modeling...")
        
        # Préparer les textes
        documents, non_empty_docs = self.prepare_text_for_modeling(df)
        
        if len(non_empty_docs) < 10:
            logger.error("Pas assez de documents pour le topic modeling")
            return df, {}
        
        # Créer les vectoriseurs
        tfidf_vectorizer, count_vectorizer = self.create_vectorizers(non_empty_docs)
        
        # Vectoriser
        tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_docs)
        count_matrix = count_vectorizer.fit_transform(non_empty_docs)
        
        # Trouver le nombre optimal de topics
        optimal_topics = self.find_optimal_topics(count_matrix)
        
        # Exécuter LDA
        lda_model = self.run_lda_analysis(count_matrix, count_vectorizer, optimal_topics)
        
        # Extraire les mots des topics
        topic_words = self.extract_topic_words(lda_model, count_vectorizer)
        
        # Assigner des labels
        topic_labels = self.assign_topic_labels(topic_words)
        
        # Assigner les documents aux topics (sur tous les documents, y compris vides)
        # Pour cela, on doit transformer tous les documents
        all_count_matrix = count_vectorizer.transform(documents)
        assignments = self.assign_documents_to_topics(lda_model, all_count_matrix)
        
        # Ajouter les résultats au DataFrame
        df_topics = df.copy()
        df_topics['topic_id'] = [assign[0] for assign in assignments]
        df_topics['topic_probability'] = [assign[1] for assign in assignments]
        df_topics['topic_label'] = df_topics['topic_id'].map(topic_labels)
        
        # Créer les nuages de mots
        self.create_wordclouds(topic_words, topic_labels)
        
        # Analyser les résultats
        sentiment_analysis = self.analyze_topics_by_sentiment(df_topics)
        bank_analysis = self.analyze_topics_by_bank(df_topics)
        
        # Compiler les statistiques
        stats = {
            'topic_words': topic_words,
            'topic_labels': topic_labels,
            'sentiment_analysis': sentiment_analysis,
            'bank_analysis': bank_analysis,
            'model_info': {
                'n_topics': optimal_topics,
                'n_documents': len(non_empty_docs),
                'perplexity': lda_model.perplexity(count_matrix)
            }
        }
        
        logger.info("Analyse de topic modeling terminée")
        return df_topics, stats
    
    def save_topic_data(self, df: pd.DataFrame, stats: Dict, output_path: str = None) -> str:
        """Sauvegarde les données avec topics"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/bank_reviews_topics_{timestamp}.csv"
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Sauvegarder aussi les statistiques
        stats_path = output_path.replace('.csv', '_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=== STATISTIQUES TOPIC MODELING ===\n\n")
            
            f.write("Mots principaux par topic:\n")
            for topic_id, words in stats['topic_words'].items():
                label = stats['topic_labels'][topic_id]
                f.write(f"Topic {topic_id} ({label}): {', '.join(words)}\n")
            
            f.write(f"\nInformations modèle:\n")
            f.write(f"Nombre de topics: {stats['model_info']['n_topics']}\n")
            f.write(f"Documents analysés: {stats['model_info']['n_documents']}\n")
            f.write(f"Perplexité: {stats['model_info']['perplexity']:.2f}\n")
        
        logger.info(f"Données avec topics sauvegardées: {output_path}")
        return output_path
    
    def print_topic_report(self, stats: Dict):
        """Affiche un rapport détaillé du topic modeling"""
        
        logger.info("=== RAPPORT TOPIC MODELING ===")
        
        # Topics identifiés
        logger.info(f"Nombre de topics: {stats['model_info']['n_topics']}")
        logger.info(f"Documents analysés: {stats['model_info']['n_documents']}")
        
        logger.info("\nTopics identifiés:")
        for topic_id, words in stats['topic_words'].items():
            label = stats['topic_labels'][topic_id]
            logger.info(f"  Topic {topic_id} ({label}): {', '.join(words[:5])}")
        
        # Analyse par sentiment
        if stats['sentiment_analysis']:
            logger.info("\nSentiment par topic:")
            for topic_id, data in stats['sentiment_analysis'].items():
                label = stats['topic_labels'][topic_id]
                pos_pct = data['sentiment_distribution'].get('positive', 0) / data['total_reviews'] * 100
                logger.info(f"  {label}: {pos_pct:.1f}% positifs ({data['total_reviews']} avis)")

def main():
    """Fonction principale de topic modeling"""
    
    topic_analyzer = BankTopicModeling(n_topics=8)
    
    try:
        # Charger les données avec sentiment
        input_file = "data/bank_reviews_sentiment_20250611_234415.csv"  # Ajuster le nom
        df = topic_analyzer.load_sentiment_data(input_file)
        
        # Analyser les topics
        df_topics, stats = topic_analyzer.run_complete_analysis(df)
        
        # Sauvegarder
        output_file = topic_analyzer.save_topic_data(df_topics, stats)
        
        # Afficher le rapport
        topic_analyzer.print_topic_report(stats)
        
        print(f"\n✅ Topic modeling terminé!")
        print(f"📊 {len(df_topics)} avis avec {stats['model_info']['n_topics']} topics")
        print(f"💾 Fichier sauvegardé: {output_file}")
        print(f"🎨 Nuages de mots créés dans plots/")
        
        # Aperçu des résultats
        print(f"\n🔎 Aperçu des topics:")
        topic_cols = ['bank_name', 'sentiment_label', 'topic_label', 'topic_probability']
        print(df_topics[topic_cols].head(10))
        
    except Exception as e:
        logger.error(f"Erreur lors du topic modeling: {e}")
        raise

if __name__ == "__main__":
    main()