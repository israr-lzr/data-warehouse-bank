import pandas as pd
import numpy as np
import re
import string
import logging
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Bibliothèques NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

import spacy
from textblob import TextBlob

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankReviewsCleaner:
    """Classe pour nettoyer et prétraiter les avis bancaires"""
    
    def __init__(self, language='french'):
        """
        Initialise le nettoyeur
        
        Args:
            language: Langue principale des textes ('french' ou 'arabic')
        """
        self.language = language
        self.setup_nlp_tools()
        
    def setup_nlp_tools(self):
        """Configure les outils NLP nécessaires"""
        try:
            # Télécharger les ressources NLTK si nécessaire
            nltk_downloads = [
                'punkt', 'stopwords', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words'
            ]
            
            for resource in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            # Initialiser les outils
            if self.language == 'french':
                self.stop_words = set(stopwords.words('french'))
                self.stemmer = SnowballStemmer('french')
                
                # Ajouter des stopwords spécifiques au domaine bancaire
                banking_stopwords = {
                    'banque', 'agence', 'client', 'service', 'monsieur', 'madame',
                    'bonjour', 'merci', 'salut', 'bjr', 'slt', 'dh', 'dirham',
                    'très', 'bien', 'mal', 'bon', 'mauvais'
                }
                self.stop_words.update(banking_stopwords)
                
            else:  # Arabic
                self.stop_words = set()  # Définir manuellement pour l'arabe
                self.stemmer = None
            
            # Charger spaCy si disponible
            try:
                self.nlp = spacy.load("fr_core_news_sm")
            except OSError:
                logger.warning("Modèle spaCy français non trouvé. Utilisation de NLTK uniquement.")
                self.nlp = None
            
            logger.info("Outils NLP initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des outils NLP: {e}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV
        
        Args:
            file_path: Chemin vers le fichier CSV
            
        Returns:
            DataFrame pandas avec les données
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Données chargées: {len(df)} lignes, {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les doublons basés sur le texte de l'avis et la banque"""
        
        initial_count = len(df)
        
        # Marquer les doublons basés sur review_text et bank_name
        df_clean = df.drop_duplicates(
            subset=['review_text', 'bank_name'], 
            keep='first'
        ).reset_index(drop=True)
        
        # Supprimer les avis vides ou trop courts
        df_clean = df_clean[
            (df_clean['review_text'].notna()) & 
            (df_clean['review_text'].str.len() >= 10)
        ].reset_index(drop=True)
        
        removed_count = initial_count - len(df_clean)
        logger.info(f"Doublons et avis courts supprimés: {removed_count} lignes")
        
        return df_clean
    
    def clean_text_basic(self, text: str) -> str:
        """
        Nettoyage de base du texte
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Supprimer les mentions email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Supprimer les numéros de téléphone
        text = re.sub(r'(\+212|0)[5-7]\d{8}', '', text)
        
        # Supprimer les balises HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Supprimer les emojis (plus complexe)
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Nettoyer les caractères spéciaux mais garder la ponctuation importante
        text = re.sub(r'[^\w\s\.,!?;:\'"-]', ' ', text)
        
        # Remplacer les espaces multiples par un seul
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les espaces en début/fin
        text = text.strip()
        
        return text
    
    def remove_accents(self, text: str) -> str:
        """Supprime les accents du texte français"""
        if not text:
            return text
            
        # Dictionnaire de remplacement des accents
        accent_map = {
            'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenise le texte en mots
        
        Args:
            text: Texte à tokeniser
            
        Returns:
            Liste des tokens
        """
        if not text:
            return []
        
        # Utiliser spaCy si disponible, sinon NLTK
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_space]
        else:
            tokens = word_tokenize(text, language='french')
            tokens = [token.lower() for token in tokens]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les mots vides
        
        Args:
            tokens: Liste des tokens
            
        Returns:
            Liste des tokens sans mots vides
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        Supprime la ponctuation
        
        Args:
            tokens: Liste des tokens
            
        Returns:
            Liste des tokens sans ponctuation
        """
        return [token for token in tokens if token not in string.punctuation]
    
    def lemmatize_text(self, tokens: List[str]) -> List[str]:
        """
        Lemmatise les tokens
        
        Args:
            tokens: Liste des tokens
            
        Returns:
            Liste des tokens lemmatisés
        """
        if self.nlp:
            # Utiliser spaCy pour la lemmatisation
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_.lower() for token in doc if not token.is_space]
        else:
            # Utiliser le stemmer NLTK comme alternative
            return [self.stemmer.stem(token) for token in tokens]
    
    def filter_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """
        Filtre les tokens selon la longueur et le contenu
        
        Args:
            tokens: Liste des tokens
            min_length: Longueur minimale des tokens
            
        Returns:
            Liste des tokens filtrés
        """
        filtered = []
        for token in tokens:
            # Garder seulement les tokens avec des lettres et suffisamment longs
            if (len(token) >= min_length and 
                re.match(r'^[a-zA-Z]+$', token) and 
                not token.isdigit()):
                filtered.append(token)
        
        return filtered
    
    def process_single_review(self, text: str, remove_accents: bool = False) -> Dict:
        """
        Traite un avis complet
        
        Args:
            text: Texte de l'avis
            remove_accents: Supprimer les accents ou non
            
        Returns:
            Dictionnaire avec les versions nettoyées
        """
        if not text or pd.isna(text):
            return {
                'original_text': text,
                'cleaned_text': '',
                'tokens': [],
                'filtered_tokens': [],
                'processed_text': '',
                'word_count': 0,
                'char_count': 0
            }
        
        # Nettoyage de base
        cleaned = self.clean_text_basic(text)
        
        # Supprimer les accents si demandé
        if remove_accents:
            cleaned = self.remove_accents(cleaned)
        
        # Tokenisation
        tokens = self.tokenize_text(cleaned)
        
        # Pipeline de nettoyage des tokens
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.filter_tokens(tokens)
        
        # Lemmatisation
        lemmatized_tokens = self.lemmatize_text(tokens)
        
        # Texte final
        processed_text = ' '.join(lemmatized_tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'tokens': tokens,
            'filtered_tokens': lemmatized_tokens,
            'processed_text': processed_text,
            'word_count': len(lemmatized_tokens),
            'char_count': len(processed_text)
        }
    
    def clean_dataset(self, df: pd.DataFrame, remove_accents: bool = False) -> pd.DataFrame:
        """
        Nettoie l'ensemble du dataset
        
        Args:
            df: DataFrame avec les avis
            remove_accents: Supprimer les accents
            
        Returns:
            DataFrame nettoyé avec nouvelles colonnes
        """
        logger.info("Début du nettoyage du dataset...")
        
        # Copier le DataFrame
        df_clean = df.copy()
        
        # Supprimer les doublons
        df_clean = self.remove_duplicates(df_clean)
        
        # Traiter chaque avis
        results = []
        for idx, text in enumerate(df_clean['review_text']):
            if idx % 100 == 0:
                logger.info(f"Traitement: {idx}/{len(df_clean)} avis")
            
            result = self.process_single_review(text, remove_accents)
            results.append(result)
        
        # Ajouter les nouvelles colonnes
        df_clean['cleaned_text'] = [r['cleaned_text'] for r in results]
        df_clean['processed_text'] = [r['processed_text'] for r in results]
        df_clean['tokens'] = [r['filtered_tokens'] for r in results]
        df_clean['word_count'] = [r['word_count'] for r in results]
        df_clean['char_count'] = [r['char_count'] for r in results]
        
        # Filtrer les avis trop courts après nettoyage
        min_words = 3
        df_clean = df_clean[df_clean['word_count'] >= min_words].reset_index(drop=True)
        
        logger.info(f"Nettoyage terminé. {len(df_clean)} avis conservés")
        
        return df_clean
    
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des caractéristiques textuelles supplémentaires"""
        
        logger.info("Ajout de caractéristiques textuelles...")
        
        df_features = df.copy()
        
        # Longueurs
        df_features['original_length'] = df_features['review_text'].str.len()
        df_features['cleaned_length'] = df_features['cleaned_text'].str.len()
        
        # Comptages
        df_features['sentence_count'] = df_features['review_text'].apply(
            lambda x: len(sent_tokenize(x, language='french')) if pd.notna(x) else 0
        )
        
        # Présence de ponctuation exclamative/interrogative
        df_features['has_exclamation'] = df_features['review_text'].str.contains('!', na=False)
        df_features['has_question'] = df_features['review_text'].str.contains('\?', na=False)
        
        # Présence de majuscules (cris)
        df_features['has_caps'] = df_features['review_text'].apply(
            lambda x: any(word.isupper() and len(word) > 2 for word in x.split()) if pd.notna(x) else False
        )
        
        # Polarité basique avec TextBlob
        def get_polarity(text):
            if pd.isna(text) or not text:
                return 0
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity
            except:
                return 0
        
        df_features['textblob_polarity'] = df_features['cleaned_text'].apply(get_polarity)
        
        logger.info("Caractéristiques textuelles ajoutées")
        
        return df_features
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str = None):
        """Sauvegarde les données nettoyées"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/bank_reviews_cleaned_{timestamp}.csv"
        
        # Convertir les listes en strings pour la sauvegarde CSV
        df_save = df.copy()
        df_save['tokens'] = df_save['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        
        df_save.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Données nettoyées sauvegardées: {output_path}")
        
        return output_path
    
    def generate_cleaning_report(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame):
        """Génère un rapport de nettoyage"""
        
        report = {
            'original_count': len(df_original),
            'cleaned_count': len(df_cleaned),
            'removed_count': len(df_original) - len(df_cleaned),
            'removal_rate': (len(df_original) - len(df_cleaned)) / len(df_original) * 100,
            'avg_original_length': df_original['review_text'].str.len().mean(),
            'avg_cleaned_length': df_cleaned['cleaned_text'].str.len().mean(),
            'avg_word_count': df_cleaned['word_count'].mean(),
            'banks_count': df_cleaned['bank_name'].nunique(),
            'branches_count': df_cleaned['branch_name'].nunique()
        }
        
        logger.info("=== RAPPORT DE NETTOYAGE ===")
        logger.info(f"Avis originaux: {report['original_count']}")
        logger.info(f"Avis conservés: {report['cleaned_count']}")
        logger.info(f"Avis supprimés: {report['removed_count']} ({report['removal_rate']:.1f}%)")
        logger.info(f"Longueur moyenne originale: {report['avg_original_length']:.1f} caractères")
        logger.info(f"Longueur moyenne nettoyée: {report['avg_cleaned_length']:.1f} caractères")
        logger.info(f"Nombre moyen de mots: {report['avg_word_count']:.1f}")
        logger.info(f"Banques: {report['banks_count']}, Agences: {report['branches_count']}")
        
        return report

def main():
    """Fonction principale de nettoyage"""
    
    # Initialiser le nettoyeur
    cleaner = BankReviewsCleaner(language='french')
    
    try:
        # Charger les données
        input_file = "data/bank_reviews_scraping.csv"  # Votre fichier de scraping
        df_original = cleaner.load_data(input_file)
        
        logger.info(f"Données chargées: {len(df_original)} avis")
        
        # Nettoyer les données
        df_cleaned = cleaner.clean_dataset(df_original, remove_accents=False)
        
        # Ajouter des caractéristiques
        df_features = cleaner.add_text_features(df_cleaned)
        
        # Sauvegarder
        output_file = cleaner.save_cleaned_data(df_features)
        
        # Générer le rapport
        report = cleaner.generate_cleaning_report(df_original, df_features)
        
        print(f"\n✅ Nettoyage terminé!")
        print(f"📊 {report['cleaned_count']} avis nettoyés sur {report['original_count']}")
        print(f"💾 Fichier sauvegardé: {output_file}")
        
        # Aperçu des données nettoyées
        print(f"\n🔎 Aperçu des données nettoyées:")
        print(df_features[['bank_name', 'rating', 'word_count', 'textblob_polarity']].head())
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        raise

if __name__ == "__main__":
    main()