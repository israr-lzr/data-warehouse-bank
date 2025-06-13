import os
import pandas as pd
import sqlite3
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_warehouse.log'),
        logging.StreamHandler()
    ]
)

class DataWarehouse:
    def __init__(self, db_path='data/bank_reviews_dw.db'):
        self.db_path = db_path

    def create_star_schema(self, conn):
        """Créer les tables selon un schéma en étoile."""
        cursor = conn.cursor()

        cursor.executescript("""
        DROP TABLE IF EXISTS fact_reviews;
        DROP TABLE IF EXISTS dim_bank;
        DROP TABLE IF EXISTS dim_branch;
        DROP TABLE IF EXISTS dim_date;
        DROP TABLE IF EXISTS dim_sentiment;

        CREATE TABLE dim_bank (
            bank_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE dim_branch (
            branch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_name TEXT,
            branch_location TEXT
        );

        CREATE TABLE dim_date (
            date_id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_date TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER
        );

        CREATE TABLE dim_sentiment (
            sentiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            textblob_polarity REAL
        );

        CREATE TABLE fact_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id INTEGER,
            branch_id INTEGER,
            date_id INTEGER,
            sentiment_id INTEGER,
            review_text TEXT,
            rating REAL,
            word_count INTEGER,
            char_count INTEGER,
            has_exclamation BOOLEAN,
            has_question BOOLEAN,
            has_caps BOOLEAN,
            FOREIGN KEY(bank_id) REFERENCES dim_bank(bank_id),
            FOREIGN KEY(branch_id) REFERENCES dim_branch(branch_id),
            FOREIGN KEY(date_id) REFERENCES dim_date(date_id),
            FOREIGN KEY(sentiment_id) REFERENCES dim_sentiment(sentiment_id)
        );
        """)
        conn.commit()
        logging.info("Schéma en étoile créé avec succès.")

    def insert_dimensions_and_facts(self, conn, df):
        """Insérer les données dans les tables dimensionnelles et la table de faits."""
        cursor = conn.cursor()

        # Dimensions: bank
        bank_map = {}
        for bank in df['bank_name'].unique():
            cursor.execute("INSERT OR IGNORE INTO dim_bank (bank_name) VALUES (?)", (bank,))
            conn.commit()
            cursor.execute("SELECT bank_id FROM dim_bank WHERE bank_name = ?", (bank,))
            bank_map[bank] = cursor.fetchone()[0]

        # Dimensions: branch
        branch_map = {}
        for _, row in df[['branch_name', 'branch_location']].drop_duplicates().iterrows():
            cursor.execute("INSERT INTO dim_branch (branch_name, branch_location) VALUES (?, ?)", (row['branch_name'], row['branch_location']))
            conn.commit()
            cursor.execute("SELECT branch_id FROM dim_branch WHERE branch_name = ? AND branch_location = ?", (row['branch_name'], row['branch_location']))
            branch_map[(row['branch_name'], row['branch_location'])] = cursor.fetchone()[0]

        # Dimensions: date
        date_map = {}
        for date_str in df['review_date'].unique():
            try:
                parsed_date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(parsed_date):
                    continue
                year = parsed_date.year
                month = parsed_date.month
                day = parsed_date.day
                cursor.execute("INSERT INTO dim_date (review_date, year, month, day) VALUES (?, ?, ?, ?)", (date_str, year, month, day))
                conn.commit()
                cursor.execute("SELECT date_id FROM dim_date WHERE review_date = ?", (date_str,))
                date_map[date_str] = cursor.fetchone()[0]
            except Exception:
                continue

        # Dimensions: sentiment
        sentiment_map = {}
        for pol in df['textblob_polarity'].dropna().unique():
            cursor.execute("INSERT INTO dim_sentiment (textblob_polarity) VALUES (?)", (pol,))
            conn.commit()
            cursor.execute("SELECT sentiment_id FROM dim_sentiment WHERE textblob_polarity = ?", (pol,))
            sentiment_map[pol] = cursor.fetchone()[0]

        # Faits
        for _, row in df.iterrows():
            bank_id = bank_map.get(row['bank_name'])
            branch_id = branch_map.get((row['branch_name'], row['branch_location']))
            date_id = date_map.get(row['review_date'])
            sentiment_id = sentiment_map.get(row['textblob_polarity'], None)

            cursor.execute("""
                INSERT INTO fact_reviews (
                    bank_id, branch_id, date_id, sentiment_id,
                    review_text, rating, word_count, char_count,
                    has_exclamation, has_question, has_caps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bank_id, branch_id, date_id, sentiment_id,
                row['review_text'], row['rating'], row['word_count'], row['char_count'],
                row['has_exclamation'], row['has_question'], row['has_caps']
            ))
        conn.commit()
        logging.info("Données insérées dans la table de faits.")

    def build_complete_warehouse(self, csv_file):
        """Pipeline complet de construction du data warehouse."""
        logging.info("=== CONSTRUCTION DU DATA WAREHOUSE ===")

        if not os.path.exists(csv_file):
            logging.error(f"Fichier introuvable : {csv_file}")
            return

        conn = sqlite3.connect(self.db_path)
        logging.info(f"Connexion SQLite établie: {self.db_path}")

        df = pd.read_csv(csv_file)
        logging.info(f"Données sources chargées: {len(df)} avis")

        self.create_star_schema(conn)
        logging.info("Création du schéma en étoile...")

        self.insert_dimensions_and_facts(conn, df)
        logging.info("✅ Data warehouse construit avec succès.")
        conn.close()

def main():
    dw = DataWarehouse()
    source_file = "data/bank_reviews_cleaned_20250611_234258.csv"
    dw.build_complete_warehouse(source_file)

if __name__ == "__main__":
    main()
