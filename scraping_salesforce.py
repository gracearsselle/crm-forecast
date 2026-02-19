import yfinance as yf
import os

def extraire_vers_csv():
    # 1. Configuration
    ticker_symbol = "CRM"
    nom_fichier = "self.csv"
    
    print(f"Extraction des données de Salesforce ({ticker_symbol})...")
    
    # 2. Récupération de TOUT l'historique (depuis 2004)
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="max")
    
    if df.empty:
        print("Erreur : Impossible de récupérer les données.")
        return

    # 3. Nettoyage : on remet la Date en colonne et on enlève le fuseau horaire
    df.index = df.index.date
    df.index.name = 'Date'
    
    # 4. Sauvegarde
    df.to_csv(nom_fichier)
    
    # 5. Localisation du fichier
    chemin_complet = os.path.abspath(nom_fichier)
    
    print("-" * 30)
    print(f"TERMINÉ AVEC SUCCÈS !")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nom du fichier : {nom_fichier}")
    print(f"Emplacement exact : {chemin_complet}")
    print("-" * 30)

if __name__ == "__main__":
    extraire_vers_csv()