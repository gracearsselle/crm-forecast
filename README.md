# CRM Deep Learning Forecast — Streamlit App

## 🚀 Déploiement sur Streamlit Cloud

### Étape 1 — Préparer le repository GitHub
```
crm-forecast/
├── app.py
├── requirements.txt
└── README.md
```

### Étape 2 — Pousser sur GitHub
```bash
git init
git add app.py requirements.txt README.md
git commit -m "CRM Deep Learning Forecast App"
git remote add origin https://github.com/TON_USERNAME/crm-forecast.git
git push -u origin main
```

### Étape 3 — Déployer sur Streamlit Cloud
1. Aller sur https://share.streamlit.io
2. Connecter ton compte GitHub
3. Sélectionner le repo `crm-forecast`
4. Fichier principal : `app.py`
5. Cliquer **Deploy**

---

## 🤖 Sauvegarder les modèles (dans ton notebook)

```python
# NeuralProphet
from neuralprophet import save
save(model_np, "model_neuralprophet_crm.pkl")

# GRU PyTorch
torch.save(model_gru.state_dict(), "model_gru_salesforce.pth")
```

## 📊 Fonctionnalités
- **Marché** : Candlestick, volume, rendements annuels
- **Prédictions** : GRU + NeuralProphet + Monte Carlo
- **Techniques** : RSI, volatilité, corrélations
- **Comparaison** : Radar chart, métriques côte à côte
- **Modèles** : Architecture, pipeline, feature importance
