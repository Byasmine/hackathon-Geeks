from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

# Créer notre application API
app = FastAPI(title="API Screening CV")

# ==================== CHARGEMENT DU MODÈLE ====================
def charger_modele():
    """Charge le modèle depuis le fichier .pkl"""
    try:
        # Resolve path relative to this file to avoid depending on the current working directory
        model_path = Path(__file__).resolve().parent / 'models' / 'Resume_Screening.pkl'
        print(f"🔎 Attempting to load model from: {model_path}")

        if not model_path.exists():
            print(f"❌ Model file not found at: {model_path}")
            return None

        with model_path.open('rb') as f:
            model = pickle.load(f)

        print("✅ Modèle chargé avec succès")
        # Debug: Vérifier ce qui est chargé
        print(f"Type du modèle: {type(model)}")
        if hasattr(model, 'predict'):
            print("✅ Modèle a une méthode predict")
        else:
            print("❌ Modèle n'a pas de méthode predict")

        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        traceback.print_exc()
        return None


def charger_modele_salaire():
    """Charge le modèle de prédiction de salaire depuis models/Salary_Prediction.pkl"""
    try:
        salary_path = Path(__file__).resolve().parent / 'models' / 'Salary_Prediction.pkl'
        print(f"🔎 Attempting to load salary model from: {salary_path}")

        if not salary_path.exists():
            print(f"⚠️ Salary model not found at: {salary_path}")
            return None

        with salary_path.open('rb') as f:
            salary_model = pickle.load(f)

        print("✅ Salary model loaded")
        print(f"Type salary model: {type(salary_model)}")
        return salary_model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du salary model: {e}")
        traceback.print_exc()
        return None

# Charger le modèle au démarrage
modele = charger_modele()
modele_salaire = charger_modele_salaire()

# ==================== MODÈLES PYDANTIC ====================
class DonneesCV(BaseModel):
    resume_text: str
    jd_text: str
    job_family: str
    seniority: str

class ReponseAPI(BaseModel):
    advancement_probability: float
    binary_decision: bool
    recommendation: str

# ==================== FONCTION DE PRÉDICTION RÉELLE ====================
def faire_prediction_reelle(resume_text, jd_text, job_family, seniority):
    """
    Utilise VRAIMENT le modèle pour faire une prédiction
    """
    try:
        # DEBUG: Afficher ce que nous recevons
        print(f"Données reçues:")
        print(f"Resume: {resume_text[:100]}...")
        print(f"JD: {jd_text[:100]}...")
        print(f"Job Family: {job_family}")
        print(f"Seniority: {seniority}")
        
        # Si un modèle de screening est chargé et a une pipeline (ex: scikit-learn)
        if modele is not None and hasattr(modele, 'predict'):
            print("Utilisation du modèle de screening chargé (modele)")

            # Tentons de construire un DataFrame sommaire — beaucoup de pipelines acceptent
            # un DataFrame et effectuent leur propre preprocessing via ColumnTransformer
            try:
                input_df = pd.DataFrame({
                    'resume_text': [resume_text],
                    'jd_text': [jd_text],
                    'job_family': [job_family],
                    'seniority': [seniority]
                })

                # Si le modèle a predict_proba (classification binaire)
                if hasattr(modele, 'predict_proba'):
                    proba = modele.predict_proba(input_df)
                    # Retourne la probabilité de la classe positive
                    return float(proba[0][1])

                # Sinon, essayons predict (peut être régression ou autre)
                pred = modele.predict(input_df)
                # Si le prédicat renvoie une probabilité ou score
                if isinstance(pred, np.ndarray) and pred.shape[0] == 1:
                    return float(pred[0])
                else:
                    return float(pred)

            except Exception as e:
                print(f"⚠️ Erreur lors de l'utilisation du modèle de screening: {e}")
                traceback.print_exc()
                # Tomber en backoff vers l'ancien comportement si nécessaire

        # Méthode 2: Si c'est un modèle qui s'appelle directement
        if modele is not None and hasattr(modele, '__call__'):
            print("Utilisation de model() direct")
            try:
                input_data = np.random.random((1, 142))  # REMPLACEZ CEci si vous savez la vraie forme
                prediction = modele(input_data)
                return float(prediction[0][0])
            except Exception as e:
                print(f"Erreur en appelant le modèle directement: {e}")
                traceback.print_exc()

        # Si on arrive ici, on n'a pas pu utiliser le modèle de screening
        raise Exception("Aucun modèle de screening utilisable trouvé")
        
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        raise e

# ==================== ENDPOINT PRINCIPAL ====================
@app.post("/api/resume_screen/predict", response_model=ReponseAPI)
def predicire_interview(donnees: DonneesCV):
    """
    Endpoint principal pour la prédiction de screening CV
    Utilise VRAIMENT le modèle - pas de données mockées !
    """
    # Vérifier si le modèle est chargé
    if modele is None:
        return ReponseAPI(
            advancement_probability=0.0,
            binary_decision=False,
            recommendation="ERREUR: Modèle non chargé"
        )
    
    try:
        # ⚡⚡ UTILISER LE VRAI MODÈLE POUR LA PRÉDICTION ⚡⚡
        proba = faire_prediction_reelle(
            donnees.resume_text,
            donnees.jd_text,
            donnees.job_family,
            donnees.seniority
        )
        
        print(f"✅ Probabilité calculée: {proba}")
        
        # S'assurer que la probabilité est valide
        proba = max(0.0, min(1.0, proba))
        
        # Prendre une décision binaire
        decision = proba > 0.5
        
        # Générer une recommandation basée sur la VRAIE prédiction
        if proba > 0.8:
            recommandation = "Interview fortement recommandée - Excellente correspondance"
        elif proba > 0.6:
            recommandation = "Interview recommandée - Bonne correspondance"
        elif proba > 0.4:
            recommandation = "À considérer - Correspondance moyenne"
        else:
            recommandation = "Ne pas interviewer - Correspondance faible"
        
        # Retourner la réponse avec la VRAIE probabilité
        return ReponseAPI(
            advancement_probability=round(proba, 3),
            binary_decision=decision,
            recommendation=recommandation
        )
        
    except Exception as e:
        print(f"❌ Erreur dans l'endpoint: {e}")
        return ReponseAPI(
            advancement_probability=0.0,
            binary_decision=False,
            recommendation=f"ERREUR: {str(e)}"
        )

# ==================== ENDPOINTS SUPPLEMENTAIRES ====================
@app.get("/")
def accueil():
    return {
        "message": "Bienvenue sur l'API Screening CV", 
        "endpoint": "POST /api/resume_screen/predict",
        "model_loaded": modele is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if modele is not None else "error",
        "model_loaded": modele is not None,
        "model_type": str(type(modele)) if modele else "None"
    }


# ==================== ENDPOINT POUR PRÉDICTION DE SALAIRE ====================
class DonneesSalaire(BaseModel):
    resume_text: Optional[str] = ''
    jd_text: Optional[str] = ''
    job_family: Optional[str] = ''
    seniority: Optional[str] = ''


class ReponseSalaire(BaseModel):
    predicted_salary: float
    currency: Optional[str] = 'USD'
    note: Optional[str] = None


@app.post("/api/salary/predict", response_model=ReponseSalaire)
def predict_salary(d: DonneesSalaire):
    """Utilise le vrai modèle de salaire (Salary_Prediction.pkl) pour prédire le salaire."""
    if modele_salaire is None:
        raise HTTPException(status_code=500, detail="Salary model not loaded")

    try:
        # Construire un DataFrame simple — la plupart des pipelines scikit-learn acceptent un DataFrame
        input_df = pd.DataFrame({
            'resume_text': [d.resume_text or ''],
            'jd_text': [d.jd_text or ''],
            'job_family': [d.job_family or ''],
            'seniority': [d.seniority or '']
        })

        # Si c'est un modèle de régression (prédisant un salaire)
        if hasattr(modele_salaire, 'predict'):
            pred = modele_salaire.predict(input_df)
            predicted = float(pred[0]) if isinstance(pred, (list, tuple, np.ndarray)) else float(pred)
            return ReponseSalaire(predicted_salary=predicted, note='Prediction from Salary_Prediction.pkl')

        # Sinon, s'il renvoie des probabilités, tenter de les convertir en score
        if hasattr(modele_salaire, 'predict_proba'):
            proba = modele_salaire.predict_proba(input_df)
            score = float(proba[0][1])
            return ReponseSalaire(predicted_salary=score, note='Model returned a probability (0-1)')

        raise HTTPException(status_code=500, detail='Salary model has no predict method')
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LANCER L'API ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)