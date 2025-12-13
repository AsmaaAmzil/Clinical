from core.alert_system import AlertSystem

def demo_architecture():
    """Démontre l'utilisation de l'architecture complète"""
    
    print("=" * 70)
    print("DÉMONSTRATION : Architecture Modulaire virus_diag")
    print("=" * 70)
    
    # Créer des données synthétiques pour la démo
    from sklearn.datasets import make_classification
    import numpy as np
    import pandas as pd
    
    X, y = make_classification(
        n_samples=100, 
        n_features=5, 
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    
    # ===== DÉMONSTRATION DU SYSTÈME D'ALERTE =====
    print("\n🚨 DÉMONSTRATION DU SYSTÈME D'ALERTE MÉDICALE")
    print("-" * 70)
    
    # Initialiser le système d'alerte
    alert_system = AlertSystem()
    
    # Exemple de données de patients
    patients = [
        {
            'id': 'P001',
            'name': 'Jean Dupont',
            'vitals': {
                'temperature': 38.7,
                'heart_rate': 92,
                'blood_pressure_systolic': 145,
                'blood_pressure_diastolic': 95,
                'oxygen_saturation': 97,
                'respiratory_rate': 18
            }
        },
        {
            'id': 'P002',
            'name': 'Marie Martin',
            'vitals': {
                'temperature': 40.5,
                'heart_rate': 130,
                'blood_pressure_systolic': 160,
                'blood_pressure_diastolic': 85,
                'oxygen_saturation': 82,
                'respiratory_rate': 28
            }
        },
        {
            'id': 'P003',
            'name': 'Pierre Durand',
            'vitals': {
                'temperature': 37.2,
                'heart_rate': 75,
                'blood_pressure_systolic': 120,
                'blood_pressure_diastolic': 80,
                'oxygen_saturation': 98,
                'respiratory_rate': 16
            }
        }
    ]
    
    # Vérifier les alertes pour chaque patient
    for patient in patients:
        print(f"\n🔍 Analyse du patient: {patient['name']} ({patient['id']})")
        print("-" * 40)
        
        # Vérifier les signes vitaux
        alerts = alert_system.check_vital_signs(patient['vitals'])
        
        # Afficher les alertes
        print("📊 Signes vitaux:")
        for param, value in patient['vitals'].items():
            print(f"   • {param.replace('_', ' ').title()}: {value}")
        
        print("\n🚨 Alertes:")
        print(alert_system.format_alerts(alerts))
        
        # Afficher une recommandation basée sur la sévérité
        severities = [alert['severity'] for alert in alerts]
        if 'high' in severities:
            print("\n❌ RECOMMANDATION: Intervention médicale urgente requise!")
        elif 'medium' in severities:
            print("\n⚠️  RECOMMANDATION: Surveillance médicale recommandée.")
        elif alerts:
            print("\nℹ️  RECOMMANDATION: Surveillance standard.")
        else:
            print("\n✅ Aucune alerte - Paramètres dans les normes.")
    
    print("\n" + "=" * 70 + "\n")
    
    # Créer un CSV temporaire pour simuler patient_data.csv
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['diagnosis'] = y
    temp_csv = '/tmp/patient_data.csv'
    
    # ===== ÉTAPE 1 : CHARGEMENT DES DONNÉES (data/) =====
    print("\n1️⃣  CHARGEMENT DES DONNÉES (data/)")
    print("-" * 70)
    
    dataset = Dataset()
    dataset.load_from_csv(temp_csv)
    print(f"✓ Données chargées depuis patient_data.csv")
    print(f"  - Échantillons d'entraînement: {len(dataset.X_train)}")
    print(f"  - Échantillons de test: {len(dataset.X_test)}")
    
    # ===== ÉTAPE 2 : PRÉTRAITEMENT (utils/) =====
    print("\n2️⃣  PRÉTRAITEMENT (utils/)")
    print("-" * 70)
    
    preprocessor = Preprocessor()
    dataset.X_train = preprocessor.normalize(dataset.X_train)
    dataset.X_test = preprocessor.normalize(dataset.X_test)
    print("✓ Données normalisées")
    
    # ===== ÉTAPE 3 : ENTRAÎNEMENT (core/ + pipeline/) =====
    print("\n3️⃣  ENTRAÎNEMENT DU MODÈLE (core/ + pipeline/)")
    print("-" * 70)
    
    # Créer un modèle
    model = LogisticRegressionModel(max_iter=1000)
    print(f"✓ Modèle créé: LogisticRegression")
    
    # Entraîner avec le Trainer
    trainer = Trainer(model, dataset)
    trainer.train()
    trained_model = trainer.get_trained_model()
    print(f"✓ Modèle entraîné avec succès")
    
    # ===== ÉTAPE 4 : ÉVALUATION (pipeline/) =====
    print("\n4️⃣  ÉVALUATION (pipeline/)")
    print("-" * 70)
    
    evaluator = Evaluator(trained_model)
    X_test, y_test = dataset.get_test_data()
    metrics = evaluator.evaluate(X_test, y_test)
    evaluator.print_report()
    
    # ===== ÉTAPE 5 : DÉPLOIEMENT EN PRODUCTION (app/) =====
    print("\n5️⃣  DÉPLOIEMENT EN PRODUCTION (app/)")
    print("-" * 70)
    
    # Créer l'interface clinique (réponse à l'exercice)
    predictor = ClinicalPredictor(model=trained_model)
    print("✓ ClinicalPredictor initialisé avec le modèle entraîné")
    
    # Tester des prédictions
    print("\n📋 Tests de diagnostic:")
    for i in range(3):
        patient = X_test[i]
        diagnosis = predictor.diagnose(patient)
        actual = "Infecté" if y_test[i] == 1 else "Sain"
        match = "✓" if diagnosis == actual else "✗"
        print(f"  Patient {i+1}: {diagnosis:8s} (Réel: {actual:8s}) {match}")
    
    # ===== ÉTAPE 6 : API REST (app/) =====
    print("\n6️⃣  API REST (app/)")
    print("-" * 70)
    
    api = ClinicalAPI(predictor)
    patient_data = {f'feature_{i}': X_test[0][i] for i in range(5)}
    response = api.predict_endpoint(patient_data)
    print(f"✓ API Response: {response}")
    
    print("\n" + "=" * 70)


def explain_architecture():
    """Explique l'architecture modulaire"""
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE MODULAIRE virus_diag")
    print("=" * 70)
    
    print("""
📁 virus_diag/
│
├── 📂 data/                    # Données brutes
│   └── patient_data.csv        # Données des patients
│
├── 📂 core/                    # Cœur de l'application IA
│   ├── dataset.py              # Gestion des données
│   ├── model.py                # Interface de base des modèles
│   ├── logistic_regression.py  # Modèle régression logistique
│   ├── neural_network.py       # Modèle réseau de neurones
│   └── optimizer.py            # Optimisation hyperparamètres
│
├── 📂 pipeline/                # Pipeline d'entraînement
│   ├── trainer.py              # Entraînement des modèles
│   └── evaluator.py            # Évaluation des performances
│
├── 📂 utils/                   # Utilitaires
│   ├── preprocessing.py        # Prétraitement des données
│   └── metrics.py              # Calcul de métriques
│
└── 📂 app/                     # Application de production
    ├── interface_clinique.py   # ClinicalPredictor ⭐
    └── api.py                  # API REST

⭐ ClinicalPredictor est dans app/interface_clinique.py
""")
    
    print("\n💡 PRINCIPE D'ARCHITECTURE")
    print("-" * 70)
    print("""
1️⃣  SÉPARATION DES COUCHES
   • data/       : Stockage des données
   • core/       : Logique métier IA (modèles, optimisation)
   • pipeline/   : Orchestration (entraînement, évaluation)
   • utils/      : Fonctions réutilisables
   • app/        : Interface utilisateur (production)

2️⃣  FLUX DE TRAVAIL
   data/ → utils/ → core/ → pipeline/ → app/
   
   • Les données sont chargées et prétraitées
   • Les modèles sont créés et entraînés
   • Les performances sont évaluées
   • Le modèle est déployé en production

3️⃣  AVANTAGES
   ✓ Modularité : chaque module a une responsabilité
   ✓ Réutilisabilité : composants indépendants
   ✓ Maintenabilité : facile à déboguer et améliorer
   ✓ Testabilité : chaque module peut être testé séparément
   ✓ Scalabilité : ajout de nouvelles fonctionnalités facilité
""")


def answer_questions():
    """Répond aux questions de l'exercice"""
    
    print("\n" + "=" * 70)
    print("RÉPONSES AUX QUESTIONS DE L'EXERCICE")
    print("=" * 70)
    
    print("\n❓ Question 2 : Concept POO permettant à ClinicalPredictor")
    print("   de fonctionner avec n'importe quel modèle IA")
    print("-" * 70)
    print("""
📌 POLYMORPHISME et DUCK TYPING

ClinicalPredictor accepte n'importe quel objet qui hérite de Model.
Tous les modèles (LogisticRegression, NeuralNetwork, etc.) partagent
la même interface : predict() et predict_proba().

Exemple:
    model1 = LogisticRegressionModel()
    model2 = NeuralNetworkModel()
    
    predictor1 = ClinicalPredictor(model1)  # ✓ Fonctionne
    predictor2 = ClinicalPredictor(model2)  # ✓ Fonctionne aussi

Le prédicteur ne connaît pas les détails d'implémentation, seulement
l'interface commune → POLYMORPHISME.
""")
    
    print("\n❓ Question 3 : Pourquoi séparer ClinicalPredictor (app/)")
    print("   de Trainer (pipeline/) dans l'architecture IA ?")
    print("-" * 70)
    print("""
📌 SÉPARATION DES RESPONSABILITÉS (Separation of Concerns)

┌─────────────────────┬──────────────────────────┐
│   TRAINER           │   CLINICAL PREDICTOR     │
│   (pipeline/)       │   (app/)                 │
├─────────────────────┼──────────────────────────┤
│ • Entraîne          │ • Prédit                 │
│ • Optimise          │ • Diagnostique           │
│ • Évalue            │ • Sert les utilisateurs  │
│ • Expérimente       │ • Performance temps réel │
├─────────────────────┼──────────────────────────┤
│ ENVIRONNEMENT       │ ENVIRONNEMENT            │
│ • Dev/Research      │ • Production             │
│ • GPU/TPU           │ • CPU léger              │
│ • Accès aux données │ • Pas de données train   │
│ • Longues sessions  │ • Latence < 100ms        │
└─────────────────────┴──────────────────────────┘

AVANTAGES ARCHITECTURAUX :

1️⃣  DÉPLOIEMENT INDÉPENDANT
   • Trainer : cluster de calcul (GPU) pour l'entraînement
   • Predictor : serveurs web légers pour les prédictions

2️⃣  SÉCURITÉ
   • Les données d'entraînement restent isolées
   • Le code de production n'accède pas aux données sensibles

3️⃣  PERFORMANCE
   • Predictor optimisé pour la latence
   • Trainer optimisé pour le throughput

4️⃣  ÉVOLUTIVITÉ
   • Mettre à jour le modèle sans redéployer l'app
   • Scaler horizontalement les prédicteurs

5️⃣  MAINTENANCE
   • Tester l'entraînement sans affecter la prod
   • Déboguer les prédictions indépendamment
""")


if __name__ == "__main__":
    demo_architecture()
    # Désactivez les appels suivants si nécessaire pour la démonstration
    # explain_architecture()
    # answer_questions()
    
    print("\n" + "=" * 70)
    print("✅ EXERCICE COMPLET : Architecture respectée !")
    print("=" * 70)