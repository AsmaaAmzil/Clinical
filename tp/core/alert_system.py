"""
Système d'alerte pour identifier les cas urgents nécessitant une attention immédiate.
"""
from typing import Dict, List, Any
import numpy as np

class AlertSystem:
    """
    Classe pour gérer les alertes médicales basées sur les données des patients.
    """
    
    def __init__(self, alert_thresholds: Dict[str, tuple] = None):
        """
        Initialise le système d'alerte avec des seuils personnalisables.
        
        Args:
            alert_thresholds: Dictionnaire des seuils d'alerte par paramètre (min, max)
        """
        # Seuils par défaut (à adapter selon les besoins cliniques)
        self.alert_thresholds = alert_thresholds or {
            'temperature': (35.0, 39.0),  # °C
            'heart_rate': (50, 120),      # bpm
            'blood_pressure_systolic': (90, 160),  # mmHg
            'blood_pressure_diastolic': (60, 100),  # mmHg
            'oxygen_saturation': (90, 100),  # %
            'respiratory_rate': (12, 20)   # respirations/min
        }
        self.critical_conditions = {
            'high': [
                'temperature > 40',
                'heart_rate > 140',
                'blood_pressure_systolic > 180',
                'blood_pressure_diastolic > 120',
                'oxygen_saturation < 85',
                'respiratory_rate > 30'
            ],
            'medium': [
                'temperature > 38.5',
                'heart_rate > 120',
                'blood_pressure_systolic > 160',
                'blood_pressure_diastolic > 100',
                'oxygen_saturation < 90',
                'respiratory_rate > 25'
            ]
        }
    
    def check_vital_signs(self, patient_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Vérifie les signes vitaux d'un patient et retourne les alertes si nécessaire.
        
        Args:
            patient_data: Dictionnaire des données du patient
            
        Returns:
            Liste des alertes générées
        """
        alerts = []
        
        # Vérification des valeurs critiques
        for condition in self.critical_conditions['high']:
            param, operator, value = self._parse_condition(condition)
            if param in patient_data and self._evaluate_condition(patient_data[param], operator, float(value)):
                alerts.append({
                    'parameter': param,
                    'value': patient_data[param],
                    'threshold': value,
                    'severity': 'high',
                    'message': f'CRITIQUE: {param} = {patient_data[param]} ({operator} {value})',
                    'action': 'Nécessite une attention médicale immédiate!'
                })
        
        # Vérification des valeurs d'alerte moyenne
        if not alerts:  # On ne vérifie les alertes moyennes que s'il n'y a pas d'alerte critique
            for condition in self.critical_conditions['medium']:
                param, operator, value = self._parse_condition(condition)
                if param in patient_data and self._evaluate_condition(patient_data[param], operator, float(value)):
                    alerts.append({
                        'parameter': param,
                        'value': patient_data[param],
                        'threshold': value,
                        'severity': 'medium',
                        'message': f'Alerte: {param} = {patient_data[param]} ({operator} {value})',
                        'action': 'Surveillance recommandée.'
                    })
        
        # Vérification des seuils normaux
        for param, (min_val, max_val) in self.alert_thresholds.items():
            if param in patient_data:
                value = patient_data[param]
                if value < min_val and not any(a['parameter'] == param for a in alerts):
                    alerts.append({
                        'parameter': param,
                        'value': value,
                        'threshold': f'< {min_val}',
                        'severity': 'low',
                        'message': f'Valeur basse: {param} = {value} (min: {min_val})',
                        'action': 'Surveillance conseillée.'
                    })
                elif value > max_val and not any(a['parameter'] == param for a in alerts):
                    alerts.append({
                        'parameter': param,
                        'value': value,
                        'threshold': f'> {max_val}',
                        'severity': 'low',
                        'message': f'Valeur élevée: {param} = {value} (max: {max_val})',
                        'action': 'Surveillance conseillée.'
                    })
        
        return alerts
    
    def _parse_condition(self, condition: str) -> tuple:
        """Parse une condition en paramètre, opérateur et valeur."""
        import re
        match = re.match(r'([a-zA-Z_]+)\s*([<>])\s*([0-9.]+)', condition)
        if not match:
            raise ValueError(f"Format de condition invalide: {condition}")
        return match.group(1), match.group(2), match.group(3)
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Évalue une condition simple."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        else:
            raise ValueError(f"Opérateur non supporté: {operator}")
    
    def format_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Formate les alertes pour l'affichage."""
        if not alerts:
            return "Aucune alerte pour le moment."
        
        output = []
        for alert in sorted(alerts, key=lambda x: x['severity'], reverse=True):
            severity_emoji = '🔴' if alert['severity'] == 'high' else '🟠' if alert['severity'] == 'medium' else '🟡'
            output.append(
                f"{severity_emoji} {alert['message']}\n"
                f"   → Action: {alert['action']}\n"
                f"   → Valeur: {alert['value']} (seuil: {alert['threshold']})\n"
            )
        
        return "\n".join(output)
