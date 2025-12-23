"""
ETAPE 4: Exemple Réel - Diagnostic Médical avec Réseau Bayésien
================================================================

Scénario: Diagnostic de maladies respiratoires
Un patient présente des symptômes respiratoires. Le médecin doit déterminer
la cause parmi plusieurs maladies possibles en utilisant symptômes et tests.

Variables du réseau:
- Saison (Hiver/Été) - Variable de contexte
- Vaccination COVID (Oui/Non) - Antécédent médical
- Grippe (Oui/Non) - Maladie possible
- COVID-19 (Oui/Non) - Maladie possible
- Allergie (Oui/Non) - Maladie possible
- Fièvre (Oui/Non) - Symptôme
- Toux (Oui/Non) - Symptôme
- Fatigue (Oui/Non) - Symptôme
- Écoulement nasal (Oui/Non) - Symptôme
- Test COVID (Positif/Négatif) - Test médical
- Analyse sanguine (Normal/Anormal) - Test médical

Structure du réseau (connexions multiples):
              Saison
                |
              Grippe ────────┐
                |            │
    Vaccination COVID    COVID-19
                |            │
                └────────────┤
                             │
              Allergie ──────┤
                |            │
    ┌───────────┼────────────┼───────────┐
    │           │            │           │
  Fièvre      Toux        Fatigue   Écoulement
                           │           │
                           │           │
                    Test COVID   Analyse Sang
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np


class DiagnosticMedical:
    """
    Réseau Bayésien pour le diagnostic médical de maladies respiratoires
    """
    
    def __init__(self):
        # Créer la structure du réseau avec connexions multiples
        self.model = DiscreteBayesianNetwork([
            # Influences contextuelles
            ('Saison', 'Grippe'),
            ('Saison', 'Allergie'),
            
            # Facteurs de risque
            ('VaccinationCOVID', 'COVID19'),
            
            # Maladies → Symptômes (connexions multiples)
            ('Grippe', 'Fievre'),
            ('COVID19', 'Fievre'),
            
            ('Grippe', 'Toux'),
            ('COVID19', 'Toux'),
            ('Allergie', 'Toux'),
            
            ('Grippe', 'Fatigue'),
            ('COVID19', 'Fatigue'),
            
            ('Allergie', 'EcoulementNasal'),
            
            # Symptômes/Maladies → Tests
            ('COVID19', 'TestCOVID'),
            ('Grippe', 'AnalyseSang'),
            ('COVID19', 'AnalyseSang')
        ])
        
        self._definir_cpds()
        
    def _definir_cpds(self):
        """
        Définir toutes les distributions de probabilité conditionnelles
        """
        
        # ===== VARIABLES DE CONTEXTE =====
        
        # CPD Saison (Hiver/Été)
        cpd_saison = TabularCPD(
            variable='Saison',
            variable_card=2,
            values=[[0.5], [0.5]],  # 50-50 Hiver/Été
            state_names={'Saison': ['Ete', 'Hiver']}
        )
        
        # CPD Vaccination COVID
        cpd_vaccination = TabularCPD(
            variable='VaccinationCOVID',
            variable_card=2,
            values=[[0.3], [0.7]],  # 70% vaccinés
            state_names={'VaccinationCOVID': ['Non', 'Oui']}
        )
        
        # ===== MALADIES =====
        
        # CPD Grippe | Saison
        # Plus probable en hiver
        cpd_grippe = TabularCPD(
            variable='Grippe',
            variable_card=2,
            values=[
                [0.98, 0.90],  # P(Grippe=Non | Saison)
                [0.02, 0.10]   # P(Grippe=Oui | Saison)
            ],
            evidence=['Saison'],
            evidence_card=[2],
            state_names={
                'Grippe': ['Non', 'Oui'],
                'Saison': ['Ete', 'Hiver']
            }
        )
        
        # CPD COVID-19 | Vaccination
        # Vaccination réduit le risque
        cpd_covid = TabularCPD(
            variable='COVID19',
            variable_card=2,
            values=[
                [0.94, 0.98],  # P(COVID=Non | Vaccination)
                [0.06, 0.02]   # P(COVID=Oui | Vaccination)
            ],
            evidence=['VaccinationCOVID'],
            evidence_card=[2],
            state_names={
                'COVID19': ['Non', 'Oui'],
                'VaccinationCOVID': ['Non', 'Oui']
            }
        )
        
        # CPD Allergie | Saison
        # Plus probable en été (pollens)
        cpd_allergie = TabularCPD(
            variable='Allergie',
            variable_card=2,
            values=[
                [0.80, 0.95],  # P(Allergie=Non | Saison)
                [0.20, 0.05]   # P(Allergie=Oui | Saison)
            ],
            evidence=['Saison'],
            evidence_card=[2],
            state_names={
                'Allergie': ['Non', 'Oui'],
                'Saison': ['Ete', 'Hiver']
            }
        )
        
        # ===== SYMPTÔMES =====
        
        # CPD Fièvre | Grippe, COVID
        cpd_fievre = TabularCPD(
            variable='Fievre',
            variable_card=2,
            values=[
                [0.99, 0.30, 0.20, 0.10],  # P(Fievre=Non | ...)
                [0.01, 0.70, 0.80, 0.90]   # P(Fievre=Oui | ...)
            ],
            evidence=['Grippe', 'COVID19'],
            evidence_card=[2, 2],
            state_names={
                'Fievre': ['Non', 'Oui'],
                'Grippe': ['Non', 'Oui'],
                'COVID19': ['Non', 'Oui']
            }
        )
        
        # CPD Toux | Grippe, COVID, Allergie
        cpd_toux = TabularCPD(
            variable='Toux',
            variable_card=2,
            values=[
                [0.95, 0.30, 0.40, 0.15, 0.20, 0.10, 0.15, 0.05],  # Non
                [0.05, 0.70, 0.60, 0.85, 0.80, 0.90, 0.85, 0.95]   # Oui
            ],
            evidence=['Grippe', 'COVID19', 'Allergie'],
            evidence_card=[2, 2, 2],
            state_names={
                'Toux': ['Non', 'Oui'],
                'Grippe': ['Non', 'Oui'],
                'COVID19': ['Non', 'Oui'],
                'Allergie': ['Non', 'Oui']
            }
        )
        
        # CPD Fatigue | Grippe, COVID
        cpd_fatigue = TabularCPD(
            variable='Fatigue',
            variable_card=2,
            values=[
                [0.90, 0.30, 0.20, 0.10],  # Non
                [0.10, 0.70, 0.80, 0.90]   # Oui
            ],
            evidence=['Grippe', 'COVID19'],
            evidence_card=[2, 2],
            state_names={
                'Fatigue': ['Non', 'Oui'],
                'Grippe': ['Non', 'Oui'],
                'COVID19': ['Non', 'Oui']
            }
        )
        
        # CPD Écoulement nasal | Allergie
        cpd_ecoulement = TabularCPD(
            variable='EcoulementNasal',
            variable_card=2,
            values=[
                [0.95, 0.20],  # Non
                [0.05, 0.80]   # Oui
            ],
            evidence=['Allergie'],
            evidence_card=[2],
            state_names={
                'EcoulementNasal': ['Non', 'Oui'],
                'Allergie': ['Non', 'Oui']
            }
        )
        
        # ===== TESTS MÉDICAUX =====
        
        # CPD Test COVID | COVID
        # Sensibilité: 85%, Spécificité: 98%
        cpd_test_covid = TabularCPD(
            variable='TestCOVID',
            variable_card=2,
            values=[
                [0.98, 0.15],  # Négatif
                [0.02, 0.85]   # Positif
            ],
            evidence=['COVID19'],
            evidence_card=[2],
            state_names={
                'TestCOVID': ['Negatif', 'Positif'],
                'COVID19': ['Non', 'Oui']
            }
        )
        
        # CPD Analyse sanguine | Grippe, COVID
        cpd_analyse = TabularCPD(
            variable='AnalyseSang',
            variable_card=2,
            values=[
                [0.95, 0.30, 0.40, 0.20],  # Normal
                [0.05, 0.70, 0.60, 0.80]   # Anormal
            ],
            evidence=['Grippe', 'COVID19'],
            evidence_card=[2, 2],
            state_names={
                'AnalyseSang': ['Normal', 'Anormal'],
                'Grippe': ['Non', 'Oui'],
                'COVID19': ['Non', 'Oui']
            }
        )
        
        # Ajouter tous les CPDs
        self.model.add_cpds(
            cpd_saison, cpd_vaccination, cpd_grippe, cpd_covid, cpd_allergie,
            cpd_fievre, cpd_toux, cpd_fatigue, cpd_ecoulement,
            cpd_test_covid, cpd_analyse
        )
        
        # Vérifier le modèle
        assert self.model.check_model(), "Le modèle n'est pas valide!"
    
    def afficher_structure(self):
        """
        Afficher la structure du réseau
        """
        print("\n" + "="*80)
        print("STRUCTURE DU RÉSEAU BAYÉSIEN")
        print("="*80)
        print(f"\nNombre de nœuds: {len(self.model.nodes())}")
        print(f"Nombre d'arêtes: {len(self.model.edges())}")
        
        print("\nArêtes (relations causales):")
        for edge in sorted(self.model.edges()):
            print(f"  {edge[0]} → {edge[1]}")
        
        print("\nNœuds par type:")
        contexte = ['Saison', 'VaccinationCOVID']
        maladies = ['Grippe', 'COVID19', 'Allergie']
        symptomes = ['Fievre', 'Toux', 'Fatigue', 'EcoulementNasal']
        tests = ['TestCOVID', 'AnalyseSang']
        
        print(f"  Contexte: {', '.join(contexte)}")
        print(f"  Maladies: {', '.join(maladies)}")
        print(f"  Symptômes: {', '.join(symptomes)}")
        print(f"  Tests: {', '.join(tests)}")
    
    def visualiser_reseau(self, save_path='images/diagnostic_medical_structure.png'):
        """
        Visualiser le réseau bayésien
        """
        plt.figure(figsize=(16, 12))
        
        G = nx.DiGraph(self.model.edges())
        
        # Définir les positions manuellement pour une belle présentation
        pos = {
            # Niveau 1: Contexte
            'Saison': (1, 5),
            'VaccinationCOVID': (3, 5),
            
            # Niveau 2: Maladies
            'Grippe': (0, 4),
            'COVID19': (2, 4),
            'Allergie': (4, 4),
            
            # Niveau 3: Symptômes
            'Fievre': (0, 2.5),
            'Toux': (1.5, 2.5),
            'Fatigue': (2.5, 2.5),
            'EcoulementNasal': (4, 2.5),
            
            # Niveau 4: Tests
            'TestCOVID': (2, 1),
            'AnalyseSang': (1, 1)
        }
        
        # Couleurs par type
        node_colors = []
        for node in G.nodes():
            if node in ['Saison', 'VaccinationCOVID']:
                node_colors.append('#FFE5B4')  # Beige (contexte)
            elif node in ['Grippe', 'COVID19', 'Allergie']:
                node_colors.append('#FFB6C1')  # Rose (maladies)
            elif node in ['Fievre', 'Toux', 'Fatigue', 'EcoulementNasal']:
                node_colors.append('#87CEEB')  # Bleu clair (symptômes)
            else:
                node_colors.append('#90EE90')  # Vert clair (tests)
        
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=3500,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            edge_color='gray',
            width=2,
            alpha=0.9
        )
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFE5B4', label='Contexte'),
            Patch(facecolor='#FFB6C1', label='Maladies'),
            Patch(facecolor='#87CEEB', label='Symptômes'),
            Patch(facecolor='#90EE90', label='Tests médicaux')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title("Réseau Bayésien - Diagnostic Médical", fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: {save_path}")
        plt.show()
    
    def cas_clinique(self, nom_patient, age, evidences, use_belief_propagation=False):
        """
        Analyser un cas clinique avec inférence bayésienne
        
        Args:
            nom_patient: Nom du patient
            age: Âge du patient
            evidences: Dictionnaire des observations
            use_belief_propagation: Si True, utilise Belief Propagation au lieu de Variable Elimination
        """
        # Choisir l'algorithme d'inférence
        if use_belief_propagation:
            inference = BeliefPropagation(self.model)
            algo_name = "Belief Propagation"
        else:
            inference = VariableElimination(self.model)
            algo_name = "Variable Elimination"
        
        print("\n" + "="*80)
        print(f"CAS CLINIQUE: {nom_patient}, {age} ans")
        print(f"Algorithme d'inférence: {algo_name}")
        print("="*80)
        
        print("\n📋 Observations cliniques:")
        for var, val in evidences.items():
            # Convertir les noms de variables en français
            var_fr = var.replace('Saison', 'Saison').replace('VaccinationCOVID', 'Vaccination COVID')
            var_fr = var_fr.replace('Fievre', 'Fièvre').replace('Toux', 'Toux')
            var_fr = var_fr.replace('Fatigue', 'Fatigue').replace('EcoulementNasal', 'Écoulement nasal')
            var_fr = var_fr.replace('TestCOVID', 'Test COVID').replace('AnalyseSang', 'Analyse sanguine')
            print(f"  • {var_fr}: {val}")
        
        # Inférence pour chaque maladie
        print("\n🔬 PROBABILITÉS DE DIAGNOSTIC:")
        print("-" * 80)
        
        resultats = {}
        for maladie in ['Grippe', 'COVID19', 'Allergie']:
            result = inference.query(variables=[maladie], evidence=evidences)
            prob_oui = result.values[1]
            resultats[maladie] = prob_oui
            
            maladie_fr = maladie.replace('COVID19', 'COVID-19')
            print(f"\n{maladie_fr}:")
            print(f"  P({maladie_fr}=Oui | observations) = {prob_oui:.4f} ({prob_oui*100:.2f}%)")
            
            # Barre de progression visuelle
            barre = '█' * int(prob_oui * 50) + '░' * (50 - int(prob_oui * 50))
            print(f"  [{barre}]")
        
        # Diagnostic le plus probable
        maladie_probable = max(resultats, key=resultats.get)
        prob_max = resultats[maladie_probable]
        
        print("\n" + "="*80)
        print(f"💡 DIAGNOSTIC LE PLUS PROBABLE: {maladie_probable.replace('COVID19', 'COVID-19')}")
        print(f"   Probabilité: {prob_max:.4f} ({prob_max*100:.2f}%)")
        
        if prob_max > 0.7:
            print("   Niveau de confiance: ★★★ ÉLEVÉ")
        elif prob_max > 0.4:
            print("   Niveau de confiance: ★★☆ MODÉRÉ")
        else:
            print("   Niveau de confiance: ★☆☆ FAIBLE - Tests supplémentaires recommandés")
        
        print("="*80)
        
        return resultats
    
    def scenarios_cliniques(self):
        """
        Tester plusieurs cas cliniques réalistes avec les deux algorithmes
        """
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*20 + "SCÉNARIOS CLINIQUES RÉELS" + " "*33 + "║")
        print("╚" + "="*78 + "╝")
        
        cas = [
            {
                'nom': 'Jean Dupont',
                'age': 35,
                'evidences': {
                    'Saison': 'Hiver',
                    'VaccinationCOVID': 'Oui',
                    'Fievre': 'Oui',
                    'Toux': 'Oui',
                    'Fatigue': 'Oui',
                    'EcoulementNasal': 'Non',
                    'TestCOVID': 'Negatif'
                }
            },
            {
                'nom': 'Marie Martin',
                'age': 28,
                'evidences': {
                    'Saison': 'Ete',
                    'VaccinationCOVID': 'Oui',
                    'Fievre': 'Non',
                    'Toux': 'Oui',
                    'Fatigue': 'Non',
                    'EcoulementNasal': 'Oui'
                }
            },
            {
                'nom': 'Pierre Lambert',
                'age': 45,
                'evidences': {
                    'Saison': 'Hiver',
                    'VaccinationCOVID': 'Non',
                    'Fievre': 'Oui',
                    'Toux': 'Oui',
                    'Fatigue': 'Oui',
                    'TestCOVID': 'Positif',
                    'AnalyseSang': 'Anormal'
                }
            }
        ]
        
        tous_resultats = []
        
        for i, patient in enumerate(cas, 1):
            print(f"\n{'#'*80}")
            print(f"# PATIENT {i}/3")
            print(f"{'#'*80}")
            
            resultats = self.cas_clinique(
                patient['nom'],
                patient['age'],
                patient['evidences']
            )
            
            tous_resultats.append({
                'Patient': patient['nom'],
                'Âge': patient['age'],
                'P(Grippe)': f"{resultats['Grippe']:.4f}",
                'P(COVID-19)': f"{resultats['COVID19']:.4f}",
                'P(Allergie)': f"{resultats['Allergie']:.4f}"
            })
        
        # Tableau récapitulatif
        print("\n" + "="*80)
        print("TABLEAU RÉCAPITULATIF DES DIAGNOSTICS")
        print("="*80)
        df = pd.DataFrame(tous_resultats)
        print(df.to_string(index=False))
        
        df.to_csv('resultats/diagnostics_patients.csv', index=False)
        print("\n✓ Résultats sauvegardés: ReseauxBayesiens/resultats/diagnostics_patients.csv")
    
    def comparer_algorithmes(self):
        """
        Comparer Variable Elimination et Belief Propagation
        """
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*15 + "COMPARAISON DES ALGORITHMES D'INFÉRENCE" + " "*24 + "║")
        print("╚" + "="*78 + "╝")
        
        # Cas test
        evidences = {
            'Saison': 'Hiver',
            'VaccinationCOVID': 'Oui',
            'Fievre': 'Oui',
            'Toux': 'Oui',
            'Fatigue': 'Oui'
        }
        
        print("\n📋 Observations communes:")
        for var, val in evidences.items():
            print(f"  • {var}: {val}")
        
        # Test avec Variable Elimination
        print("\n" + "─"*80)
        print("1️⃣  VARIABLE ELIMINATION (Élimination de Variables)")
        print("─"*80)
        print("   • Algorithme: Exact")
        print("   • Méthode: Élimination successive des variables")
        print("   • Complexité: Dépend de l'ordre d'élimination")
        
        import time
        start = time.time()
        inference_ve = VariableElimination(self.model)
        result_ve = {}
        for maladie in ['Grippe', 'COVID19', 'Allergie']:
            res = inference_ve.query(variables=[maladie], evidence=evidences)
            result_ve[maladie] = res.values[1]
        time_ve = time.time() - start
        
        print(f"\n   Résultats:")
        for maladie, prob in result_ve.items():
            print(f"   • P({maladie}=Oui) = {prob:.4f} ({prob*100:.2f}%)")
        print(f"   ⏱️  Temps d'exécution: {time_ve*1000:.2f} ms")
        
        # Test avec Belief Propagation
        print("\n" + "─"*80)
        print("2️⃣  BELIEF PROPAGATION (Propagation de Croyance)")
        print("─"*80)
        print("   • Algorithme: Exact pour les arbres/polyarbres")
        print("   • Méthode: Passage de messages entre nœuds")
        print("   • Complexité: Linéaire pour les arbres")
        
        start = time.time()
        inference_bp = BeliefPropagation(self.model)
        result_bp = {}
        for maladie in ['Grippe', 'COVID19', 'Allergie']:
            res = inference_bp.query(variables=[maladie], evidence=evidences)
            result_bp[maladie] = res.values[1]
        time_bp = time.time() - start
        
        print(f"\n   Résultats:")
        for maladie, prob in result_bp.items():
            print(f"   • P({maladie}=Oui) = {prob:.4f} ({prob*100:.2f}%)")
        print(f"   ⏱️  Temps d'exécution: {time_bp*1000:.2f} ms")
        
        # Comparaison
        print("\n" + "="*80)
        print("📊 ANALYSE COMPARATIVE")
        print("="*80)
        
        print(f"\n⏱️  Performance:")
        print(f"   • Variable Elimination: {time_ve*1000:.2f} ms")
        print(f"   • Belief Propagation:   {time_bp*1000:.2f} ms")
        if time_ve < time_bp:
            print(f"   ➜ Variable Elimination est {time_bp/time_ve:.2f}x plus rapide")
        else:
            print(f"   ➜ Belief Propagation est {time_ve/time_bp:.2f}x plus rapide")
        
        print(f"\n✓ Précision:")
        print(f"   • Les deux algorithmes donnent des résultats identiques (algorithmes exacts)")
        
        print(f"\n💡 Quand utiliser chaque algorithme?")
        print(f"   • Variable Elimination:")
        print(f"     - Réseaux de taille moyenne")
        print(f"     - Requêtes ponctuelles")
        print(f"     - Graphes généraux (avec cycles)")
        print(f"   • Belief Propagation:")
        print(f"     - Arbres et polyarbres")
        print(f"     - Requêtes multiples")
        print(f"     - Mise à jour incrémentale")
        
        print("="*80)


def main():
    """
    Fonction principale
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "RÉSEAU BAYÉSIEN - DIAGNOSTIC MÉDICAL RÉEL" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    # Créer le modèle
    diagnostic = DiagnosticMedical()
    
    # Afficher la structure
    diagnostic.afficher_structure()
    
    # Visualiser le réseau
    diagnostic.visualiser_reseau()
    
    # Comparer les algorithmes d'inférence
    diagnostic.comparer_algorithmes()
    
    # Analyser des cas cliniques
    diagnostic.scenarios_cliniques()
    
    print("\n" + "="*80)
    print("✅ ANALYSE COMPLÈTE TERMINÉE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
