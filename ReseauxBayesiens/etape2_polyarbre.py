"""
ETAPE 2: Réseau Bayésien - Polyarbre (Polytree)
================================================

Un polyarbre est un DAG où il existe au plus un chemin non-dirigé entre deux nœuds.

Exemple: Système d'alarme domestique
Nœuds:
    - Cambriolage (B): Variable racine
    - Tremblement de terre (E): Variable racine
    - Alarme (A): Dépend de B et E
    - JohnAppelle (J): Dépend de A (variable d'évidence)
    - MaryAppelle (M): Dépend de A (variable d'évidence)

Structure: Polyarbre en forme de V inversé
    B     E
     \   /
       A
      / \
     J   M

Variable d'intérêt: Cambriolage (B)
Évidences: John et/ou Mary ont appelé
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class PolyarbreAlarme:
    """
    Modèle de polyarbre pour le système d'alarme classique
    """
    
    def __init__(self):
        # Créer la structure du réseau (DAG)
        self.model = DiscreteBayesianNetwork([
            ('Cambriolage', 'Alarme'),
            ('TremblementTerre', 'Alarme'),
            ('Alarme', 'JohnAppelle'),
            ('Alarme', 'MaryAppelle')
        ])
        
        self._definir_cpds()
        
    def _definir_cpds(self):
        """
        Définir les distributions de probabilité conditionnelles (CPD)
        """
        
        # CPD pour Cambriolage (nœud racine)
        # P(Cambriolage) = 0.001 (0.1%)
        cpd_cambriolage = TabularCPD(
            variable='Cambriolage',
            variable_card=2,
            values=[[0.999], [0.001]],
            state_names={'Cambriolage': ['Non', 'Oui']}
        )
        
        # CPD pour Tremblement de terre (nœud racine)
        # P(TremblementTerre) = 0.002 (0.2%)
        cpd_tremblement = TabularCPD(
            variable='TremblementTerre',
            variable_card=2,
            values=[[0.998], [0.002]],
            state_names={'TremblementTerre': ['Non', 'Oui']}
        )
        
        # CPD pour Alarme (dépend de Cambriolage et TremblementTerre)
        # P(Alarme | Cambriolage, TremblementTerre)
        cpd_alarme = TabularCPD(
            variable='Alarme',
            variable_card=2,
            values=[
                [0.999, 0.29, 0.71, 0.05],  # P(Alarme=Non | ...)
                [0.001, 0.71, 0.29, 0.95]   # P(Alarme=Oui | ...)
            ],
            evidence=['Cambriolage', 'TremblementTerre'],
            evidence_card=[2, 2],
            state_names={
                'Alarme': ['Non', 'Oui'],
                'Cambriolage': ['Non', 'Oui'],
                'TremblementTerre': ['Non', 'Oui']
            }
        )
        
        # CPD pour John Appelle (dépend de Alarme)
        # P(JohnAppelle | Alarme)
        cpd_john = TabularCPD(
            variable='JohnAppelle',
            variable_card=2,
            values=[
                [0.95, 0.10],  # P(JohnAppelle=Non | Alarme)
                [0.05, 0.90]   # P(JohnAppelle=Oui | Alarme)
            ],
            evidence=['Alarme'],
            evidence_card=[2],
            state_names={
                'JohnAppelle': ['Non', 'Oui'],
                'Alarme': ['Non', 'Oui']
            }
        )
        
        # CPD pour Mary Appelle (dépend de Alarme)
        # P(MaryAppelle | Alarme)
        cpd_mary = TabularCPD(
            variable='MaryAppelle',
            variable_card=2,
            values=[
                [0.99, 0.30],  # P(MaryAppelle=Non | Alarme)
                [0.01, 0.70]   # P(MaryAppelle=Oui | Alarme)
            ],
            evidence=['Alarme'],
            evidence_card=[2],
            state_names={
                'MaryAppelle': ['Non', 'Oui'],
                'Alarme': ['Non', 'Oui']
            }
        )
        
        # Ajouter les CPDs au modèle
        self.model.add_cpds(
            cpd_cambriolage,
            cpd_tremblement,
            cpd_alarme,
            cpd_john,
            cpd_mary
        )
        
        # Vérifier que le modèle est valide
        assert self.model.check_model(), "Le modèle n'est pas valide!"
        
    def afficher_structure(self):
        """
        Afficher la structure du réseau bayésien
        """
        print("\n" + "="*70)
        print("STRUCTURE DU POLYARBRE")
        print("="*70)
        print("\nArêtes du graphe:")
        for edge in self.model.edges():
            print(f"  {edge[0]} → {edge[1]}")
        
        print("\nNœuds racine (sans parents):")
        for node in self.model.nodes():
            if len(list(self.model.predecessors(node))) == 0:
                print(f"  - {node}")
        
        print("\nNœuds feuilles (sans enfants):")
        for node in self.model.nodes():
            if len(list(self.model.successors(node))) == 0:
                print(f"  - {node}")
    
    def afficher_cpds(self):
        """
        Afficher toutes les distributions de probabilité
        """
        print("\n" + "="*70)
        print("DISTRIBUTIONS DE PROBABILITÉ CONDITIONNELLES (CPD)")
        print("="*70)
        
        for cpd in self.model.get_cpds():
            print(f"\n{cpd}")
    
    def faire_inference(self, evidences=None):
        """
        Effectuer l'inférence bayésienne
        
        Args:
            evidences: Dictionnaire des évidences {variable: état}
        """
        inference = VariableElimination(self.model)
        
        print("\n" + "="*70)
        print("INFÉRENCE BAYÉSIENNE")
        print("="*70)
        
        if evidences:
            print(f"\nÉvidences observées: {evidences}")
        else:
            print("\nAucune évidence (probabilités a priori)")
        
        # Calculer P(Cambriolage | evidences)
        result = inference.query(
            variables=['Cambriolage'],
            evidence=evidences
        )
        
        print(f"\n{result}")
        
        # Extraire les probabilités
        prob_non = result.values[0]
        prob_oui = result.values[1]
        
        print(f"\nRésumé:")
        print(f"  P(Cambriolage=Non | évidences) = {prob_non:.6f} ({prob_non*100:.4f}%)")
        print(f"  P(Cambriolage=Oui | évidences) = {prob_oui:.6f} ({prob_oui*100:.4f}%)")
        
        return result
    
    def visualiser_reseau(self, save_path='resultats/polyarbre_structure.png'):
        """
        Visualiser la structure du réseau
        """
        plt.figure(figsize=(12, 8))
        
        # Utiliser networkx pour le layout
        G = nx.DiGraph(self.model.edges())
        pos = {
            'Cambriolage': (0, 2),
            'TremblementTerre': (2, 2),
            'Alarme': (1, 1),
            'JohnAppelle': (0.5, 0),
            'MaryAppelle': (1.5, 0)
        }
        
        # Dessiner le graphe
        nx.draw(
            G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=3000,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            edge_color='gray',
            width=2
        )
        
        plt.title("Polyarbre - Système d'Alarme", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Créer le dossier si nécessaire
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGraphique sauvegardé: {save_path}")
        plt.show()
    
    def scenarios_inference(self):
        """
        Tester différents scénarios d'inférence
        """
        print("\n" + "="*70)
        print("SCÉNARIOS D'INFÉRENCE")
        print("="*70)
        
        scenarios = [
            {
                'nom': 'Scénario 1: Aucune évidence',
                'evidences': None
            },
            {
                'nom': 'Scénario 2: John appelle',
                'evidences': {'JohnAppelle': 'Oui'}
            },
            {
                'nom': 'Scénario 3: Mary appelle',
                'evidences': {'MaryAppelle': 'Oui'}
            },
            {
                'nom': 'Scénario 4: John ET Mary appellent',
                'evidences': {'JohnAppelle': 'Oui', 'MaryAppelle': 'Oui'}
            },
            {
                'nom': 'Scénario 5: John appelle MAIS Mary n\'appelle pas',
                'evidences': {'JohnAppelle': 'Oui', 'MaryAppelle': 'Non'}
            },
            {
                'nom': 'Scénario 6: Alarme sonne',
                'evidences': {'Alarme': 'Oui'}
            },
            {
                'nom': 'Scénario 7: Alarme + John appelle',
                'evidences': {'Alarme': 'Oui', 'JohnAppelle': 'Oui'}
            }
        ]
        
        resultats = []
        
        for scenario in scenarios:
            print(f"\n{'─'*70}")
            print(f"⚡ {scenario['nom']}")
            print(f"{'─'*70}")
            
            result = self.faire_inference(scenario['evidences'])
            prob_cambriolage = result.values[1]  # Probabilité de cambriolage
            
            resultats.append({
                'Scénario': scenario['nom'],
                'Évidences': str(scenario['evidences']),
                'P(Cambriolage=Oui)': prob_cambriolage
            })
        
        # Afficher le tableau récapitulatif
        print("\n" + "="*70)
        print("TABLEAU RÉCAPITULATIF")
        print("="*70)
        
        import pandas as pd
        df = pd.DataFrame(resultats)
        print(df.to_string(index=False))
        
        # Sauvegarder
        df.to_csv('resultats/polyarbre_scenarios.csv', index=False)
        print("\nRésultats sauvegardés: resultats/polyarbre_scenarios.csv")


def main():
    """
    Fonction principale
    """
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "ETAPE 2: POLYARBRE - SYSTÈME D'ALARME" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    # Créer le modèle
    alarme = PolyarbreAlarme()
    
    # Afficher la structure
    alarme.afficher_structure()
    
    # Afficher les CPDs
    alarme.afficher_cpds()
    
    # Visualiser le réseau
    alarme.visualiser_reseau()
    
    # Tester différents scénarios
    alarme.scenarios_inference()
    
    print("\n" + "="*70)
    print("✅ Étape 2 terminée avec succès!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
