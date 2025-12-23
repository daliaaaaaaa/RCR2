"""
ETAPE 3: Réseau Bayésien avec Connexions Multiples
====================================================

Ce script crée un réseau bayésien avec un DAG général (graphe acyclique dirigé)
contenant plusieurs chemins entre les nœuds, démontrant ainsi des connexions
multiples plus complexes qu'un simple polyarbre.

Nous utilisons pgmpy pour modéliser et effectuer l'inférence.
"""

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

def create_complex_network():
    """
    Crée un réseau bayésien avec connexions multiples.
    
    Structure du réseau:
    - A et B sont des nœuds racine
    - C dépend de A et B (connexion multiple)
    - D dépend de A et C (connexion multiple)
    - E dépend de B, C et D (connexion multiple - 3 parents)
    
    Ce réseau n'est PAS un polyarbre car il existe plusieurs chemins
    entre certains nœuds (par exemple, de A à E: A->C->E, A->D->E, A->C->D->E)
    """
    
    # Définir la structure du réseau (DAG)
    model = DiscreteBayesianNetwork([
        ('A', 'C'),
        ('B', 'C'),
        ('A', 'D'),
        ('C', 'D'),
        ('B', 'E'),
        ('C', 'E'),
        ('D', 'E')
    ])
    
    # CPD pour A (nœud racine)
    cpd_a = TabularCPD(
        variable='A',
        variable_card=2,
        values=[[0.6],  # P(A=0)
                [0.4]]  # P(A=1)
    )
    
    # CPD pour B (nœud racine)
    cpd_b = TabularCPD(
        variable='B',
        variable_card=2,
        values=[[0.7],  # P(B=0)
                [0.3]]  # P(B=1)
    )
    
    # CPD pour C (dépend de A et B)
    cpd_c = TabularCPD(
        variable='C',
        variable_card=2,
        values=[
            [0.9, 0.6, 0.5, 0.1],  # P(C=0 | A, B)
            [0.1, 0.4, 0.5, 0.9]   # P(C=1 | A, B)
        ],
        evidence=['A', 'B'],
        evidence_card=[2, 2]
    )
    
    # CPD pour D (dépend de A et C)
    cpd_d = TabularCPD(
        variable='D',
        variable_card=2,
        values=[
            [0.95, 0.7, 0.6, 0.2],  # P(D=0 | A, C)
            [0.05, 0.3, 0.4, 0.8]   # P(D=1 | A, C)
        ],
        evidence=['A', 'C'],
        evidence_card=[2, 2]
    )
    
    # CPD pour E (dépend de B, C et D - 3 parents!)
    cpd_e = TabularCPD(
        variable='E',
        variable_card=2,
        values=[
            # P(E=0 | B, C, D) pour toutes les combinaisons
            [0.99, 0.8, 0.7, 0.5, 0.6, 0.4, 0.3, 0.1],
            # P(E=1 | B, C, D)
            [0.01, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 0.9]
        ],
        evidence=['B', 'C', 'D'],
        evidence_card=[2, 2, 2]
    )
    
    # Ajouter les CPDs au modèle
    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e)
    
    # Vérifier que le modèle est valide
    assert model.check_model(), "Le modèle n'est pas valide!"
    
    return model


def visualize_network(model):
    """Visualise le réseau bayésien."""
    plt.figure(figsize=(12, 8))
    
    # Créer le graphe
    G = nx.DiGraph(model.edges())
    
    # Position des nœuds pour une meilleure visualisation
    pos = {
        'A': (0, 2),
        'B': (2, 2),
        'C': (1, 1),
        'D': (0, 0),
        'E': (2, 0)
    }
    
    # Dessiner le réseau
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray', width=2)
    
    plt.title("Réseau Bayésien avec Connexions Multiples\n" + 
              "DAG général (non-polyarbre)", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/connexions_multiples_structure.png', dpi=300, bbox_inches='tight')
    print("✓ Graphe sauvegardé: ReseauxBayesiens/images/connexions_multiples_structure.png")


def perform_inference(model):
    """Effectue plusieurs inférences sur le réseau."""
    
    # Créer l'objet d'inférence par élimination de variables
    inference = VariableElimination(model)
    
    results = []
    
    print("\n" + "="*70)
    print("INFÉRENCE DANS LE RÉSEAU À CONNEXIONS MULTIPLES")
    print("="*70)
    
    # Scénario 1: Distribution a priori de E (sans évidence)
    print("\n--- Scénario 1: Distribution a priori ---")
    print("Question: P(E) sans aucune évidence")
    result1 = inference.query(variables=['E'])
    print(result1)
    results.append({
        'Scenario': 'A priori',
        'Evidence': 'Aucune',
        'Query': 'P(E)',
        'P(E=0)': result1.values[0],
        'P(E=1)': result1.values[1]
    })
    
    # Scénario 2: E sachant A=1
    print("\n--- Scénario 2: Évidence simple ---")
    print("Question: P(E | A=1)")
    result2 = inference.query(variables=['E'], evidence={'A': 1})
    print(result2)
    results.append({
        'Scenario': 'Évidence A',
        'Evidence': 'A=1',
        'Query': 'P(E | A=1)',
        'P(E=0)': result2.values[0],
        'P(E=1)': result2.values[1]
    })
    
    # Scénario 3: E sachant A=1 et B=1 (deux évidences)
    print("\n--- Scénario 3: Évidences multiples (A et B) ---")
    print("Question: P(E | A=1, B=1)")
    result3 = inference.query(variables=['E'], evidence={'A': 1, 'B': 1})
    print(result3)
    results.append({
        'Scenario': 'Évidences A,B',
        'Evidence': 'A=1, B=1',
        'Query': 'P(E | A=1, B=1)',
        'P(E=0)': result3.values[0],
        'P(E=1)': result3.values[1]
    })
    
    # Scénario 4: D sachant E=1 (inférence remontante)
    print("\n--- Scénario 4: Inférence remontante ---")
    print("Question: P(D | E=1)")
    result4 = inference.query(variables=['D'], evidence={'E': 1})
    print(result4)
    results.append({
        'Scenario': 'Inférence remontante',
        'Evidence': 'E=1',
        'Query': 'P(D | E=1)',
        'P(E=0)': result4.values[0],
        'P(E=1)': result4.values[1]
    })
    
    # Scénario 5: C sachant E=1 et A=0
    print("\n--- Scénario 5: Évidence mixte ---")
    print("Question: P(C | E=1, A=0)")
    result5 = inference.query(variables=['C'], evidence={'E': 1, 'A': 0})
    print(result5)
    results.append({
        'Scenario': 'Évidence mixte',
        'Evidence': 'E=1, A=0',
        'Query': 'P(C | E=1, A=0)',
        'P(E=0)': result5.values[0],
        'P(E=1)': result5.values[1]
    })
    
    # Scénario 6: Inférence avec toutes les évidences intermédiaires
    print("\n--- Scénario 6: Chemin complet d'évidence ---")
    print("Question: P(E | A=1, C=1)")
    result6 = inference.query(variables=['E'], evidence={'A': 1, 'C': 1})
    print(result6)
    results.append({
        'Scenario': 'Chemin complet',
        'Evidence': 'A=1, C=1',
        'Query': 'P(E | A=1, C=1)',
        'P(E=0)': result6.values[0],
        'P(E=1)': result6.values[1]
    })
    
    return results


def save_results(results):
    """Sauvegarde les résultats dans un fichier CSV."""
    df = pd.DataFrame(results)
    df.to_csv('resultats/connexions_multiples_results.csv', index=False)
    print(f"\n✓ Résultats sauvegardés: ReseauxBayesiens/resultats/connexions_multiples_results.csv")
    print("\nTableau des résultats:")
    print(df.to_string(index=False))


def analyze_network_properties(model):
    """Analyse les propriétés du réseau."""
    print("\n" + "="*70)
    print("ANALYSE DES PROPRIÉTÉS DU RÉSEAU")
    print("="*70)
    
    print(f"\nNombre de nœuds: {len(model.nodes())}")
    print(f"Nœuds: {list(model.nodes())}")
    
    print(f"\nNombre d'arêtes: {len(model.edges())}")
    print(f"Arêtes: {list(model.edges())}")
    
    print("\nNœuds racine (sans parents):")
    roots = [node for node in model.nodes() if len(list(model.predecessors(node))) == 0]
    print(f"  {roots}")
    
    print("\nNœuds feuilles (sans enfants):")
    leaves = [node for node in model.nodes() if len(list(model.successors(node))) == 0]
    print(f"  {leaves}")
    
    print("\nNombre de parents par nœud:")
    for node in model.nodes():
        parents = list(model.predecessors(node))
        print(f"  {node}: {len(parents)} parent(s) {parents if parents else ''}")
    
    print("\n✓ Ce réseau contient des CONNEXIONS MULTIPLES:")
    print("  - Le nœud E a 3 parents (B, C, D)")
    print("  - Il existe plusieurs chemins de A vers E:")
    print("    * A → C → E")
    print("    * A → D → E")
    print("    * A → C → D → E")
    print("  - Ce n'est donc PAS un polyarbre!")
    
    # Vérifier l'acyclicité
    G = nx.DiGraph(model.edges())
    print(f"\n✓ Le graphe est acyclique: {nx.is_directed_acyclic_graph(G)}")


def main():
    """Fonction principale."""
    print("="*70)
    print("ÉTAPE 3: RÉSEAU BAYÉSIEN AVEC CONNEXIONS MULTIPLES")
    print("="*70)
    
    # Créer le réseau
    print("\n1. Création du réseau bayésien...")
    model = create_complex_network()
    print("✓ Réseau créé et validé avec succès!")
    
    # Analyser les propriétés
    analyze_network_properties(model)
    
    # Visualiser
    print("\n2. Visualisation du réseau...")
    visualize_network(model)
    
    # Effectuer l'inférence
    print("\n3. Inférence bayésienne...")
    results = perform_inference(model)
    
    # Sauvegarder les résultats
    print("\n4. Sauvegarde des résultats...")
    save_results(results)
    
    print("\n" + "="*70)
    print("✓ ÉTAPE 3 TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\nFichiers générés:")
    print("  - resultats/connexions_multiples_structure.png")
    print("  - resultats/connexions_multiples_results.csv")


if __name__ == "__main__":
    main()
