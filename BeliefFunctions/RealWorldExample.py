"""
EXEMPLE RÉEL: Diagnostic Médical avec la Théorie de Dempster-Shafer
====================================================================

Scénario: Un patient présente des symptômes et plusieurs tests médicaux 
sont effectués. Chaque test fournit des preuves partielles sur le diagnostic.

Hypothèses de diagnostic:
- Grippe (F)
- COVID-19 (C)
- Allergie (A)
- Rhume (R)
"""

import numpy as np
import pandas as pd
from itertools import combinations, chain


class DempsterShaferDiagnosis:
    """
    Système de diagnostic médical utilisant la théorie de Dempster-Shafer
    """
    
    def __init__(self):
        # Cadre de discernement: {Grippe, COVID, Allergie, Rhume}
        self.diseases = ['Grippe', 'COVID-19', 'Allergie', 'Rhume']
        self.frame = set(self.diseases)
        
    def powerset(self, iterable):
        """Génère l'ensemble des parties (powerset)"""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def format_hypothesis(self, hyp_set):
        """Formate une hypothèse pour l'affichage"""
        if len(hyp_set) == 0:
            return "∅"
        elif len(hyp_set) == len(self.diseases):
            return "Ω (Ignorance totale)"
        else:
            return "{" + ", ".join(sorted(hyp_set)) + "}"
    
    def create_mass_function(self, name, masses):
        """
        Crée une fonction de masse
        masses: dict avec clés = frozenset des hypothèses, valeurs = masse
        """
        total = sum(masses.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"La somme des masses doit être 1.0, obtenu: {total}")
        return {"name": name, "masses": masses}
    
    def dempster_combination(self, mf1, mf2):
        """
        Règle de combinaison de Dempster pour combiner deux fonctions de masse
        """
        combined = {}
        conflict = 0.0
        
        # Pour chaque paire de masses
        for hyp1, mass1 in mf1['masses'].items():
            for hyp2, mass2 in mf2['masses'].items():
                # Intersection des hypothèses
                intersection = hyp1 & hyp2
                
                if len(intersection) == 0:
                    # Conflit: hypothèses contradictoires
                    conflict += mass1 * mass2
                else:
                    # Combiner les masses
                    if intersection in combined:
                        combined[intersection] += mass1 * mass2
                    else:
                        combined[intersection] = mass1 * mass2
        
        # Normalisation par (1 - K) où K est le conflit
        if conflict < 1.0:
            for hyp in combined:
                combined[hyp] /= (1 - conflict)
        else:
            raise ValueError("Conflit total! Les sources sont complètement contradictoires.")
        
        return {
            "name": f"{mf1['name']} ⊕ {mf2['name']}",
            "masses": combined,
            "conflict": conflict
        }
    
    def calculate_belief(self, mass_function, hypothesis):
        """
        Calcule Bel(A) = somme des masses de tous les sous-ensembles de A
        """
        belief = 0.0
        hyp_set = frozenset(hypothesis) if isinstance(hypothesis, (list, set)) else frozenset([hypothesis])
        
        for focal_set, mass in mass_function['masses'].items():
            if focal_set.issubset(hyp_set):
                belief += mass
        
        return belief
    
    def calculate_plausibility(self, mass_function, hypothesis):
        """
        Calcule Pl(A) = somme des masses qui intersectent A
        """
        plausibility = 0.0
        hyp_set = frozenset(hypothesis) if isinstance(hypothesis, (list, set)) else frozenset([hypothesis])
        
        for focal_set, mass in mass_function['masses'].items():
            if len(focal_set & hyp_set) > 0:
                plausibility += mass
        
        return plausibility
    
    def print_mass_function(self, mass_function):
        """Affiche une fonction de masse de manière lisible"""
        print(f"\n{'='*70}")
        print(f"Source: {mass_function['name']}")
        print(f"{'='*70}")
        
        # Trier par masse décroissante
        sorted_masses = sorted(mass_function['masses'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for hyp, mass in sorted_masses:
            print(f"  m({self.format_hypothesis(hyp):40s}) = {mass:.4f}")
        
        if 'conflict' in mass_function:
            print(f"\n  Conflit détecté: {mass_function['conflict']:.4f}")
    
    def print_belief_plausibility(self, mass_function):
        """Affiche Bel et Pl pour chaque maladie"""
        print(f"\n{'='*70}")
        print(f"Belief (Bel) et Plausibility (Pl) pour: {mass_function['name']}")
        print(f"{'='*70}")
        print(f"{'Maladie':<20} {'Bel':<10} {'Pl':<10} {'Intervalle':<20}")
        print(f"{'-'*70}")
        
        results = []
        for disease in self.diseases:
            bel = self.calculate_belief(mass_function, [disease])
            pl = self.calculate_plausibility(mass_function, [disease])
            print(f"{disease:<20} {bel:<10.4f} {pl:<10.4f} [{bel:.4f}, {pl:.4f}]")
            results.append({
                'Maladie': disease,
                'Belief': bel,
                'Plausibility': pl,
                'Uncertainty': pl - bel
            })
        
        return results


def main():
    """
    SCÉNARIO RÉEL: Diagnostic d'un patient avec symptômes respiratoires
    """
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "EXEMPLE RÉEL: DIAGNOSTIC MÉDICAL AVEC D-S" + " "*17 + "║")
    print("╚" + "="*68 + "╝\n")
    
    ds = DempsterShaferDiagnosis()
    
    print("="*70)
    print("SITUATION CLINIQUE")
    print("="*70)
    print("""
Patient: Jean, 35 ans
Symptômes observés:
  - Fièvre (38.5°C)
  - Toux sèche
  - Fatigue
  - Léger mal de gorge
  - Début des symptômes: il y a 2 jours

Contexte:
  - Saison: Hiver
  - Exposition: Contact avec un collègue malade il y a 5 jours
  - Vaccination COVID: À jour
    """)
    
    # ========================================================================
    # SOURCE 1: Examen clinique du médecin
    # ========================================================================
    print("\n" + "="*70)
    print("SOURCE 1: EXAMEN CLINIQUE")
    print("="*70)
    print("""
Le médecin observe:
  - Fièvre modérée + toux sèche → Compatible avec Grippe ou COVID
  - Pas d'écoulement nasal → Moins probable: Rhume ou Allergie
  - Saison hivernale → Grippe très possible
    """)
    
    source1 = ds.create_mass_function(
        "Examen Clinique",
        {
            frozenset(['Grippe', 'COVID-19']): 0.60,  # Forte suspicion
            frozenset(['Rhume']): 0.05,                # Peu probable
            frozenset(['Allergie']): 0.05,             # Peu probable
            frozenset(ds.diseases): 0.30               # Incertitude
        }
    )
    ds.print_mass_function(source1)
    
    # ========================================================================
    # SOURCE 2: Test rapide antigénique COVID
    # ========================================================================
    print("\n\n" + "="*70)
    print("SOURCE 2: TEST RAPIDE ANTIGÉNIQUE COVID-19")
    print("="*70)
    print("""
Résultat: NÉGATIF
Fiabilité du test:
  - Sensibilité: 85% (15% de faux négatifs)
  - Spécificité: 98% (2% de faux positifs)
  
Interprétation:
  - Test négatif → Forte évidence CONTRE COVID
  - Mais test non parfait → On garde une petite incertitude
    """)
    
    source2 = ds.create_mass_function(
        "Test Antigénique",
        {
            frozenset(['Grippe', 'Allergie', 'Rhume']): 0.80,  # Probablement pas COVID
            frozenset(['COVID-19']): 0.05,                      # Petite possibilité (faux négatif)
            frozenset(ds.diseases): 0.15                        # Incertitude du test
        }
    )
    ds.print_mass_function(source2)
    
    # ========================================================================
    # COMBINAISON 1: Examen + Test antigénique
    # ========================================================================
    print("\n\n" + "="*70)
    print("COMBINAISON: Examen Clinique ⊕ Test Antigénique")
    print("="*70)
    
    combined1 = ds.dempster_combination(source1, source2)
    ds.print_mass_function(combined1)
    results1 = ds.print_belief_plausibility(combined1)
    
    # ========================================================================
    # SOURCE 3: Analyse sanguine (prise de sang)
    # ========================================================================
    print("\n\n" + "="*70)
    print("SOURCE 3: ANALYSE SANGUINE")
    print("="*70)
    print("""
Résultats:
  - Leucocytes: 8500/mm³ (légèrement élevés)
  - Lymphocytes: Un peu bas
  - CRP (protéine C-réactive): 25 mg/L (légèrement élevée)
  
Interprétation:
  - Pattern inflammatoire → Compatible avec infection virale
  - Profil typique de la Grippe
  - Peut aussi être COVID (mais test négatif)
  - Pas typique d'allergie
    """)
    
    source3 = ds.create_mass_function(
        "Analyse Sanguine",
        {
            frozenset(['Grippe']): 0.65,                # Très compatible
            frozenset(['COVID-19']): 0.15,              # Possible
            frozenset(['Rhume']): 0.10,                 # Moins probable
            frozenset(ds.diseases): 0.10                # Incertitude
        }
    )
    ds.print_mass_function(source3)
    
    # ========================================================================
    # COMBINAISON FINALE: Toutes les sources
    # ========================================================================
    print("\n\n" + "="*70)
    print("COMBINAISON FINALE: Toutes les sources d'information")
    print("="*70)
    
    final_combined = ds.dempster_combination(combined1, source3)
    ds.print_mass_function(final_combined)
    final_results = ds.print_belief_plausibility(final_combined)
    
    # ========================================================================
    # DÉCISION MÉDICALE
    # ========================================================================
    print("\n\n" + "="*70)
    print("DÉCISION MÉDICALE BASÉE SUR LA THÉORIE D-S")
    print("="*70)
    
    # Trouver le diagnostic le plus probable
    df = pd.DataFrame(final_results)
    df = df.sort_values('Belief', ascending=False)
    
    print("\nClassement par croyance (Belief):")
    print(df.to_string(index=False))
    
    best_diagnosis = df.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RECOMMANDÉ: {best_diagnosis['Maladie']}")
    print(f"{'='*70}")
    print(f"  Croyance (Bel):        {best_diagnosis['Belief']:.4f} ({best_diagnosis['Belief']*100:.1f}%)")
    print(f"  Plausibilité (Pl):     {best_diagnosis['Plausibility']:.4f} ({best_diagnosis['Plausibility']*100:.1f}%)")
    print(f"  Incertitude:           {best_diagnosis['Uncertainty']:.4f} ({best_diagnosis['Uncertainty']*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMANDATIONS CLINIQUES")
    print(f"{'='*70}")
    
    if best_diagnosis['Maladie'] == 'Grippe' and best_diagnosis['Belief'] > 0.5:
        print("""
✓ Diagnostic: GRIPPE (Influenza) - Haute confiance
✓ Traitement recommandé:
  - Repos au lit
  - Hydratation abondante
  - Paracétamol pour la fièvre
  - Antiviraux si < 48h depuis début (ex: Oseltamivir)
✓ Suivi:
  - Revoir si aggravation ou pas d'amélioration en 3-4 jours
  - Test PCR COVID si doute persiste
        """)
    elif best_diagnosis['Uncertainty'] > 0.4:
        print("""
⚠ ATTENTION: Niveau d'incertitude élevé!
✓ Recommandations:
  - Effectuer des tests supplémentaires
  - Test PCR COVID-19 (plus fiable que l'antigénique)
  - Réévaluation dans 24-48h
  - Traitement symptomatique en attendant
        """)
    
    # Sauvegarder les résultats
    df.to_csv('diagnostic_results.csv', index=False)
    print(f"\n{'='*70}")
    print("Résultats sauvegardés dans: diagnostic_results.csv")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # ANALYSE DE SENSIBILITÉ
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSE DE SENSIBILITÉ")
    print("="*70)
    print("\nQue se passe-t-il si le test COVID était POSITIF?")
    
    # Modifier la source 2
    source2_positive = ds.create_mass_function(
        "Test Antigénique (POSITIF)",
        {
            frozenset(['COVID-19']): 0.95,      # Forte évidence pour COVID
            frozenset(ds.diseases): 0.05        # Petite incertitude
        }
    )
    
    combined_alt1 = ds.dempster_combination(source1, source2_positive)
    final_alt = ds.dempster_combination(combined_alt1, source3)
    
    ds.print_mass_function(final_alt)
    alt_results = ds.print_belief_plausibility(final_alt)
    
    df_alt = pd.DataFrame(alt_results).sort_values('Belief', ascending=False)
    print(f"\nNouveau diagnostic: {df_alt.iloc[0]['Maladie']}")
    print(f"Croyance: {df_alt.iloc[0]['Belief']:.4f} ({df_alt.iloc[0]['Belief']*100:.1f}%)")


if __name__ == "__main__":
    main()
