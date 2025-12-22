"""
EXEMPLE RÉEL: Diagnostic de Problèmes Informatiques avec la Théorie de Dempster-Shafer
=======================================================================================

Scénario: Un ordinateur présente des problèmes de performance et plusieurs tests 
sont effectués. Chaque test fournit des preuves partielles sur le diagnostic.

Hypothèses de diagnostic:
- Surchauffe CPU (H)
- Défaillance RAM (R)
- Disque Dur Défaillant (D)
- Problème Logiciel (S)
"""

import numpy as np
import pandas as pd
from itertools import combinations, chain


class DempsterShaferDiagnosis:
    """
    Système de diagnostic informatique utilisant la théorie de Dempster-Shafer
    """
    
    def __init__(self):
        # Cadre de discernement: {Surchauffe CPU, RAM Défaillante, Disque Dur, Problème Logiciel}
        self.diseases = ['Surchauffe_CPU', 'RAM_Defaillante', 'Disque_Dur_Defaillant', 'Probleme_Logiciel']
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
        """Affiche Bel et Pl pour chaque problème"""
        print(f"\n{'='*70}")
        print(f"Belief (Bel) et Plausibility (Pl) pour: {mass_function['name']}")
        print(f"{'='*70}")
        print(f"{'Problème':<25} {'Bel':<10} {'Pl':<10} {'Intervalle':<20}")
        print(f"{'-'*70}")
        
        results = []
        for disease in self.diseases:
            bel = self.calculate_belief(mass_function, [disease])
            pl = self.calculate_plausibility(mass_function, [disease])
            display_name = disease.replace('_', ' ')
            print(f"{display_name:<25} {bel:<10.4f} {pl:<10.4f} [{bel:.4f}, {pl:.4f}]")
            results.append({
                'Probleme': display_name,
                'Belief': bel,
                'Plausibility': pl,
                'Uncertainty': pl - bel
            })
        
        return results


def main():
    """
    SCÉNARIO RÉEL: Diagnostic d'un ordinateur avec problèmes de performance
    """
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*8 + "EXEMPLE RÉEL: DIAGNOSTIC INFORMATIQUE AVEC D-S" + " "*14 + "║")
    print("╚" + "="*68 + "╝\n")
    
    ds = DempsterShaferDiagnosis()
    
    print("="*70)
    print("SITUATION TECHNIQUE")
    print("="*70)
    print("""
Ordinateur: PC de Bureau Gaming
Configuration:
  - CPU: Intel Core i7-10700K
  - RAM: 16 GB DDR4 (2x8GB)
  - Disque: SSD 500GB + HDD 2TB
  - GPU: NVIDIA RTX 3070
  - Âge: 2 ans

Symptômes observés:
  - Ralentissements soudains
  - Freezes (gel) pendant 2-3 secondes
  - Bruits inhabituels
  - Température élevée au toucher
  - Début des problèmes: il y a 1 semaine

Contexte:
  - Utilisation intensive (gaming, rendering)
  - Dernier nettoyage: il y a 8 mois
  - Dernière mise à jour Windows: il y a 3 jours
    """)
    
    # ========================================================================
    # SOURCE 1: Observation visuelle et auditive
    # ========================================================================
    print("\n" + "="*70)
    print("SOURCE 1: INSPECTION VISUELLE ET AUDITIVE")
    print("="*70)
    print("""
Observations du technicien:
  - Ventilateur CPU bruyant → Possiblement encrassé ou surchauffe
  - Boîtier très chaud au toucher → Problème de refroidissement probable
  - Pas de bruit de clic du disque dur
  - Accumulation de poussière visible
    """)
    
    source1 = ds.create_mass_function(
        "Inspection Visuelle",
        {
            frozenset(['Surchauffe_CPU']): 0.55,                              # Forte suspicion
            frozenset(['Surchauffe_CPU', 'Disque_Dur_Defaillant']): 0.20,   # Possible combinaison
            frozenset(['Probleme_Logiciel']): 0.05,                          # Peu probable
            frozenset(ds.diseases): 0.20                                      # Incertitude
        }
    )
    ds.print_mass_function(source1)
    
    # ========================================================================
    # SOURCE 2: Monitoring de température (HWMonitor)
    # ========================================================================
    print("\n\n" + "="*70)
    print("SOURCE 2: MONITORING DE TEMPÉRATURE CPU")
    print("="*70)
    print("""
Résultats de HWMonitor/HWiNFO:
  - Température CPU au repos: 55°C (normal: 30-40°C)
  - Température CPU sous charge: 95°C (critique: >85°C)
  - Ventilateur CPU: 2800 RPM (devrait être ~3500 RPM)
  - Température GPU: 68°C (normale)
  
Interprétation:
  - CPU clairement en surchauffe
  - Ventilateur ne tourne pas assez vite
  - Pâte thermique peut-être sèche
    """)
    
    source2 = ds.create_mass_function(
        "Monitoring Température",
        {
            frozenset(['Surchauffe_CPU']): 0.85,                           # Très forte évidence
            frozenset(['RAM_Defaillante', 'Disque_Dur_Defaillant']): 0.05, # Peu probable
            frozenset(ds.diseases): 0.10                                    # Petite incertitude
        }
    )
    ds.print_mass_function(source2)
    
    # ========================================================================
    # COMBINAISON 1: Inspection + Monitoring
    # ========================================================================
    print("\n\n" + "="*70)
    print("COMBINAISON: Inspection Visuelle ⊕ Monitoring Température")
    print("="*70)
    
    combined1 = ds.dempster_combination(source1, source2)
    ds.print_mass_function(combined1)
    results1 = ds.print_belief_plausibility(combined1)
    
    # ========================================================================
    # SOURCE 3: Test de RAM (MemTest86)
    # ========================================================================
    print("\n\n" + "="*70)
    print("SOURCE 3: TEST MÉMOIRE RAM (MemTest86)")
    print("="*70)
    print("""
Résultats après 2 passes complètes:
  - 0 erreurs détectées
  - Tous les tests passés avec succès
  - Temps de réponse: Normal
  - Modules RAM détectés correctement (2x8GB)
  
Interprétation:
  - RAM fonctionne parfaitement
  - Problème RAM très improbable
  - Renforce hypothèse: problème matériel autre ou surchauffe
    """)
    
    source3 = ds.create_mass_function(
        "Test MemTest86",
        {
            frozenset(['Surchauffe_CPU', 'Disque_Dur_Defaillant', 'Probleme_Logiciel']): 0.75,  # Pas la RAM
            frozenset(['RAM_Defaillante']): 0.05,                                                 # Très peu probable
            frozenset(ds.diseases): 0.20                                                          # Incertitude du test
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
    # DÉCISION TECHNIQUE
    # ========================================================================
    print("\n\n" + "="*70)
    print("DIAGNOSTIC TECHNIQUE BASÉ SUR LA THÉORIE D-S")
    print("="*70)
    
    # Trouver le diagnostic le plus probable
    df = pd.DataFrame(final_results)
    df = df.sort_values('Belief', ascending=False)
    
    print("\nClassement par croyance (Belief):")
    print(df.to_string(index=False))
    
    best_diagnosis = df.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RECOMMANDÉ: {best_diagnosis['Probleme']}")
    print(f"{'='*70}")
    print(f"  Croyance (Bel):        {best_diagnosis['Belief']:.4f} ({best_diagnosis['Belief']*100:.1f}%)")
    print(f"  Plausibilité (Pl):     {best_diagnosis['Plausibility']:.4f} ({best_diagnosis['Plausibility']*100:.1f}%)")
    print(f"  Incertitude:           {best_diagnosis['Uncertainty']:.4f} ({best_diagnosis['Uncertainty']*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMANDATIONS TECHNIQUES")
    print(f"{'='*70}")
    
    if best_diagnosis['Probleme'] == 'Surchauffe CPU' and best_diagnosis['Belief'] > 0.5:
        print("""
✓ Diagnostic: SURCHAUFFE CPU - Haute confiance
✓ Actions correctives recommandées:
  1. Nettoyage complet du boîtier (enlever la poussière)
  2. Remplacement de la pâte thermique du CPU
  3. Vérification/remplacement du ventilateur CPU si nécessaire
  4. Amélioration du flux d'air (ventilateurs supplémentaires)
  5. Vérification des profils de ventilation dans le BIOS
  
✓ Coût estimé: 20-50€ (pâte thermique + nettoyage)
✓ Temps d'intervention: 1-2 heures
✓ Niveau de difficulté: Moyen (tutoriels disponibles en ligne)

✓ Suivi:
  - Tester les températures après intervention
  - Monitoring sur 24-48h
  - Si problème persiste: vérifier le montage du ventirad
        """)
    elif best_diagnosis['Probleme'] == 'RAM Defaillante' and best_diagnosis['Belief'] > 0.5:
        print("""
✓ Diagnostic: RAM DÉFAILLANTE - Haute confiance
✓ Actions correctives:
  - Remplacer les modules RAM défectueux
  - Tester avec d'autres modules pour confirmer
  - Vérifier la compatibilité des barrettes
        """)
    elif best_diagnosis['Uncertainty'] > 0.4:
        print("""
⚠ ATTENTION: Niveau d'incertitude élevé!
✓ Recommandations:
  - Effectuer des tests supplémentaires
  - Test S.M.A.R.T du disque dur (CrystalDiskInfo)
  - Vérifier les logs système Windows (Event Viewer)
  - Test de stress CPU (Prime95) sous monitoring
  - Réévaluation après premiers correctifs
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
    print("\nQue se passe-t-il si le test RAM avait détecté des ERREURS?")
    
    # Modifier la source 3
    source3_errors = ds.create_mass_function(
        "Test MemTest86 (AVEC ERREURS)",
        {
            frozenset(['RAM_Defaillante']): 0.90,      # Forte évidence pour RAM défaillante
            frozenset(ds.diseases): 0.10                # Petite incertitude
        }
    )
    
    combined_alt1 = ds.dempster_combination(source1, source2)
    final_alt = ds.dempster_combination(combined_alt1, source3_errors)
    
    ds.print_mass_function(final_alt)
    alt_results = ds.print_belief_plausibility(final_alt)
    
    df_alt = pd.DataFrame(alt_results).sort_values('Belief', ascending=False)
    print(f"\nNouveau diagnostic: {df_alt.iloc[0]['Probleme']}")
    print(f"Croyance: {df_alt.iloc[0]['Belief']:.4f} ({df_alt.iloc[0]['Belief']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
La théorie de Dempster-Shafer permet de:
  ✓ Combiner plusieurs sources d'information imparfaites
  ✓ Quantifier l'incertitude de manière explicite
  ✓ Prendre des décisions robustes même avec informations partielles
  ✓ Détecter les conflits entre sources
  
Dans ce cas:
  → Diagnostic clair: SURCHAUFFE CPU
  → Confiance élevée grâce à la convergence des tests
  → Action corrective bien définie et économique
    """)


if __name__ == "__main__":
    main()
