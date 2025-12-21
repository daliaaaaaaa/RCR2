"""
Modèle de la Théorie des Fonctions de Croyance (Dempster-Shafer)
Modélisation d'une base de connaissances stratifiée avec les fonctions de croyance
"""

import numpy as np
import pandas as pd
import csv
import os

# Cadre de discernement : Valeurs booléennes pour les variables {a, b, c, d, e, f}
# Chaque variable peut être Vraie ou Fausse

class BeliefFunctionKB:
    """
    Modélise une base de connaissances stratifiée avec la théorie de Dempster-Shafer
    """
    
    def __init__(self):
        # Définir la base de connaissances stratifiée
        # Correspondance : a=1, b=2, c=3, d=4, e=5, f=6
        self.strata = [
            {"weight": 0.9,  "clauses": [[1, -2, 4, 5], [-2, 3]]},
            {"weight": 0.78, "clauses": [[2, -3]]},
            {"weight": 0.65, "clauses": [[1, -2, 4, 5], [-2, 3], [-1, 2, 4]]},
            {"weight": 0.60, "clauses": [[1, -2, 4, 5], [2, 3]]},
            {"weight": 0.58, "clauses": [[-1, -2, 4]]},
            {"weight": 0.43, "clauses": [[1, -2, 4, 5], [-2, 3], [1, 2, 4]]},
            {"weight": 0.36, "clauses": [[1, 5], [-2, 3, 6]]},
            {"weight": 0.26, "clauses": [[-1, 4]]},
            {"weight": 0.14, "clauses": [[4]]}
        ]
        
        self.variables = ['a', 'b', 'c', 'd', 'e', 'f']
        self.variable_map = {i+1: var for i, var in enumerate(self.variables)}
        
    def create_mass_function_for_variable(self, var_name):
        """
        Crée une fonction de masse pour une variable spécifique basée sur la base de connaissances
        
        Args:
            var_name: Nom de la variable ('a', 'b', 'c', 'd', 'e', 'f')
        
        Returns:
            MassFunction: Croyance et plausibilité pour la variable
        """
        var_idx = self.variables.index(var_name) + 1
        
        # Calculer les affectations de masse
        # m({var}) = croyance que var est vraie
        # m({¬var}) = croyance que var est fausse  
        # m({var, ¬var}) = incertitude
        
        # Utiliser les poids des strates pour déterminer la croyance
        belief_true = 0
        belief_false = 0
        
        for stratum in self.strata:
            weight = stratum['weight']
            for clause in stratum['clauses']:
                # Vérifier si la variable apparaît dans la clause
                if var_idx in clause:
                    belief_true += weight / len(self.strata)
                elif -var_idx in clause:
                    belief_false += weight / len(self.strata)
        
        # Normaliser
        total = belief_true + belief_false
        if total > 0:
            belief_true /= total
            belief_false /= total
        
        uncertainty = 1 - (belief_true + belief_false)
        
        # Créer la fonction de masse
        # Cadre de discernement pour cette variable : {V, F}
        mass_dict = {}
        
        if belief_true > 0.001:
            mass_dict[f'{var_name}=V'] = belief_true
        if belief_false > 0.001:
            mass_dict[f'{var_name}=F'] = belief_false
        if uncertainty > 0.001:
            mass_dict[f'{var_name}=?'] = uncertainty
            
        return mass_dict, belief_true, belief_false, uncertainty
    
    def create_mass_from_strata(self, var_name, interest_value):
        """
        Crée une fonction de masse basée sur la valeur d'intérêt calculée
        
        Args:
            var_name: Nom de la variable
            interest_value: Valeur d'intérêt calculée des exercices précédents
        
        Returns:
            Dictionnaire représentant la fonction de masse
        """
        if interest_value > 0:
            # Valeur d'intérêt plus élevée = croyance plus forte
            belief = interest_value
            uncertainty = 1 - belief
            
            return {
                f'{var_name}=V': belief,
                f'{var_name}=?': uncertainty
            }
        else:
            # Aucune preuve
            return {
                f'{var_name}=?': 1.0
            }
    
    def dempster_combination(self, mass1, mass2):
        """
        Combine deux fonctions de masse en utilisant la règle de combinaison de Dempster
        
        Args:
            mass1, mass2: Dictionnaires représentant les fonctions de masse
        
        Returns:
            Fonction de masse combinée
        """
        combined = {}
        conflict = 0
        
        # Calculer l'intersection de tous les ensembles focaux
        for key1, val1 in mass1.items():
            for key2, val2 in mass2.items():
                # Calculer l'intersection
                if key1 == key2:
                    # Même élément focal
                    intersection = key1
                elif '?' in key1:
                    intersection = key2
                elif '?' in key2:
                    intersection = key1
                else:
                    # Conflit : affectations différentes
                    conflict += val1 * val2
                    continue
                
                # Ajouter la masse
                if intersection in combined:
                    combined[intersection] += val1 * val2
                else:
                    combined[intersection] = val1 * val2
        
        # Normaliser par (1 - conflit)
        if conflict < 1:
            for key in combined:
                combined[key] /= (1 - conflict)
        
        return combined, conflict
    
    def calculate_belief_plausibility(self, mass_func, hypothesis):
        """
        Calcule la croyance et la plausibilité pour une hypothèse
        
        Args:
            mass_func: Dictionnaire représentant la fonction de masse
            hypothesis: Hypothèse à évaluer (ex: 'a=V')
        
        Returns:
            Tuple (croyance, plausibilité)
        """
        belief = 0
        plausibility = 0
        
        for focal_set, mass in mass_func.items():
            # Croyance : somme des masses des sous-ensembles de l'hypothèse
            if hypothesis in focal_set and '?' not in focal_set:
                belief += mass
            
            # Plausibilité : somme des masses qui intersectent avec l'hypothèse
            if hypothesis in focal_set or '?' in focal_set:
                plausibility += mass
        
        return belief, plausibility
    
    def generate_report(self):
        """
        Génère un rapport complet des fonctions de croyance pour toutes les variables
        """
        print("=" * 80)
        print("MODÈLE DE FONCTIONS DE CROYANCE - Base de Connaissances Stratifiée")
        print("=" * 80)
        print()
        
        # Charger les valeurs d'intérêt des exercices précédents
        interest_values = {}
        interest_file = os.path.join('..', 'PossibilityTheory', 'interest_values.csv')
        
        try:
            with open(interest_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var_idx = int(row['Variable'])
                    interest_values[self.variable_map[var_idx]] = float(row['Interest Value'])
        except:
            print("Note : Utilisation des valeurs d'intérêt par défaut (fichier introuvable)")
            interest_values = {'a': 0, 'b': 0.6, 'c': 0.6, 'd': 0.14, 'e': 0, 'f': 0}
        
        results = []
        
        for var in self.variables:
            print(f"\n{'='*60}")
            print(f"Variable: {var}")
            print(f"{'='*60}")
            
            # Méthode 1 : Direct depuis la structure de la base de connaissances
            mass_dict1, bel_t, bel_f, unc = self.create_mass_function_for_variable(var)
            
            print(f"\n1. Fonction de Masse depuis la Structure BC :")
            for focal_set, mass in mass_dict1.items():
                print(f"   m({focal_set}) = {mass:.4f}")
            
            # Méthode 2 : Depuis les valeurs d'intérêt
            interest_val = interest_values.get(var, 0)
            mass_dict2 = self.create_mass_from_strata(var, interest_val)
            
            print(f"\n2. Fonction de Masse depuis la Valeur d'Intérêt ({interest_val}) :")
            for focal_set, mass in mass_dict2.items():
                print(f"   m({focal_set}) = {mass:.4f}")
            
            # Calculer la croyance et la plausibilité
            bel_true, pl_true = self.calculate_belief_plausibility(mass_dict2, f'{var}=V')
            bel_false, pl_false = self.calculate_belief_plausibility(mass_dict2, f'{var}=F')
            
            print(f"\n3. Croyance et Plausibilité :")
            print(f"   Bel({var}=Vrai)  = {bel_true:.4f}")
            print(f"   Pl({var}=Vrai)   = {pl_true:.4f}")
            print(f"   Bel({var}=Faux)  = {bel_false:.4f}")
            print(f"   Pl({var}=Faux)   = {pl_false:.4f}")
            print(f"   Intervalle d'Incertitude: [{bel_true:.4f}, {pl_true:.4f}]")
            
            results.append({
                'Variable': var,
                'Valeur_Interet': interest_val,
                'Bel_Vrai': bel_true,
                'Pl_Vrai': pl_true,
                'Bel_Faux': bel_false,
                'Pl_Faux': pl_false,
                'Incertitude': pl_true - bel_true
            })
        
        # Sauvegarder les résultats
        df = pd.DataFrame(results)
        df.to_csv('resultats_fonctions_croyance.csv', index=False)
        print("\n" + "="*80)
        print("Résultats sauvegardés dans : resultats_fonctions_croyance.csv")
        print("="*80)
        
        return results
    
    def demonstrate_combination(self):
        """
        Démontre la règle de combinaison de Dempster
        """
        print("\n" + "="*80)
        print("RÈGLE DE COMBINAISON DE DEMPSTER - Exemple")
        print("="*80)
        
        # Exemple : Combiner les preuves pour la variable 'd' depuis deux sources
        print("\nCombinaison des preuves pour la variable 'd' :")
        
        # Source 1 : Depuis la valeur d'intérêt (0.14)
        mass1 = {'d=V': 0.14, 'd=?': 0.86}
        print("\nSource 1 (Valeur d'Intérêt) :")
        for k, v in mass1.items():
            print(f"   m1({k}) = {v:.4f}")
        
        # Source 2 : Preuve forte depuis le poids des strates
        mass2 = {'d=V': 0.60, 'd=?': 0.40}
        print("\nSource 2 (Analyse des Strates) :")
        for k, v in mass2.items():
            print(f"   m2({k}) = {v:.4f}")
        
        # Combiner
        combined, conflict = self.dempster_combination(mass1, mass2)
        
        print(f"\nFonction de Masse Combinée (Conflit = {conflict:.4f}) :")
        for k, v in combined.items():
            print(f"   m({k}) = {v:.4f}")
        
        bel, pl = self.calculate_belief_plausibility(combined, 'd=V')
        print(f"\nÉvaluation Finale :")
        print(f"   Bel(d=Vrai) = {bel:.4f}")
        print(f"   Pl(d=Vrai)  = {pl:.4f}")
        print(f"   Incertitude = {pl - bel:.4f}")


def main():
    """
    Fonction principale pour exécuter le modèle de fonctions de croyance
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "MODÈLE DE FONCTIONS DE CROYANCE DEMPSTER-SHAFER" + " "*16 + "║")
    print("╚" + "="*78 + "╝")
    
    # Créer le modèle
    model = BeliefFunctionKB()
    
    # Générer un rapport complet
    results = model.generate_report()
    
    # Démontrer la combinaison
    model.demonstrate_combination()
    
    print("\n" + "="*80)
    print("Analyse Terminée !")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
