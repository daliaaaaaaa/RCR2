# TP3 - Réseaux Causaux Bayésiens
# Python Bayesian Networks Implementation

## Installation

Pour Python, nous utiliserons **pgmpy** (Probabilistic Graphical Models in Python), 
une bibliothèque moderne et complète pour les réseaux bayésiens.

```bash
pip install -r requirements.txt
```

## Contenu du TP

### ETAPE 1: Installation de la toolbox ✅
- **pgmpy**: Bibliothèque Python pour les modèles graphiques probabilistes
- Documentation: https://pgmpy.org/

### ETAPE 2: Polyarbre (Polytree)
Génération d'un réseau bayésien en forme de polyarbre avec:
- Variable(s) d'évidence
- Variable d'intérêt
- Distributions a priori pour les nœuds racine
- Distributions conditionnelles pour les autres nœuds
- Calcul de P(variable d'intérêt | évidence(s))

**Fichier**: `etape2_polyarbre.py`

### ETAPE 3: Graphe à connexions multiples
Génération d'un réseau bayésien avec connexions multiples (DAG général) avec:
- Structure plus complexe que le polyarbre
- Multiple chemins entre nœuds
- Inférence avec élimination de variables ou belief propagation

**Fichier**: `etape3_connexions_multiples.py`

### ETAPE 4: Problème réel
Formalisation et modélisation d'un problème réel sous forme de réseau Bayésien.
- Modélisation d'un scénario médical ou autre domaine
- Simulation et inférence
- Analyse de sensibilité

**Fichier**: `etape4_probleme_reel.py`

## Structure des fichiers

```
ReseauxBayesiens/
├── README.md                           # Ce fichier
├── requirements.txt                    # Dépendances Python
├── etape2_polyarbre.py                # Polyarbre simple
├── etape3_connexions_multiples.py     # DAG avec cycles multiples
├── etape4_probleme_reel.py            # Application réelle (diagnostic médical)
└── resultats/                          # Dossier pour sauvegarder les graphiques
```

## Utilisation

### Étape 2 - Polyarbre
```bash
python etape2_polyarbre.py
```

### Étape 3 - Connexions multiples
```bash
python etape3_connexions_multiples.py
```

### Étape 4 - Problème réel
```bash
python etape4_probleme_reel.py
```

## Concepts clés

### Réseau Bayésien
Un réseau bayésien est un modèle graphique probabiliste qui représente un ensemble de variables aléatoires et leurs dépendances conditionnelles via un graphe acyclique dirigé (DAG).

### Polyarbre vs Graphe général
- **Polyarbre**: Chaque nœud a au plus un chemin non-dirigé vers tout autre nœud
- **Graphe général**: Peut avoir plusieurs chemins entre les nœuds

### Inférence
Calcul de P(Variable | Evidence) en utilisant:
- **Variable Elimination**: Élimination successive de variables
- **Belief Propagation**: Propagation de messages dans le graphe
- **MCMC**: Méthodes de Monte-Carlo par chaînes de Markov

## Bibliothèques Python pour Réseaux Bayésiens

1. **pgmpy** ⭐ (Recommandé)
   - Moderne et bien maintenu
   - Documentation complète
   - Support de multiples algorithmes d'inférence

2. **PyMC** 
   - Pour inférence bayésienne avancée
   - MCMC et variational inference

3. **BayesPy**
   - Inférence variationnelle bayésienne

4. **pomegranate**
   - Rapide, basé sur Cython
   - Modèles de Markov cachés aussi

## Références

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*
- pgmpy documentation: https://pgmpy.org/
