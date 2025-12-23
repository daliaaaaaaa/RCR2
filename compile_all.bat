@echo off
REM Script pour compiler tous les fichiers LaTeX du projet

echo ========================================
echo Compilation de TOUS les fichiers LaTeX
echo ========================================

echo.
echo [1/2] Compilation de BeliefFunctions...
cd BeliefFunctions
pdflatex -interaction=nonstopmode belief_functions.tex
pdflatex -interaction=nonstopmode belief_functions.tex
echo ✓ belief_functions.pdf créé !

cd ..

echo.
echo [2/2] Compilation de ReseauxBayesiens...
cd ReseauxBayesiens
pdflatex -interaction=nonstopmode reseaux_bayesiens.tex
pdflatex -interaction=nonstopmode reseaux_bayesiens.tex
echo ✓ reseaux_bayesiens.pdf créé !

cd ..

echo.
echo ========================================
echo ✓ TOUS les PDFs ont été générés !
echo ========================================
echo - BeliefFunctions/belief_functions.pdf
echo - ReseauxBayesiens/reseaux_bayesiens.pdf
echo ========================================

pause
