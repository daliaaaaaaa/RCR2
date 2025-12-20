%% CPU Fuzzy Controller - GNU Octave
% Contrôleur flou de type Mamdani
% Entrées :
%   - CPULoad (0-100 %) : charge du processeur
%   - ProcessImportance (0-10) : importance du processus
% Sortie :
%   - CPUShare (0-100 %) : part de CPU allouée
%
% Méthode :
%   - ET logique : min
%   - Agrégation : max
%   - Défuzzification : centre de gravité

clear; clc; close all;

%% 1. Univers de discours

cpuLoad     = 0:1:100;
importance  = 0:0.1:10;
cpuShare    = 0:1:100;

%% 2. Fonctions d’appartenance (définition manuelle)

% --- CPULoad ---
lowLoad    = trapmf(cpuLoad, [0 0 30 50]);
mediumLoad = trimf(cpuLoad, [30 50 70]);
highLoad   = trapmf(cpuLoad, [50 70 100 100]);

% --- ProcessImportance ---
lowImp    = trapmf(importance, [0 0 2 4]);
mediumImp = trimf(importance, [2 5 8]);
highImp   = trapmf(importance, [6 8 10 10]);

% --- CPUShare ---
veryLow  = trapmf(cpuShare, [0 0 10 25]);
low      = trimf(cpuShare, [15 30 45]);
medium   = trimf(cpuShare, [35 50 65]);
high     = trimf(cpuShare, [55 70 85]);
veryHigh = trapmf(cpuShare, [75 90 100 100]);

%% 3. Visualisation des fonctions d’appartenance

figure;
subplot(3,1,1);
plot(cpuLoad,lowLoad,cpuLoad,mediumLoad,cpuLoad,highLoad,'LineWidth',1.5);
legend("Low","Medium","High");
title("CPULoad"); ylabel("Degré");

subplot(3,1,2);
plot(importance,lowImp,importance,mediumImp,importance,highImp,'LineWidth',1.5);
legend("Low","Medium","High");
title("Process Importance"); ylabel("Degré");

subplot(3,1,3);
plot(cpuShare,veryLow,cpuShare,low,cpuShare,medium,cpuShare,high,cpuShare,veryHigh,'LineWidth',1.5);
legend("VeryLow","Low","Medium","High","VeryHigh");
title("CPUShare"); xlabel("CPU (%)"); ylabel("Degré");

%% 4. Valeurs d’entrée (exemple de simulation)

inputCPULoad    = 40;
inputImportance = 7;

%% 5. Fuzzification

mu_load_low  = interp1(cpuLoad, lowLoad, inputCPULoad);
mu_load_med  = interp1(cpuLoad, mediumLoad, inputCPULoad);
mu_load_high = interp1(cpuLoad, highLoad, inputCPULoad);

mu_imp_low  = interp1(importance, lowImp, inputImportance);
mu_imp_med  = interp1(importance, mediumImp, inputImportance);
mu_imp_high = interp1(importance, highImp, inputImportance);

%% 6. Base de règles (Mamdani)

r1 = min(mu_load_low , mu_imp_low );   % -> Medium
r2 = min(mu_load_low , mu_imp_med );   % -> High
r3 = min(mu_load_low , mu_imp_high);   % -> VeryHigh

r4 = min(mu_load_med , mu_imp_low );   % -> Low
r5 = min(mu_load_med , mu_imp_med );   % -> Medium
r6 = min(mu_load_med , mu_imp_high);   % -> High

r7 = min(mu_load_high, mu_imp_low );   % -> VeryLow
r8 = min(mu_load_high, mu_imp_med );   % -> Low
r9 = min(mu_load_high, mu_imp_high);   % -> Medium

%% 7. Agrégation des sorties floues

outVeryLow  = r7 * veryLow;
outLow      = max([r4, r8]) * low;
outMedium   = max([r1, r5, r9]) * medium;
outHigh     = max([r2, r6]) * high;
outVeryHigh = r3 * veryHigh;

aggregated = max([
    outVeryLow;
    outLow;
    outMedium;
    outHigh;
    outVeryHigh
]);

%% 8. Défuzzification (centre de gravité)

cpuShareOutput = sum(aggregated .* cpuShare) / sum(aggregated);

fprintf("\n===== Résultat =====\n");
fprintf("CPULoad = %.1f %%\n", inputCPULoad);
fprintf("Importance = %.1f\n", inputImportance);
fprintf("CPUShare allouée = %.2f %%\n", cpuShareOutput);

%% 9. Sortie floue agrégée

figure;
plot(cpuShare, aggregated, 'LineWidth', 2);
grid on;
title("Sortie floue agrégée");
xlabel("CPUShare (%)");
ylabel("Degré d'appartenance");

%% ===== Fonctions locales =====

function y = trimf(x, p)
    a = p(1); b = p(2); c = p(3);
    y = max(min((x-a)/(b-a), (c-x)/(c-b)), 0);
end

function y = trapmf(x, p)
    a = p(1); b = p(2); c = p(3); d = p(4);
    y = max(min(min((x-a)/(b-a), 1), (d-x)/(d-c)), 0);
end

