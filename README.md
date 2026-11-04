# nanogpt-numpy

Un petit GPT au niveau caractère, écrit en NumPy et rien d'autre, sans
PyTorch ni différentiation automatique. Chaque couche a sa passe avant et sa
rétropropagation écrites à la main, et chaque gradient est vérifié par
différences finies. Le modèle s'entraîne en 25 minutes sur deux cœurs de CPU,
sur le texte des *Misérables* de Victor Hugo.

Code : https://github.com/pinpin-http/nanogpt-numpy

![Courbes de loss](docs/loss.png)

## Pourquoi

Avec un framework, `loss.backward()` fait tout le travail et on finit par
oublier ce qu'il calcule. Ici, il faut dériver soi-même une LayerNorm, un
softmax masqué ou une attention multi-têtes, puis le prouver avec un test. Le
modèle reste assez petit pour tourner sur un portable, mais il contient toutes
les pièces d'un transformer décodeur.

## Résultats

Mesures sur toute la validation, soit 308 927 caractères à prédire, chacun
compté une fois :

| | nats / caractère | bits / caractère |
|---|---:|---:|
| Unigramme (fréquence de chaque caractère) | 3,111 | 4,488 |
| Bigramme (d'après le caractère précédent) | 2,379 | 3,432 |
| nanogpt-numpy (823 296 paramètres) | 1,698 | 2,450 |

Le modèle a besoin de 2,45 bits par caractère, contre 3,43 pour le bigramme,
soit 29 % de moins. Il est évalué par fenêtres glissantes de 128 caractères
avec un pas de 64 : hormis le tout début, chaque caractère a au moins 64
caractères de contexte derrière lui. Les baselines utilisent un lissage additif dont le
coefficient est choisi pour minimiser leur loss sur la validation elle-même,
ce qui les avantage.

Conditions du run publié (`docs/results.json`) : 4 couches, 4 têtes,
dimension 128, contexte de 128 caractères, batchs de 32 séquences, poids de
sortie liés aux embeddings. Budget de 25 minutes sur 2 threads, sur une
machine qui faisait tourner en même temps un rendu vidéo (charge moyenne
autour de 22 pour 8 cœurs). Le run a fait 796 itérations, 1,9 s par pas
évaluations comprises, soit 3,3 millions de caractères vus : à peu près une
passe sur le texte d'entraînement. Un essai court sur la même machine, moins
chargée, tournait à 0,47 s par pas. La courbe n'avait pas fini de descendre :
sur une machine libre, le même budget irait sans doute plus loin.

Sur la courbe, la loss marque un palier vers 3,45 bits par caractère, le
niveau du bigramme, entre les itérations 80 et 200, puis repart à la baisse.
À la dernière estimation, la loss d'entraînement (1,743 nats) et celle de
validation (1,731) sont au même niveau : pas de sur-apprentissage après une
passe.

Génération de 300 caractères, même graine, même machine chargée : 185
caractères par seconde avec le cache KV, 42 sans, soit 4,3 fois plus vite.

### Échantillons

Température 0,8, top-k 20, graines 1 à 3, 400 caractères générés après
l'amorce, sans retouche (`docs/samples.json`) :

```text
Javert regardait à la marde.
Elle sempère du coupt, il un semperit sa sous autre mour, il souvrait comme de la quille sembéde de ferme, par de la vouter, cela serrendant.
Cetette enteus pers lui des fûtses de la cette sos qu'étoir de lui bon.
C'est sest chapent qui le l'aingre redése des brices le sous d'avermie de la vis de la ricades enra cette somme par le dont enfant s'étroirait de l'été à une peine était d
```

```text
La nuit était dire, ces chermés de sont par souter ses sires carivurés. De l'avait lui plus l'ava d'y avain de ces nes, dit des soix ces insores, avait dagues le vrener et premirs, viendre le troin qu'il avait passais. Dis vien avient été la ces caries, une voir des des bennier, et arrées il avait les biens des fermots, et les restres clorques, sa retoure.
L'est dornemect ritisté direntaint maigne la voive. À 
```

```text
--Monsieur, dit dit la relle, une d'épasser ou même prit pour de l'ont les cotes savenuts d'aindinés.
Il se rignand monaire et ferenures. On le parrait ces partes de tenspées.
Il y a lieur persent et à la nuit. La pricition passé dire l'entant. Et l'hose. Deux pournes et c'était le trouver, livair des finoirs nous devant ce metit à m'enouper se doire qui en dit et caronne à la parte. Sa pemance. Ils repuis les f
```

Au bout de 796 pas, le modèle place bien les espaces, la ponctuation, les
apostrophes et les retours à la ligne entre paragraphes. Il orthographie
correctement les mots courts et fréquents (« il avait », « de la »,
« c'était »), mais les mots longs sont inventés : ils ont l'air français sans
l'être.

## Comment c'est construit

| Fichier | Contenu |
|---|---|
| `nanogpt/layers.py` | Linear, Embedding, LayerNorm, GELU, attention causale multi-têtes, MLP, bloc, softmax + entropie croisée |
| `nanogpt/model.py` | GPT : embeddings de caractères et de positions, blocs, tête de sortie liée aux embeddings, cache KV, sauvegarde |
| `nanogpt/optim.py` | AdamW, warmup puis cosinus, écrêtage de la norme globale du gradient |
| `nanogpt/data.py` | téléchargement Gutenberg, nettoyage, tokenizer caractère, découpage train/val, tirage des batches |
| `nanogpt/train.py` | boucle d'entraînement avec budget de temps, rapport final |
| `nanogpt/evaluate.py` | loss sur un split entier, baselines unigramme et bigramme |
| `nanogpt/generate.py` | échantillonnage (température, top-k), génération avec ou sans cache |
| `nanogpt/plot.py` | courbes de loss (matplotlib, backend Agg) |
| `nanogpt/cli.py` | `python -m nanogpt download / train / baselines / sample / report` |

Quelques choix de conception :

- Chaque couche garde pendant `forward` ce dont `backward` aura besoin.
  `backward` renvoie le gradient par rapport à l'entrée et *ajoute* les
  gradients des paramètres à ceux déjà présents. Grâce à ça, la matrice
  d'embedding sert aussi de tête de sortie : les deux contributions
  s'additionnent sans code spécial.
- Blocs pre-LN comme GPT-2, GELU en approximation tanh, initialisation
  N(0, 0,02) avec les projections du flux résiduel divisées par
  √(2 × nombre de couches).
- Dans l'attention, les positions futures reçoivent un score de −∞, donc un
  poids nul après softmax, et leur gradient est nul sans traitement à part. Le
  jacobien du softmax est appliqué ligne par ligne, sans matrice explicite.
- AdamW avec β = (0,9 ; 0,95) et un weight decay de 0,1 appliqué aux matrices
  et aux embeddings, pas aux biais ni aux gains. Taux d'apprentissage de 2e-3
  à 2e-4, 100 itérations de warmup, écrêtage de la norme à 1.
- L'entraînement a un budget de temps. Le cosinus du taux d'apprentissage
  suit la part du budget consommée (en temps ou en itérations, la plus avancée
  des deux) : si la machine ralentit, on fait moins de pas mais la
  décroissance va quand même à son terme.
- Le cache KV garde les clés et valeurs déjà calculées, si bien qu'à chaque
  nouveau caractère on ne calcule qu'une position. Comme les positions sont
  absolues, quand le contexte atteint 128 caractères, le cache est reconstruit
  à partir des 64 derniers.

## Tests

`make test` lance 33 tests en quelques secondes, sans télécharger le corpus,
et le workflow GitHub Actions lance les mêmes à chaque push.

- Gradient de chaque couche, en float64 : on compare le gradient analytique à
  des différences finies centrées (ε = 1e-6), pour l'entrée et pour chaque
  élément de chaque paramètre. Sont couverts la couche linéaire, l'embedding
  (avec des indices répétés), la LayerNorm, la GELU, le softmax avec entropie
  croisée, l'attention causale, le MLP et un bloc complet. Le seuil est fixé à
  1e-7 en erreur relative ; les erreurs mesurées vont de 3e-10 à 3e-9.
- Gradient du modèle complet sur un GPT minuscule (2 couches, environ 2 000
  paramètres, tous vérifiés), avec et sans poids liés : erreur maximale 1e-8
  pour un seuil de 1e-6.
- Ces tests attrapent de vraies fautes. Des erreurs introduites exprès dans les
  backward (un facteur 1,0001 sur le gradient des scores d'attention, un
  facteur 0,99 dans la LayerNorm, une transposée oubliée) font chacune échouer
  4 ou 5 tests.
- Sur-apprentissage : sur une phrase de 29 caractères, 150 pas d'AdamW font
  passer la loss de plus de 2 à moins de 0,05, et le décodage glouton restitue
  la phrase.
- Cache KV : les logits calculés avec le cache (une amorce de 5 positions puis
  une position à la fois) sont égaux à 1e-10 près à ceux de la passe complète,
  et la génération avec et sans cache produit les mêmes caractères.
- Le reste : causalité de l'attention, conservation du float32, planning du
  taux d'apprentissage, un pas d'AdamW vérifié à la main, écrêtage, nettoyage
  des fichiers Gutenberg, découpage de l'évaluation (chaque caractère compté
  une fois), baselines.

## Corpus

Les cinq tomes des *Misérables* (1862) sur Project Gutenberg, livres
[17489](https://www.gutenberg.org/ebooks/17489),
[17493](https://www.gutenberg.org/ebooks/17493),
[17494](https://www.gutenberg.org/ebooks/17494),
[17518](https://www.gutenberg.org/ebooks/17518) et
[17519](https://www.gutenberg.org/ebooks/17519). `python -m nanogpt download`
les récupère (3,4 Mo bruts, non versionnés), retire l'en-tête et la licence
Gutenberg, puis recolle chaque paragraphe sur une seule ligne : les fichiers
d'origine sont coupés à 70 colonnes, et le modèle n'a pas à apprendre ces
retours à la ligne arbitraires.

Après nettoyage, le texte compte 3 089 285 caractères, dont 106 distincts. Le
début (90 %, soit 2 780 357 caractères) sert à l'entraînement et la fin
(10 %, 308 928 caractères, la fin du tome V) à la validation. Un découpage
contigu évite de retrouver des passages voisins des deux côtés. Le
vocabulaire est construit sur tout le texte pour que la validation n'ait
aucun caractère inconnu, mais les poids du modèle comme les comptes des
baselines ne viennent que de la partie entraînement.

## Lancer

```bash
git clone https://github.com/pinpin-http/nanogpt-numpy
cd nanogpt-numpy
make install   # uv venv --python 3.12, puis dépendances épinglées
make test      # quelques secondes, sans téléchargement
make sample    # génère avec les poids fournis dans models/
make data      # télécharge et nettoie le corpus dans data/
make train     # 25 min de budget, checkpoint dans checkpoints/
make report    # évaluation complète, écrit docs/results.json, loss.png, samples.json
```

Sans `make`, les mêmes commandes s'écrivent `python -m nanogpt <commande>` ;
`python -m nanogpt train --help` liste les options (taille du modèle, taux
d'apprentissage, `--time-budget-min 0 --max-iters 3000` pour un entraînement
piloté par le nombre d'itérations, etc.). Le Makefile limite NumPy à deux
threads ; `make train THREADS=8` lève la limite.

```bash
python -m nanogpt sample --prompt "Cosette" --length 300 --temperature 0.7 --top-k 10
```

## Limites

- Le run publié est court (796 pas) à cause de la charge de la machine. Le
  modèle est sous-entraîné, et c'est un seul run avec une seule graine.
- NumPy exécute les opérations élément par élément (GELU, softmax, LayerNorm,
  AdamW) sur un seul cœur ; seuls les produits matriciels profitent des
  threads d'OpenBLAS. Au profilage, les couches linéaires représentent un
  tiers environ du temps d'un pas.
- Avec un budget de temps, le nombre de pas dépend de la machine : à graine
  égale, deux runs ne s'arrêtent pas au même endroit. Avec
  `--time-budget-min 0`, l'entraînement redevient reproductible au pas près.
- Pas de dropout. Sur une seule passe, train et val restent au même niveau,
  mais un entraînement beaucoup plus long en aurait sans doute besoin.
- Contexte de 128 caractères et positions absolues : au-delà, la génération
  reconstruit le cache sur les 64 derniers caractères et oublie le reste.
- La validation ne couvre que la fin du tome V, avec un seul découpage.

## Licence

Code sous licence MIT. Le texte des *Misérables* est dans le domaine public ;
il est téléchargé depuis Project Gutenberg et n'est pas inclus dans le dépôt.
