# Guide d'Implémentation E3 : Reconnaissance de Gravures Optiques

## 1. Développement du modèle siamois pour la comparaison croquis/gravure

Un réseau de neurones siamois est choisi pour évaluer la similarité entre un dessin de gravure et les images de gravures de référence. Ce modèle comprend deux sous-réseaux identiques (mêmes couches et mêmes poids) qui produisent chacun un vecteur d'embedding à partir d'une image en entrée.

En entraînant le modèle avec des paires d'images similaires et différentes, on apprend à rapprocher les embeddings des images de la même classe tout en éloignant ceux d'images de classes différentes. Dans notre cas, cela permettra de générer des vecteurs caractéristiques proches pour un croquis dessiné à la main et la gravure optique correspondante, et des vecteurs distants pour des gravures différentes.

Exemple d'architecture de réseau siamois : deux réseaux jumeaux à convolution partagent leurs poids pour produire des vecteurs de caractéristiques comparables. La distance Euclidienne entre ces vecteurs sert à évaluer la similarité des entrées.

Chaque sous-réseau peut être un CNN léger (quelques couches convolutionnelles + pooling) adapté aux petites images de gravures. On peut aussi utiliser un modèle pré-entraîné (par ex. ResNet) comme base, en le fine-tunant sur nos gravures, afin de bénéficier de caractéristiques visuelles robustes. 

En sortie, une couche d'embedding (par ex. 128 dimensions) fournit la représentation vectorielle de la gravure. La comparaison entre deux gravures se fait via une métrique de distance (Euclidienne ou cosinus) appliquée aux embeddings des deux images. 

Lors de l'entraînement, une couche de contraste calcule la perte en fonction de cette distance et du label « même gravure » ou « gravures différentes ». On pourra utiliser une loss contrastive ou triplet pour optimiser le modèle : celle-ci vise à minimiser la distance entre embeddings d'une même gravure et à maximiser la distance entre embeddings de gravures différentes.

## 2. Préparation des données et entraînement du modèle

Les données d'entraînement proviennent des images de gravures optiques récupérées sur un site spécialisé. Ce site recense les gravures de verres progressifs de nombreux fabricants (Essilor, Zeiss, Hoya, etc.), ce qui nous a permis de constituer un dataset varié de plusieurs centaines de symboles. Chaque image correspond à la gravure (marquage) d'un modèle de verre spécifique, typiquement une petite gravure de 1–3 caractères ou symboles.

### Pré-traitement des images

Les gravures sont converties en niveaux de gris (ou en binaire noir/blanc si nécessaire) et redimensionnées à une taille uniforme (par ex. 64×64 px) pour entrer dans le modèle. On veillera à normaliser l'intensité (éventuellement inverser noir/blanc selon le fond) afin que les croquis dessinés (fond blanc, trait noir) aient une distribution similaire aux images de référence. 

Les images de croquis fournies par l'utilisateur seront soumises au même pré-traitement pour assurer la cohérence avec les données d'entraînement.

### Constitution des paires pour l'entraînement

Comme on a généralement peu d'images par classe (parfois une seule image par gravure unique), on adopte une approche d'apprentissage par comparaison. On génère des paires positives en prenant deux images de la même gravure. Si seulement une image existe, on peut créer une paire positive en appliquant une légère augmentation sur l'image (rotation, bruit, épaisseur du trait) et en la comparant à l'originale – elles représentent la même gravure. 

Inversement, on génère des paires négatives en associant des images de gravures différentes. Ce procédé permet de créer un ensemble d'entraînement pour le modèle siamois sans nécessiter de nombreuses images par classe.

### Entraînement du réseau

On alimente le modèle siamois avec ces paires d'images. Pour chaque paire, le modèle calcule les embeddings des deux images via les sous-réseaux jumeaux, puis la distance entre ces embeddings. La loss (contrastive, par exemple) est alors calculée : faible si paire positive (on pousse la distance à 0) et forte si paire négative (on pousse la distance à une valeur marge).

On utilise un optimiseur (Adam) pour ajuster les poids du CNN. L'entraînement peut se faire sur quelques époques en surveillant la diminution de la perte et en validant sur un petit ensemble de paires de validation. À l'issue de l'entraînement, on obtient un modèle d'embedding capable de représenter chaque gravure par un vecteur discriminant dans l'espace des caractéristiques.

## 3. Sauvegarde des embeddings pour accélérer la recherche

Une fois le modèle entraîné, on l'utilise pour précalculer l'embedding de chaque gravure de référence. L'idée est de transformer l'ensemble de notre base de gravures en vecteurs numériques et de stocker ces vecteurs, de façon à éviter de recalculer le CNN pour chaque comparaison lors d'une requête. 

Pour chaque image de gravure dans la base, on calcule son vecteur d'embedding en passant l'image dans le sous-réseau CNN (un seul des jumeaux, utilisé comme extracteur de features). On obtient ainsi une liste (ou dictionnaire) de vecteurs, chacun associé à l'identifiant ou au nom de la gravure correspondante.

Ces vecteurs d'embedding sont ensuite sauvegardés dans un fichier pickle (sérialisation Python). Par exemple, on peut stocker un objet Python (liste de tuples {nom: embedding}) dans `embeddings.pkl`. Au démarrage du service de reconnaissance, il suffira de charger ce fichier en mémoire.

La comparaison lors d'une recherche se résume alors à :
1. Calculer l'embedding du croquis envoyé par l'utilisateur (en le passant dans le CNN)
2. Parcourir la liste des embeddings sauvegardés pour trouver celui dont la distance à l'embedding du croquis est la plus faible

On peut utiliser la distance Euclidienne ou la distance cosinus (si on normalise les vecteurs sur la sphère unité). Le plus proche voisin en termes de distance vectorielle correspond à la gravure la plus similaire, et sera renvoyé en résultat de la reconnaissance. Cette approche évite de passer en revue chaque image via le CNN et accélère significativement la réponse.

Enfin, si le dataset est très grand, on pourra optimiser davantage la recherche (par exemple via un index approximate nearest neighbors). Mais compte tenu du volume gérable (quelques centaines de gravures), une recherche linéaire sur les embeddings en mémoire est suffisante et quasi instantanée.

## 4. Création de l'API REST (FastAPI) pour la reconnaissance

L'application backend sera exposée via FastAPI, permettant d'interagir avec le modèle par des requêtes HTTP. On définit plusieurs endpoints REST et on sécurise l'accès par une authentification OAuth2 + JWT.

### Endpoint de reconnaissance

**POST /recognize**

Cet endpoint reçoit en entrée un dessin de gravure (par exemple sous forme de fichier image uploadé). Il applique le pipeline de reconnaissance :
- Vérification de l'authentification du client
- Pré-traitement de l'image (redimensionnement, binarisation, etc.)
- Calcul de l'embedding du croquis via le modèle CNN chargé en mémoire
- Comparaison de cet embedding à la base de vecteurs préenregistrés (fichier pickle) pour trouver la gravure la plus proche

En sortie, l'API renvoie les informations sur la gravure reconnue – par exemple un identifiant ou le nom du modèle de verre correspondant, éventuellement accompagné d'un score de similarité. Le tout est encapsulé en JSON dans la réponse HTTP.

### Endpoint de liste des gravures

**GET /gravures**

Cet endpoint renvoie la liste de toutes les gravures référencées dans notre base (celles utilisées pour la correspondance). Il permet par exemple de récupérer le catalogue des gravures connues du système.

La réponse pourrait être un JSON contenant pour chaque gravure un identifiant, un nom (par ex. code de gravure, marque, indice) et éventuellement un lien vers l'image de la gravure. Cet endpoint interroge soit une base de données, soit directement une structure en mémoire construite au lancement à partir des fichiers scrapés. Exemple de réponse:

```json
[ 
  {"id": 214, "code": "Varilux", "indice": 1.67}, 
  {"id": 340, "code": "Essilor", "indice": 1.6}, 
  ...
]
```

Cela permet à un client (ou à l'interface utilisateur) de connaître les gravures disponibles et d'afficher éventuellement les références.

### Authentification OAuth2/JWT

Les endpoints de l'API sont protégés par une authentification basée sur OAuth2 (Password flow) et des JSON Web Tokens en Bearer. Concrètement :

1. Un utilisateur (ou l'UI) devra d'abord appeler l'endpoint de login (POST /token) en fournissant ses identifiants (nom d'utilisateur/mot de passe)
2. Si les identifiants sont valides, le service génère un token JWT signé contenant l'identifiant de l'utilisateur (dans le champ sub) et une date d'expiration, puis le retourne au client
3. Ce token n'est pas chiffré (il peut être décodé par n'importe qui) mais il est signé de sorte que le serveur peut en vérifier l'authenticité et l'intégrité
4. Le client devra inclure ce jeton dans les requêtes ultérieures (dans l'en-tête HTTP `Authorization: Bearer <token>`)
5. Le backend utilisera OAuth2PasswordBearer de FastAPI pour extraire et vérifier automatiquement le token sur les endpoints protégés
6. Si le token est manquant ou invalide/expiré, FastAPI renverra une erreur 401 Unauthorized. Autrement, la requête est autorisée et le code du endpoint peut s'exécuter.

### Implémentation FastAPI

On organise le code en un module (par ex. app.py) qui crée une instance FastAPI. Au démarrage, on charge en mémoire le modèle CNN entraîné (poids depuis un fichier .pt ou .h5) ainsi que le dictionnaire des embeddings depuis le pickle. Cela peut se faire dans un handler de l'événement startup de FastAPI ou au niveau du module global.

Pour le endpoint `/recognize`, on utilisera un paramètre de type UploadFile (FastAPI gère l'upload de fichier image) ou un champ bytes dans une requête POST. Le code du endpoint lira le fichier image, le convertira en tableau numpy/PIL, puis appellera la fonction de prédiction (embedding + recherche du plus proche). Enfin, il renverra la réponse JSON avec le résultat.

Pour le endpoint `/gravures`, on peut simplement renvoyer la structure chargée en mémoire (par exemple, une liste d'objets Pydantic décrivant chaque gravure).

En ce qui concerne l'authentification, on définira un endpoint `/token` qui valide les credentials (pour l'exercice, on peut stocker un utilisateur factice en dur, avec mot de passe hashé avec Passlib/Bcrypt). En cas de succès, on crée un JWT (via PyJWT) avec une expiration (par ex. 30 minutes) et on le retourne. FastAPI fournit des utilitaires pour intégrer ce mécanisme : on utilisera OAuth2PasswordBearer pour extraire le token des requêtes entrantes et une dépendance `get_current_user` qui décode le JWT (en utilisant la même clé secrète) et récupère l'utilisateur. On applique `Depends(get_current_user)` sur les endpoints `/recognize` et `/gravures` pour les protéger. Ainsi, un appel sans token ou avec token invalide retournera automatiquement une erreur 401.

## 5. Intégration de Gradio pour l'interface utilisateur

Pour rendre le système plus interactif, on intègre une interface web utilisateur grâce à Gradio. Gradio permet de créer rapidement une interface de dessin et d'affichage des résultats, sans développement front-end lourd. L'idée est d'offrir à l'utilisateur un canvas de dessin où il peut reproduire la gravure qu'il observe, puis de soumettre ce dessin au modèle pour obtenir le nom de la gravure correspondante.

### Interface de dessin

On utilisera le composant Sketchpad ou ImageEditor de Gradio, qui offre une zone de dessin libre (par défaut 256×256 px en niveaux de gris). L'utilisateur pourra tracer à la souris (ou tactile) la forme de la gravure. En validant, l'image du dessin (sous forme numpy array) est fournie à une fonction Python de traitement.

### Fonction de prédiction

Cette fonction (qu'on peut nommer `predict_gravure(drawing)`) va reprendre la logique du endpoint `/recognize`. Pour éviter les doublons, on peut refactorer le code de reconnaissance dans une fonction utilitaire commune que la fonction Gradio et le endpoint FastAPI appelleront. La fonction effectue le pré-traitement du dessin (tel que défini plus haut), calcule l'embedding via le modèle, trouve la correspondance la plus proche dans la liste des embeddings de référence, et retourne le résultat.

### Retour affiché à l'utilisateur

On peut configurer Gradio pour afficher le résultat de plusieurs façons. Par exemple, on peut retourner le nom de la gravure identifié (texte) et/ou afficher l'image de la gravure de référence correspondante. Gradio supporte les sorties multiples, donc on peut avoir un output texte et un output image. Ainsi, après avoir dessiné sa forme, l'utilisateur verrait s'afficher (par exemple) « Gravure reconnue : Varilux 1.67 » et l'image de la gravure Varilux correspondante.

### Montage avec FastAPI

Pour faciliter le déploiement, on intègre l'interface Gradio au sein même de l'application FastAPI. FastAPI permet de monter une application WSGI/ASGI tiers sur une sous-route. Gradio fournit soit un serveur autonome, soit une interface montable.

En créant l'interface (par ex. `iface = gr.Interface(fn=predict_gravure, inputs=gr.Sketchpad(...), outputs=[...])`), on peut la monter sur FastAPI avec `app.mount("/gradio", iface)`. De cette façon, le même serveur web servira :
- l'API REST (sur les endpoints JSON décrits plus haut),
- l'UI Gradio accessible sur `/gradio` (contenant le canvas de dessin et l'affichage du résultat).

L'intégration via `app.mount` garantit un déploiement unifié de l'API et de l'interface web. L'utilisateur final peut ainsi accéder via un navigateur à l'URL du service (par exemple `http://<serveur>/gradio`) pour utiliser l'outil de reconnaissance visuellement, tandis que des clients programmatiques peuvent appeler les endpoints JSON de manière classique. Cette double interface couvre les besoins d'expérimentation utilisateur et d'accès API.

À noter : l'interface Gradio étant encapsulée dans FastAPI, les appels depuis ce front-end bypassent l'authentification JWT (on peut considérer que l'UI est une partie du service lui-même). Si on souhaitait sécuriser également l'UI, il faudrait intégrer un flux d'authentification côté front (ce qui dépasse le cadre de l'exercice). Ici, on pourra autoriser librement l'accès à `/gradio` pour des raisons de simplicité.

## 6. Tests unitaires et d'intégration (Pytest)

Pour garantir le bon fonctionnement de l'ensemble, on met en place des tests avec pytest couvrant le modèle et l'API. On organise un répertoire `tests/` contenant par exemple des tests unitaires du modèle (prétraitement d'image, fonction de calcul d'embedding, etc.) et des tests d'intégration de l'API.

### Tests unitaires du modèle

On peut tester la fonction de comparaison d'embeddings sur des cas simples. Par exemple, vérifier que la distance calculée par notre fonction est proche de 0 pour deux vecteurs identiques, ou plus grande pour des vecteurs différents. Si on a isolé la logique de pré-traitement, on peut tester qu'un dessin d'entrée est bien transformé (taille, format) comme attendu.

Il est également possible de tester l'inférence du modèle sur un échantillon : par exemple, entraîner rapidement le modèle sur 2–3 classes factices et vérifier qu'une image de classe A est plus proche de A que de B. Ces tests permettent de valider les composants de base sans lancer tout le serveur.

### Tests d'intégration API

On utilise TestClient de FastAPI pour simuler des appels HTTP sur l'API sans avoir besoin de la déployer réellement. On crée un client sur notre app (`client = TestClient(app)`) puis on peut appeler `client.post("/token", data={...})` pour obtenir un token JWT (en utilisant un utilisateur de test).

Ensuite, on teste l'endpoint `/recognize`: on peut préparer une image de gravure (par ex. un fichier PNG d'une gravure connue dans la base), et l'envoyer via `client.post("/recognize", files={"file": img_file}, headers={"Authorization": "Bearer <token>"})`. On vérifie que la réponse est un JSON avec le bon identifiant de gravure et un code 200 OK. On peut effectuer un test similaire sans token pour vérifier qu'on obtient bien un 401 Unauthorized.

De même, on peut tester le GET `/gravures` en authentifié et vérifier que la liste retournée contient le nombre attendu d'éléments et quelques valeurs connues.

On écrira plusieurs fonctions de test (test_...) pour couvrir ces cas. Pytest permettra de les exécuter facilement et de repérer les régressions. Par exemple :

- `test_recognize_success()` : vérifie qu'une requête de reconnaissance avec une image valide et un token retourne un résultat correct (éventuellement, on peut comparer que le nom renvoyé correspond à la gravure envoyée).
- `test_recognize_unauthorized()` : vérifie qu'un appel sans token est refusé.
- `test_get_gravures_list()` : vérifie que la liste des gravures n'est pas vide et que son format correspond à ce qui est attendu (par ex. contient une entrée avec champs id et code).
- `test_jwt_expiry()` (optionnel) : on pourrait simuler l'expiration d'un token en modifiant son exp dans le futur et voir si l'API le refuse.

Grâce à FastAPI et Starlette, on peut tester l'application de façon synchrone facilement, en utilisant simplement assert sur les codes de statut et le contenu JSON.

## 7. Reconfiguration de la CI/CD pour inclure tests et déploiement

Le pipeline d'Intégration Continue / Déploiement Continu existant doit être mis à jour pour intégrer les nouvelles fonctionnalités d'E3. Les étapes suivantes seront ajoutées ou modifiées dans la configuration CI (par ex. fichier .gitlab-ci.yml ou workflows GitHub Actions) :

### Installation des dépendances

S'assurer que l'environnement de CI installe les dépendances nécessaires (énoncées dans requirements.txt). Cela inclut des packages comme FastAPI, PyTorch/TensorFlow (selon le framework choisi pour le modèle), Gradio, PyJWT, etc. Ces librairies étant potentiellement lourdes, on peut utiliser un cache de dépendances pour accélérer les runs CI successifs.

### Exécution des tests

Ajouter une étape pour lancer Pytest. Par exemple, un job tests qui exécute pytest dans le répertoire du projet. On configurera éventuellement les variables d'environnement nécessaires (par ex. désactiver le chargement du modèle lourd si on veut simuler certaines parties). Le job doit produire un rapport succinct : tous les tests doivent passer, sinon le pipeline échoue. On peut aussi intégrer un rapport de couverture de code (coverage) pour évaluer la portée des tests.

### Build de l'image Docker

Une fois les tests validés, le pipeline peut enchaîner sur la construction de l'image Docker de l'application (voir section suivante). On utilisera la commande docker build sur le Dockerfile d'E3, et on taggera l'image (par ex. myapp/e3:latest ou un tag de version/commit). Cette étape garantit que l'image se construit correctement à chaque modification du code. On peut même envisager d'exécuter l'image en mode test dans la CI (démarrer un conteneur et ping un endpoint) pour vérifier que le container fonctionne, mais ce n'est pas obligatoire si les tests applicatifs couvrent déjà beaucoup.

### Déploiement continu

Si le pipeline CI/CD inclut le déploiement, on mettra à jour cette étape pour déployer la nouvelle image. Par exemple, pousser l'image sur un registre Docker (Docker Hub, ECR, GitLab Registry…) avec les bonnes balises. Ensuite, si une infrastructure est en place (Kubernetes, VM Docker, etc.), déclencher le déploiement de cette image.

Dans un contexte d'exercice, on peut simplifier en disant que l'image sera disponible pour un déploiement manuel ou automatique. L'important est de intégrer le déploiement du nouveau composant E3 : par exemple, si précédemment seules E1 et E2 étaient déployées, on ajoute le déploiement du service E3 (conteneur additionnel ou mise à jour du conteneur existant pour inclure E3).

### Mises à jour du pipeline existant

On documentera et modifiera le fichier de config CI pour inclure ces nouvelles étapes, en veillant à l'ordre (build/test avant déploiement) et aux conditions (par ex. déclencher le déploiement uniquement sur la branche principale ou sur un tag de version).

En somme, la CI/CD doit maintenant vérifier la qualité (tests) et déployer l'application complète (API + modèle + interface) automatiquement. Cela garantit qu'à chaque modification du code E3, les tests s'exécutent et l'application à jour est packagée puis prête à être mise en production.

## 8. Déploiement Docker de l'ensemble (Dockerfile E3)

L'application complète (incluant API FastAPI, modèle et interface Gradio) sera conteneurisée via un Dockerfile dédié dans le dossier E3. L'objectif est de pouvoir lancer tout le service de reconnaissance de gravures dans un conteneur unique, pour faciliter le déploiement sur tout environnement (serveur de production, cloud, etc.).

### Contenu du Dockerfile

On part d'une image de base Python 3 (par ex. python:3.10-slim). On copie les fichiers du projet E3 dans l'image et on installe les dépendances requises. Par exemple:

```dockerfile
FROM python:3.10-slim

# Installation des dépendances system (si besoin, ex: libGL pour opencv)

# Définir le répertoire de travail
WORKDIR /code

# Copier les dépendances Python et installer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le code de l'application
COPY . .

# Lancer l'application FastAPI/Gradio via Uvicorn
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]
```

Ce fichier instruit Docker pour construire l'image pas à pas : il installe Python, puis nos packages (FastAPI, gradio, modèle ML…), puis ajoute le code. La commande de démarrage utilise Uvicorn pour lancer l'application FastAPI (`app:app` fait référence à l'instance app dans le fichier app.py). On expose le port 80 du conteneur pour que l'API et l'UI soient accessibles de l'extérieur.

On veillera à inclure le modèle entraîné et le fichier d'embeddings dans l'image. Par exemple, si le modèle est sauvegardé dans model.pth et les embeddings dans embeddings.pkl, on s'assurera que le Dockerfile copie ces fichiers (via COPY) dans le conteneur. Le code FastAPI pourra alors les charger au démarrage sans dépendre de l'extérieur. Alternativement, on peut prévoir un volume pour monter ces fichiers, mais pour simplifier le déploiement on les embarque dans l'image.

Après avoir écrit le Dockerfile, on pourra tester la construction locale avec : `docker build -t gravure-app:e3 .` depuis le répertoire E3. Une fois l'image créée, lancer `docker run -p 80:80 gravure-app:e3` exécute le container et expose le service sur le port 80 de l'hôte. En accédant à http://localhost:80/gradio, on devrait voir l'interface de dessin Gradio, et les endpoints API (/recognize, /gravures, /docs pour la doc Swagger auto-générée) seront disponibles sur le même port.

Cette conteneurisation assure que l'application est portable et encapsule toutes ses dépendances. Le Dockerfile E3 sera intégré au pipeline CI/CD : après tests, l'image sera construite puis déployée. Ainsi, la fonctionnalité complète de reconnaissance de gravures (modèle + API + interface) pourra être déployée en un seul geste sur l'infrastructure cible, finalisant l'implémentation de E3.