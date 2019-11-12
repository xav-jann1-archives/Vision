# TP 2

## Question 1

Pour detecter un visage dans une image, ce code utilise le principe des cascades de Harr. 
Pour cela, une image est chargé et est convertie en niveaux de gris. 
L'algorithme utilise un fichier .xml qui contient les caractéristiques permettant d'identifier un visage.
L'image est traitée par l'algorithme en fonction de paramètres qui spécifie le fonctionnement des cascades.
Une liste de valeur correspondant à la position des visages détectés dans l'image est retournée.
Il est ainsi possible de tracer un rectange autour de chacun ces visages.

Le principe des cascades de Harr fonctionne en identifiant des caractéristiques contenues dans l'image formant l'objet que l'on souhaite détecter

Détails des différents paramètres:
- scaleFactor:
- minNeighbors:
- minSize:
- flags:



Choix de la caméra:
  Modifier la valeur de cam dans `cv2.VideoCapture(cam)` pour choisir la caméra.

éclairage faible = mauvaise détection
distance = moins bonne détection
detaction fonctionne seulement lorsque la tête est dans un position naturel (en face de la caméra sans rotation)

scale factor est inversement proportionnel au temps de traitement de l'image et de l'efficacité de la reconnaissance.
quand min neighbors est faible (2 par exemple) l'algorithme détecte des zones de l'image qui ne contiennent pas de visage
min size limite la detection en fonction de la taille du visage dans l'image.
