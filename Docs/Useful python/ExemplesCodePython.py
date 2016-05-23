# Exemple 1
import numpy as np
import numpy.random as npr

n = 1000
# tirage aleatoire de n valeurs suivant une loi normale, centree reduite
X = npr.randn(n)
#
# calcul du sinus de chaque valeur de X, du booleen, 1 pour vrai, 0 pour faux,
# indiquant si le sinus est ou non inferieur a 0.5 et de la moyenne des booleens
# Noter l'absence de boucle for
p = np.mean(np.sin(X) < 0.5)
print "Sur",n," tirages, la proba que sin(X) < 0.5 quand X est de loi N(0,1)=", p

# Exemple 2
import numpy as np #import de numpy sous l'alias np
import numpy.random as npr #import de numpy.random sous l'alias npr
import matplotlib.pyplot as plt #import de matplotlib.pyplot sous l'alias plt
import scipy.stats as sps #import de scipy.stats sous l'alias sps

# Nombre de simulations
n = 1000
# Dimension
d = 2
#moyenne et covariance
m = np.zeros(d) # vecteur de taille 2 contenant des 0
cov = np.array([[2., 1.], [1., 2.]]) # matrice de dimension 2x2
#Tirage de n realisations du vecteur gaussien de dimension 2
X = npr.multivariate_normal(m, cov, size=n)
#Trace du nuage de points X
# Initialisation d'une figure
plt.figure()
# Creation du nuage de points dont les ccordonnees sont donnees par X,
# definition du label associe qui sera utilise par la fonction matplotlib.pyplot.legend
scat = plt.scatter(X[:, 0], X[:, 1], label="1000 simulations d'un vecteur gaussien en dimension 2")
# Creation de la legende
plt.legend()
#
# Calcul de l'ecart-type de y=2 * X_1 - 3 * X_2
# on a note X_1 pour designer la premiere composante de X
# et X_2 pour designer la seconde composante de X
y = 2 * X[:, 0] - 3 * X[:, 1]
#
#Histogramme de la distribution empirique et densite de 2 * X_1 - 3 * X_2
# Initialisation d'une figure
plt.figure()
# Creation de l'histogramme de y, 
# definition du label associe qui sera utilise par la fonction matplotlib.pyplot.legend
plt.hist(y, normed=True, label="Histogramme de 2 * X_1 - 3 * X_2", bins=15)
#
# Calcul et affichage de la discretisation de la densite theorique
x = np.array([2., -3.])
s = np.sqrt(np.dot(np.dot(x.T, cov), x))
# Creation de la courbe representant la densite theorique en fonction de x, 
x = np.linspace(min(y), max(y), 100)
f_x = sps.norm.pdf(x, loc=0., scale=s)
# tracer de la courbe et definition du label associe qui 
# sera utilise par la fonction matplotlib.pyplot.legend
plt.plot(x, f_x, "r", label="Densite theorique de 2 * X_1 - 3 * X_2")
# Creation de la legende
plt.legend()
#
# Affichage des 2 figures creees, grace a la fonction matplotlib.pyplot.show
plt.show()

# Exemple 3
import numpy as np #import de numpy sous l'alias np
# On declare 4 matrices que l'on assemble pour en construire une 5eme
A = np.array([[2, 1, 1], [4, 3, 0]])
B = np.array([[1, 2], [12, 0]])
C = np.array([[1, 2], [12, 0], [-1, 2]])
D = np.array([[1, 2, -4], [2, 0, 0], [1, 2, 3]])
print np.bmat([[A, B], [C, D]])
# affiche la matrice
# [[ 2  1  1  1  2]
#  [ 4  3  0 12  0]
#  [ 1  2  1  2 -4]
#  [12  0  2  0  0]
#  [-1  2  1  2  3]]X = [10., [1, 2, 3], "Bonjour", 4]

# Exemple 4
# Boucle for
for x in X: # boucle sur les elements de X
    print x # affiche chaque element de X
# affichera successivement 10 puis [1, 2, 3] puis Bonjour puis 4  
X = [10., [1, 2, 3], "Bonjour", 4]
# boucle sur les elements de X, 
# recuperation de l'indice de chaque element dans i
for i, x in enumerate(X):
  print "X(",i,") = ", x # affiche l'indice i et l'element x de X
# affichera :
# X( 0 ) =  10.0
# X( 1 ) =  [1, 2, 3]
# X( 2 ) =  Bonjour
# X( 3 ) =  4

# Exemple 5
#Declaration de la fonction f de parametre d'appel tata, toto et tintin
#tata n'a pas de valeur par defaut
#toto = 2 par defaut, on pourra appeler f sans specifier la valeur de toto
#tintin = 4 par defaut, on pourra appeler f sans specifier la valeur de tintin 
def f(tata, toto=2, tintin=4):
    #Les 6 lignes suivantes permettent de decrire votre fonction, elles seront
    # reprises par le module d'aide. Noter le \n qui permet d'introduire des
    # retours a la ligne lors de l'affichage de l'aide. Ces lignes commencent 
    # et se terminent par '''
    '''
    f a pour parametres d'appel : \n
    tata qui n'a pas de valeur par defaut \n
    toto = 2 par defaut \n
    tintin = 4 par defaut \n
    '''
    # corps de la fonction f, indente d'une tabulation
    print tata, toto, tintin
# fin de declaration de f
f(3)  # affiche 3 2 4
f(3, 5, 12)  # affiche 3 5 12
f(3, tintin=5, toto=12)  # affiche 3 12 5

# Exemple 6
import numpy as np
import numpy.random as npr
import scipy.stats as sps

mu = 0.5
n = 1000
m = 20
X = npr.poisson(mu, n)  # Simulation de n realisations de  variables de loi de Poisson d'intensite mu
x = np.arange(m+1) # x contient le vecteur d'entiers [0, 1, 2, ..., m]
f = sps.poisson.pmf(x, mu) # Densite de probabilite de la loi de Poisson d'intensite mu aux points x

# Exemple 7
# On demande interactivement d'entrer un entier
x =(input("Donne moi un entier: "))
if x < 0:          # est-ce que x est negatif ?
    x = 0          # si oui on le remplace par zero
    print "Je remplace ton nombre negatif par zero"
elif x == 0:       # est-ce que x est egal a zero ?
    print 'Zero'   # si oui on affiche "Zero"
elif x == 1:       # est-ce que x est egal a 1 ?
    print 'Un'     # si oui on affiche "Un"
else:              # sinon
    print 'Plus'   # on affiche "Plus"# Methode 1 : import de la fonction randn

# Exemple 8
# Methode 1 : import de la fonction randn
from numpy.random import randn
X = randn(0, 1)

# Methode 2 : import des librairies de numpy 
import numpy
X = numpy.random.randn(0, 1)

# Methode 3 : import de la librairie numpy.random et création d'un alias de numpy.random que l'on appelle npr
import numpy.random as npr
X = npr.randn(0, 1)

# Methode 4 : import toutes les fonctions de la librairie numpy.random
from numpy.random import *
X = randn(0, 1)

# Methode 5 : import de toutes les fonctions des librairies de pylab, dont celles de numpy, scipy, matplotlib
from pylab import *
X = randn(0, 1)

# Import de plusieurs fonctions d'une meme librairie
from numpy import ones, zeros, empty

# Exemple 9
# Delaration de la fonction fib
def fib(n):
    # Tout le contenu descriptif de la fonction est indente
    #
    #Les 4 lignes suivantes permettent de decrire votre fonction, elles seront
    # reprises par le module d'aide. Noter le \n qui permet d'introduire des
    # retours a la ligne lors de l'affichage de l'aide. Ces lignes commencent 
    # et se terminent par '''
    """
    Fonction qui calcule le n-ieme element de la suite de Fibonacci\n
    parametre d'entree n integer
    """
    # initialisation des parametres   
    a, b = 0, 1
    # debut de la boucle d'indice i, variant de 0 a n-1
    for i in range(n):
        # les lignes decrivant les operations a repeter sont indentees
        a, b = b, a + b
    # fin de boucle, marquee par les lignes suivantes indentees
    # d'une tabulation de moins
    return a
# fin de declaration de la fonction fib, les lignes suivantes
# ne sont plus indentees
print fib(10) # donne 55

# Exemple 10
x = [4, 2, 10, 9, "toto"]
y = x  # y est juste un alias du tableau x, pas de copie effectuee ici
y[2] = "tintin" # modifie x(2) =!) en x(2)="tintin"
print x # Cela affiche  [4, 2, "tintin", 9, "toto"]
x = [4, 2, 10, 9, "toto"]
y = x[:]   # On demande une copie
y[2] = "tintin" # modifie y mais pas x
print x # Cela affiche  [4, 2, 10, 9, "toto"]
print y # Cela affiche  [4, 2, "tintin", 9, "toto"]import numpy as np #import de numpy sous l'alias np

# Exemple 11
import numpy as np
x = np.array([4, 2, 1, 5, 1, 10]) # declaration du tableau x
# construction du tableau de booleens y a partir de tests sur les valeurs de x
y=np.logical_and(x >= 3, x <= 9, x != 1)
print y
# affiche  [ True False False True False False]import numpy as np #import de numpy sous l'alias np

# Exemple 12
n = 4
x = np.arange(n) # x contient le tableau d'entiers  [0, 1, 2, 3]
# On va diviser par un entier
diviseur = 3
y = x / diviseur
print y," x tableau d'entiers, divise par 3 en tant qu'entier"
# y contient [0, 0, 0, 1], car le diviseur etant entier, la division est entiere
# On va diviser par un reel en double precision
diviseur = np.float(3.)
y = x / diviseur # y contient [ 0.          0.33333333  0.66666667  1.        ]
print y," x tableau d'entiers, divise par 3 en tant que reel"
# On va diviser un tableau de reels par un entier
diviseur = 3
x = np.array(x, dtype=np.float)
y = x / diviseur # y contient [ 0.          0.33333333  0.66666667  1.        ]
print y," x tableau de reels divise par 3 en tant qu'entier"

# Exemple 13
#Declaration de la fonction f
def f(x):
    # corps de la fonction f, indente d'une tabulation
    # Une super fonction que je coderai plus tard, 
    # ou que quelqu'un de l'equipe codera
     pass
# fin de declaration de f
# debut de boucle d'indice i variant de 0 a 9
for i in range(10):
    # corps de la boucle indente d'une tabulation
    # Une boucle tres compliquee que je coderai plus tard
    # ou que quelqu'un de l'equipe codera
    pass
# fin de boucle

# Exemple 14
#Initialisation de l'indice i de la boucle while
i = 1
# debut de la boucle while d'indice i, tant que i est inferieur a 100
while i < 100:
    # corps de la boucle indente d'une tabulation
    print i # on affiche i, tant qu'il est inferieur a 100
    i *= 3 # on multiplie i par 3, tant qu'il est inferieur a 100
# fin de boucle
# affichera 1 3 9 27 81
