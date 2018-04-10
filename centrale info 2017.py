import math
import numpy as np
import random


def générer_PI(n: int, cmax: int) -> np.ndarray:
    out = []

    assert n <= (cmax + 1) ** 2

    for i in range(n):
        x, y = random.randint(0, cmax), random.randint(0, cmax)
        while (x, y) in out:
            x, y = random.randint(0, cmax), random.randint(0, cmax)
        out.append((x, y))

    return out


def position_robot() -> tuple:
    return (0, 0)


def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx ** 2 + dy ** 2)


def calculer_distances(PI: np.ndarray) -> np.ndarray:
    n = len(PI)

    out = [[0 for c in range(n + 1)] for l in range(n + 1)]

    for l in range(1, n):
        for c in range(l):
            tmp = distance(PI[l], PI[c])
            out[l][c] = tmp
            out[c][l] = tmp

    # on calcul pour la pos du robot
    pos = position_robot()
    for l in range(n):
        tmp = distance(pos, PI[l])
        out[n][l] = tmp
        out[l][n] = tmp

    # astuce mat triangulaire inf + sa transposée
    return out


# I.B – Traitement d’image

def F1(photo: np.ndarray) -> np.ndarray:
    n = photo.min()
    b = photo.max()
    h = np.zeros(b - n + 1, np.int64)
    for p in photo.flat:
        h[p - n] += 1
    return h


# compte le nombre de pixel pour chaque intensité entre le min et max des intensitées de la photo


def sélectionner_PI(photo: np.ndarray, imin: int, imax: int) -> np.ndarray:
    PI = []
    n, p = photo.shape
    for i in range(n):
        for j in range(p):
            if imin <= photo[i, j] <= imax:
                PI.append([i, j])
    return np.array(PI)


def longueur_chemin(chemin: list, d: np.ndarray) -> float:
    return sum([d[chemin[i]][chemin[i + 1]] for i in range(len(chemin) - 1)])


def normaliser_chemin(chemin: list, n: int) -> list:
    out = []
    # supprimer les éventuels doublons
    for p in chemin:
        if not (p in out) and (p < n):
            out.append(p)
    # ajoute les elts manquants
    manquants = [k for k in range(n) if not (k in out)]

    return out + manquants


# II.B – Force brute

def plus_proche(pos, d, visités):
    possible = [k for k in range(len(d)) if not (k in visités)]
    return min(possible, key=lambda x: d[x][pos])


def plus_proche_voisin(d: np.ndarray) -> list:
    nbPoints = len(d)
    pos = nbPoints - 1

    chemin = [pos]

    for i in range(nbPoints - 1):
        pos = plus_proche(pos, d, chemin)
        chemin.append(pos)

    return chemin


##genetique

def créer_individu(d):
    chemin = [i for i in range(len(d) - 1)]  # -1 à cause de la position du robot
    random.shuffle(chemin)
    chemin = [len(d) - 1] + chemin

    return [longueur_chemin(chemin, d), chemin]


def créer_population(m: int, d: np.ndarray) -> list:
    return [créer_individu(d) for i in range(m)]


def réduire(p: list) -> None:
    p.sort()
    p[:] = p[:len(p) // 2]


def muter_chemin(c: list) -> None:
    p0, p1 = 0, 0
    while p0 == p1:
        p0, p1 = random.randint(1, len(c) - 1), random.randint(1, len(c) - 1)
    c[p0], c[p1] = c[p1], c[p0]


def muter_individu(I, d):
    muter_chemin(I[1])
    I[0] = longueur_chemin(I[1], d)  # on recalcule la distance


def muter_population(p: list, proba: float, d: np.ndarray) -> None:
    for individu in p[1:]:
        if random.random() < proba:
            muter_individu(individu, d)


def croiser(c1: list, c2: list) -> list:
    return normaliser_chemin(c1[:len(c1) // 2] + c2[len(c1) // 2:], len(c1))


def croiser_individus(i1, i2, d):
    c = croiser(i1[1], i2[1])
    return [longueur_chemin(c, d), c]


def nouvelle_génération(p: list, d: np.ndarray) -> None:
    m = len(d)
    for k in range(m - 1):
        p.append(croiser_individus(p[k], p[k + 1], d))
    p.append(croiser_individus(p[m - 1], p[0], d))


def algo_génétique(PI: np.ndarray, m: int, proba: float, g: int) -> (float, list):
    d = calculer_distances(PI)
    # init de la population
    pop = créer_population(m, d)


    for _ in range(g):

        réduire(pop)
        nouvelle_génération(pop, d)
        muter_population(pop, proba, d)

        #print(_, min(pop, key=lambda x: x[0]))

    return min(pop, key=lambda i: i[0])


nbPoint = 10
cmax = 100

PI = générer_PI(nbPoint, cmax)

print(PI)
print()

d = calculer_distances(PI)
for l in d:
    print(l)
print()

cpv = plus_proche_voisin(d)
lcpv = longueur_chemin(cpv, d)
print(cpv)
print(lcpv)
print()

lcag, cag = algo_génétique(PI, 30, 0.6, 100)
print(lcag, cag)

## view
import matplotlib.pyplot as plt

plt.figure(1)
plt.grid(True)
plt.axis([-1, cmax + 1, -1, cmax + 1])

# affichage des points
xp = [p[0] for p in PI]
yp = [p[1] for p in PI]

# chemin par algo du voisins proche
plt.subplot(131)
plt.title('plusProcheVoisin : {0:.2f}'.format(lcpv))

x = [position_robot()[0]] + [PI[p][0] for p in cpv[1:]]
y = [position_robot()[1]] + [PI[p][1] for p in cpv[1:]]

plt.plot(xp, yp, 'ro')
plt.grid(True)
plt.axis([-1, cmax + 1, -1, cmax + 1])

plt.plot(x, y)

# chemin par algo genetique
plt.subplot(132)
plt.title('génétique : {0:.2f}'.format(lcag))

x = [position_robot()[0]] + [PI[p][0] for p in cag[1:]]
y = [position_robot()[1]] + [PI[p][1] for p in cag[1:]]

plt.plot(xp, yp, 'ro')
plt.grid(True)
plt.axis([-1, cmax + 1, -1, cmax + 1])

plt.plot(x, y)






plt.show()
