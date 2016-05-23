x = [4, 2, 10, 9, "toto"]
y = x  # y est juste une reference au tableau x, pas de copie effectuee ici
y[2] = "tintin"
print x # Cela affiche  [4, 2, "tintin", 9, "toto"]
x = [4, 2, 10, 9, "toto"]
y = x[:]   # On demande une copie
y[2] = "tintin"
print x # Cela affiche  [4, 2, 10, 9, "toto"]
print y # Cela affiche  [4, 2, "tintin", 9, "toto"]