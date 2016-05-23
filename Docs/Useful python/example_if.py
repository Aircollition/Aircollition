x = (raw_input("Donne moi un entier: "))
if x < 0:          # est-ce que x est negatif ?
    x = 0          # si oui on le remplace par zero
    print "Je remplace ton nombre negatif par zero"
elif x == 0:       # est-ce que x est egal a zero ?
    print 'Zero'   # si oui on affiche "Zero"
elif x == 1:       # est-ce que x est egal a 1 ?
    print 'Un'     # si oui on affiche "Un"
else:              # sinon
    print 'Plus'   # on affiche "Plus"
