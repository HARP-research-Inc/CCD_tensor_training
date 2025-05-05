
# take in an underlying set
set = {"a", "b", "c", "d", "e"}
# take in a set of orderings [a,b] between elements of the underlying set
orderings = [["a","b"],["b","c"],["d","e"], ["a","d"], ["d","a"]]
# look for sets of two orderings with opposite domain and codomain
equivalences = []
strict_orders = []

for rel in orderings:
    invrel = [rel[1],rel[0]]
    if invrel in orderings:
        equivalences += [invrel]
    else:
        strict_orders += [rel]

print(equivalences)
print(strict_orders)