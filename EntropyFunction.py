def entropy(p):
   if p == 0 or p == 1:
       return 0
   else:
       return -p * np.log2(p) - (1- p)*np.log2(1 - p)
   
print(entropy(0.5))
