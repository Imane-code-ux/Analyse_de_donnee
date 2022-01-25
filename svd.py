from PIL import Image
import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt

#on ouvre l'image qu'on veut compresser
im=Image.open("C:\\Users\BALADI\Desktop\image_add\lena_gris.png")
#on stocke dans T l'image convertie en matrice de pixels
T=np.array(im)
#on applique la svd à la matrice T      
A=alg.svd(T)
#on trace les valeurs singulière en fonction du nombre de valeurs sigulière
c=A[1]
k=[i for i in range(0,T.shape[1])]
plt.plot(k,c)
plt.xlabel("k") 
plt.ylabel("sigmas")
plt.title("valeurs singulière en fonction de son nombre k")
plt.legend()
plt.grid()
plt.show()
#fonction de la compression d'image 
def compression(T,k):
  A=alg.svd(T)
  U=A[0]
  c=A[1]
  Vt=A[2]
  d=[c[i] for i in range(0,k)]
  s=np.zeros((k,k))
  for i in range(len(d)):
    s[i][i]=d[i]
  M=U[:,:k]@s@Vt[:k,:]
  im2=Image.fromarray(M)
  return im2
compression(T,241)
im3=Image.open("C:\\Users\BALADI\Desktop\image_add\lena.png") 
	
T2=np.array(im3)
T2R=T2[:,:,0]
T2G=T2[:, :,1]
T2B=T2[:,:,2]
T2R_compressed = compression(T2R,60).convert('L')
T2G_compressed = compression(T2G,60).convert('L')
T2B_compressed = compression(T2B,60).convert('L')

new_image=Image.merge("RGB",(T2R_compressed,T2G_compressed,T2B_compressed))
new_image.show()
im3.show()



def variance(c):
  k=0
  v=0
  a=np.sum(c)
  while v<0.95 or v>0.96 :
    v=(np.sum(c[i] for i in range(k+1))/a)**2
    k+=1
  return k
print(variance(c))
