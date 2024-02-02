#https://arxiv.org/pdf/2202.03176.pdf

from numpy import *

from pdb import set_trace as pause

class BFoV:
   
    def __init__(self, theta, phi, alpha, beta, degrees=True):
        # theta and phi are the longitude and latitude coordinates
        # alpha and beta are the horizontal and vertical FoVs
        self.theta = theta
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        if degrees: self.__utils__deg2rad__()
           
    def __str__(self):
        return '(' + str(self.theta) + ', ' + \
          str(self.phi) + ', ' + \
          str(self.alpha) + ', ' + \
          str(self.beta) + ')'
    
    def __utils__deg2rad__(self):
        '''
        self.theta = self.theta*pi/180. # 0-360? -> 0-2*pi
        self.phi = (self.phi)*pi/180. # -90-90? -> -pi/2-pi/2
        #self.phi = (self.phi+90.)*pi/180. # -90-90? -> 0-pi
        self.alpha = self.alpha*pi/180. # 0-360? -> 0-2*pi
        self.beta = self.beta*pi/180. # 0-360? -> 0-2*pi
        '''
        # AFAIK, same as:
        self.theta = deg2rad(self.theta)
        self.phi = deg2rad(self.phi)
        self.alpha = deg2rad(self.alpha)
        self.beta = deg2rad(self.beta)
        

def FoVIoU(Bg, Bd):

    # Step 1 of Alg. 1
    ABg = Bg.alpha * Bg.beta
    ABd = Bd.alpha * Bd.beta
        
    # Step 2 of Alg. 1
    deltaFoV = (Bd.theta - Bg.theta) * cos((Bg.phi+Bd.phi)/2)
    
    # Step 3 of Alg. 1
    iThetaMin = max(-Bg.alpha/2,deltaFoV-Bd.alpha/2)
    iThetaMax = min(Bg.alpha/2,deltaFoV+Bd.alpha/2)
    iPhiMin = max(Bg.phi-Bg.beta/2, Bd.phi-Bd.beta/2) 
    iPhiMax = min(Bg.phi+Bg.beta/2, Bd.phi+Bd.beta/2)
    iAlpha = iThetaMax - iThetaMin
    iBeta = iPhiMax - iPhiMin 
    Bi = BFoV(0, 0, iAlpha, iBeta, degrees=False)
    
    # Step 4 of Alg. 1
    ABi = Bi.alpha * Bi.beta

    if iThetaMax > iThetaMin and iPhiMax > iPhiMin:
        # Compute area of FoV intersection I
        ABi = Bi.alpha * Bi.beta
    else:
        ABi = 0

    ABu = ABg + ABd - ABi

    # Step 5 of Alg. 1
    FoV = ABi/ABu
    return FoV

def SphIoU(Bg, Bd):

    ABg = 2*Bg.alpha * sin(Bg.beta/2)
    ABd = 2*Bd.alpha * sin(Bd.beta/2)
            
    iThetaMin = max(Bg.theta-Bg.alpha/2,Bd.theta-Bd.alpha/2)
    iThetaMax = min(Bg.theta+Bg.alpha/2,Bd.theta+Bd.alpha/2)
    iPhiMin = max(Bg.phi-Bg.beta/2, Bd.phi-Bd.beta/2) 
    iPhiMax = min(Bg.phi+Bg.beta/2, Bd.phi+Bd.beta/2)
    iAlpha = iThetaMax - iThetaMin
    iBeta = iPhiMax - iPhiMin 
    
    ABi = iAlpha * iBeta
    ABu = ABg + ABd - ABi

    FoV = ABi/ABu
    return FoV
   
   
if __name__ == '__main__':

   # Table I. Should be 0.59/0.33
   Bg = BFoV(30, 60, 60, 60)
   Bd = BFoV(60, 60, 60, 60)
   print(FoVIoU(Bg, Bd)) #0.6000
   #print(SphIoU(Bg, Bd)) #0.3546
   # Table II. [[a,b,c],[d,e,f]]
   # Table II.a. Should be 0.235/0.227
   B1 = BFoV(40, 50, 35, 55)
   B2 = BFoV(35, 20, 37, 50)
   print(FoVIoU(B1, B2)) #0.2348
   #print(SphIoU(B1, B2)) #0.2367
   # Table II.b. Should be 0.323/0.250
   B1 = BFoV(30, 60, 60, 60)
   B2 = BFoV(55, 40, 60, 60)
   print(FoVIoU(B1, B2)) #0.3228
   #print(SphIoU(B1, B2)) #0.2556
   # Table II.c. Should be 0.617/0.112
   B1 = BFoV(50, -78, 25, 46)
   B2 = BFoV(30, -75, 26, 45)
   print(FoVIoU(B1, B2)) #0.6170
   #print(SphIoU(B1, B2)) #0.1153
   # Table II.d. Should be 0.589/0.00
   B1 = BFoV(30, 75, 30, 60)
   B2 = BFoV(60, 40, 60, 60)
   print(FoVIoU(B1, B2)) #0.1543
   #print(SphIoU(B1, B2)) #0.0784
   # Table II.e. Should be 0.259/0.073
   B1 = BFoV(40, 70, 25, 30)
   B2 = BFoV(60, 85, 30, 30)
   print(FoVIoU(B1, B2)) #0.2668
   #print(SphIoU(B1, B2)) #0.0740
   # Table II.e. Should be 0.538/0.212
   B1 = BFoV(30, 75, 30, 30)
   B2 = BFoV(60, 55, 40, 50)
   print(FoVIoU(B1, B2)) #0.1819
   #print(SphIoU(B1, B2)) #0.0366
