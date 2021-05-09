# do the imports
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
from scipy.linalg import pinvh  # Needed to calculate the pseudo-inverse
from scipy.stats import f

class RBFNN(object):
    '''
    '''
    def __init__(self, df):
        '''
        Constructor 
        '''
        self.df = df

    def rbfTrain(self,X,y,input_spread = 0.14):
        '''
        Modeled after the rfbTrain.m matlab provided in 
        EN.685.621
        Input:
          X [n x d] training data with n observations and dimension d
          y [n x 1] labeld targets for classification two classes [-1 1]
          spread
          
        Output:
          model -- structure containing:
              .W_hat -- Layer weights
              .W -- input weights
              .bias 
              .spread
              .input_spread
              .error -- training error [0 1]
    
        Much of the commands used below are 
        
        Big-O = O(n^2)
        Big-T = T(8n^2 11n + 10)
        '''
           
        n,d = X.shape                                                                    
        X = X.T                                                                          
        H = []                                                                           
        spread = input_spread                                                            
        # n2 is the number of observations so if we are thinking in terms of 
        # of big-O we need to imagine alot of observations
        #
        # Further, if one were to think in terms of vector optimiation of arrays
        # if even one of lines of code in the for loop is 
        # n, then its n^2
        pbar = tqdm(range(n))
        for j in pbar: # There are  10
            pbar.set_description(f'M Ob: {j}')
        #for j in range(n):                                                               #n*  
            W=np.array([X[:,j].T])                                                       #n
            D = X - np.array([W[0]]*n).T                                                 #n
            ss = 2* np.power(spread,2)                                                   #n
            s = np.array([self.xDiag(D.T,D)]).T /ss                                #T(3+5n)
            H.append(np.exp(-s))                                                         #1

        Htmp = np.round(H,4).reshape(n,n) # Make it a pretty shape                       #n
        HH = np.dot(Htmp.T,Htmp) # H' * H   Check                                        #n  
        HHH = pinvh(HH) # pinv(H'*H)                                                     #n
        HHHH = np.dot(HHH,Htmp.T) # pinv(H' * H) * H'                                    #n 
        W_hat = np.dot(HHHH,y) # pinv(H'*H) * H' * y                                     #n
        yt = np.dot(Htmp, W_hat).T # yt = (H*W_hat)'                                     #n
        ypred = np.ones(np.size(y))                                                      #n
        ypred[np.where(yt<0)] = -1                                                       #n
        uq,co =np.unique(y==ypred,return_counts=True)                                    #n
        # Get unique and count
        # https://stackoverflow.com/questions/28663856/\
        # how-to-count-the-occurrence-of-certain-item-in-an-ndarray
        rst = dict(zip(uq,co))                                                           #n
        predError = 1- rst[True]/np.size(y)                                              #n
        
        model = {}                                                                       #1
        model['W_hat'] = W_hat                                                           #1
        model['W'] = X                                                                   #1
        model['spread'] = spread                                                         #1
        model['error'] = predError                                                       #1
        return model                                                                     #1
            
    def xDiag(self,X1,X2):
        '''
        Method modeled after matlab code provided in EN685.621
        I think this can be simplified but need to move forward on 
        
        Big-O = O(n)
        Big-T = T(3+5n)
        
        '''
        r1,c1 = X1.shape                                                                 #1
        r2,c2 = X2.shape                                                                 #1
        X1tmp = X1.reshape(r1*c1)                                                        #n
        X2tmp = X2.T.reshape(r2*c2)                                                      #n
        X = np.array(X1tmp * X2tmp).reshape(r1,c1).T                                     #n
        r1,c1  = X.shape                                                                 #n
        # It's either the if or the else -- can't be both
        if r1 > 1:                                                                       #1
            return np.sum(X.T,axis=1)                                                    #n
        else:                # This is doing one or the other not both so only count one
            return X.T

 
    def classify(self,X,model):
        '''
        This particular version works on larger data sets
        
        Big-O = O(n^2)
        Big-T = T(6n^2+ 4n + 5)
        '''
        n1,d1 = X.shape                                                                  #1
        X = X.T                                    
        n2,d2 = model['W'].T.shape                 
        H = np.zeros((n1,n2))                      
        
        # n2 is the number of observations so if we are thinking in terms of 
        # of big-O we need to imagine alot of observations
        #
        # Further, if one were to think in terms of vector optimiation of arrays
        # if even one of lines of code in the for loop is 
        # n, then its n^2
        pbar = tqdm(range(n2))
        for j in pbar: # There are  10
        #for j in range(n2):                       
            pbar.set_description(f'C Ob: {j}')
            M = np.array([model['W'][:,j]]).T      
            MM = np.array([M]*n1).T.reshape(d2,n1) 
            D = np.array(X - MM).reshape(d2,n1)    
            ss = 2* np.power(model['spread'],2)    
            s = np.array([self.xDiag(D.T,D)]).T /ss
            H[:,j] = np.exp(-s).T                  
        
        W_hat = np.array(model['W_hat'])           
        y = np.array(np.round(np.dot(H,W_hat).T,4))
        ypred = np.ones(np.size(y))
        ypred[np.where(y<0)] = -1
        
        return y,ypred

# Doing this because I need to import this notebook into another one
if __name__ == '__main__':
    numRows = 12000
    numRows =5000
    #df = pd.read_csv('train.csv',nrows=numRows)
    df = pd.read_excel('trainFeatures42k.xls',header=None,nrows = numRows)
    mpp.applyFisher()
    print("\n\t - These are the features you want:\n\t{}".format(mpp.model['featureIndex'][0:10,]))
    # Only want the top ten features
    print("\n\t3. Extracting Features.")
    #mpp.OnlyTheTopTenFeatures()
    print("\n\t4. Normalizing Data ")
    #mpp.normalizeData()
    #print("\n\t5. Performing Wilks Outlier Removal to remove outliers")
    #print("\n\t - size before: {}".format(mpp.topTenPcaDfNorm.shape ))
    #mpp.removeOutliers()
    #print("\n\t - outlier removal complete")
    #print("\n\t - size after: {}".format(mpp.topTenPcaDfNormOutRemoved.shape ))
    #print("\n\t6. Spitting data into training and testing test")
    #mpp.splitTheData()
    RBFNN = RBFNN(df)
    #acc,cm = mRBFNN.doRBFNN(spread=0.14)
    print("\n\t8. Plotting Confusion Matrix for SVM using RBF Kernal")
    #fig, ax = plt.subplots(figsize=(10,10))
    #cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax)
