#!/usr/bin/env python
# coding: utf-8

# In[5]:

import sys
sys.path.append("../channel")
from AWGN import _AWGN
import numpy as np
import scipy
from scipy import sparse 
ch=_AWGN()

# In[6]:


class coding():
    
    def __init__(self,N):
        super().__init__() 


        self.N=N
        self.Wc=3
        self.Wr=6
        self.max_iter=20
        
        


        #prepere constants
        self.K=self.N*(self.Wr-self.Wc)//self.Wr
        self.R=self.K/self.N
        self.H=self.generate_H()
        self.tG=self.HtotG()
        self.H=sparse.csr_matrix(self.H)
        self.filename="regular_LDPC_code_{}_{}".format(self.N,self.K) 
        

    #interleave N sequence
    @staticmethod
    def interleave(N):

        interleaver_sequence=np.arange(N)
        np.random.shuffle(interleaver_sequence)
        return interleaver_sequence

    def generate_H(self):
        '''
        #generate regular parity check matrix
        #-----------
        #Wr : row weight
        #Wc : column weight
        #N : length of codeword 
        '''

        if self.N*self.Wc%self.Wr!=0:
            print("constant err")
            exit()

        

        #generate sub_H matrix(Wc=1)
        sub_H=np.zeros(((self.N-self.K)//self.Wc,self.N),dtype=int)
        for i in range((self.N-self.K)//self.Wc):
            sub_H[i][self.Wr*i:self.Wr*(i+1)]=1

        H=sub_H

        #generate other sub_H matrix(Wc=1)
        for i in range(self.Wc+1):
            sub_H2=sub_H[:,self.interleave(self.N)]
            H=np.concatenate((H,sub_H2))
        
        H=H[:self.K,:]

        return H 

#from https://github.com/hichamjanati/pyldpc/blob/master/pyldpc/code.py 
    @staticmethod
    def binaryproduct(X, Y):
        """Compute a matrix-matrix / vector product in Z/2Z."""
        A = X.dot(Y)
        try:
            A = A.toarray()
        except AttributeError:
            pass
        return A % 2

    @staticmethod
    def gaussjordan(X, change=0):
        """Compute the binary row reduced echelon form of X.
        Parameters
        ----------
        X: array (m, n)
        change : boolean (default, False). If True returns the inverse transform
        Returns
        -------
        if `change` == 'True':
            A: array (m, n). row reduced form of X.
            P: tranformations applied to the identity
        else:
            A: array (m, n). row reduced form of X.
        """
        A = np.copy(X)
        m, n = A.shape

        if change:
            P = np.identity(m).astype(int)

        pivot_old = -1
        for j in range(n):
            filtre_down = A[pivot_old+1:m, j]
            pivot = np.argmax(filtre_down)+pivot_old+1

            if A[pivot, j]:
                pivot_old += 1
                if pivot_old != pivot:
                    aux = np.copy(A[pivot, :])
                    A[pivot, :] = A[pivot_old, :]
                    A[pivot_old, :] = aux
                    if change:
                        aux = np.copy(P[pivot, :])
                        P[pivot, :] = P[pivot_old, :]
                        P[pivot_old, :] = aux

                for i in range(m):
                    if i != pivot_old and A[i, j]:
                        if change:
                            P[i, :] = abs(P[i, :]-P[pivot_old, :])
                        A[i, :] = abs(A[i, :]-A[pivot_old, :])

            if pivot_old == m-1:
                break

        if change:
            return A, P
        return A

    def HtotG(self,sparse=True):
        """Return the generating coding matrix G given the LDPC matrix H.
        Parameters
        ----------
        H: array (n_equations, n_code). Parity check matrix of an LDPC code with
            code length `n_code` and `n_equations` number of equations.
        sparse: (boolean, default True): if `True`, scipy.sparse format is used
            to speed up computation.
        Returns
        -------
        G.T: array (n_bits, n_code). Transposed coding matrix.
        """
        if type(self.H) == scipy.sparse.csr_matrix:
            self.H = self.H.toarray()
        n_equations, n_code = self.H.shape

        # DOUBLE GAUSS-JORDAN:

        Href_colonnes, tQ = self.gaussjordan(self.H.T, 1)

        Href_diag = self.gaussjordan(np.transpose(Href_colonnes))

        Q = tQ.T

        n_bits = n_code - Href_diag.sum()

        Y = np.zeros(shape=(n_code, n_bits)).astype(int)
        Y[n_code - n_bits:, :] = np.identity(n_bits)

        if sparse:
            Q = scipy.sparse.csr_matrix(Q)
            Y = scipy.sparse.csr_matrix(Y)

        tG = self.binaryproduct(Q, Y)

        return tG


# In[8]:


class encoding(coding):

    def __init__(self,N):
        super().__init__(N) 

    def generate_information(self):
        #generate information
        information=np.random.randint(0,2,self.K)
        return information

    def encode(self):
        information=self.generate_information()
        codeword=self.tG@information%2
        return information,codeword

# In[ ]:


class decoding(coding):
    def __init__(self,N):
        super().__init__(N)

        self.ii,self.jj=self.H.nonzero()
        self.m,self.n=self.H.shape

    def phi(self,mat):
        '''
        input: 2D matrix 
        output: 2D matrix (same as H)
        '''
        smat=mat.toarray()[self.ii, self.jj]

        #clipping operaiton
        smat[smat>10**2]=10**2
        smat[smat < 10**-5] = 10**-5

        smat=np.log((np.exp(smat) + 1) / (np.exp(smat) - 1))

        mat=sparse.csr_matrix((smat, (self.ii, self.jj)), shape=(self.m, self.n))

        return mat
    
    def make_alpha(self,mat):
        '''
        input: 2D matrix(same as H)
        output: 2D matrix (same as H)
        '''
        smat= mat.toarray()[self.ii, self.jj]

        salpha=np.sign(smat)
        alpha=sparse.csr_matrix((salpha, (self.ii, self.jj)), shape=(self.m, self.n))
        
        mask=(alpha-self.H).getnnz(axis=1)%2 #-1の数が奇数なら１、偶数なら０を出力
        mask=-2*mask+1
        #列ごとに掛け算する マイナスの列は１、プラスの列は０
        alpha=sparse.spdiags(mask, 0, self.m, self.m, 'csr').dot(alpha)
        return alpha

    def make_beta(self,mat):
        '''
        input: 2D array
        output: 2D matrix (same as H)
        '''
        smat= mat.toarray()[self.ii, self.jj]

        sbeta=np.abs(smat)
        beta=sparse.csr_matrix((sbeta, (self.ii, self.jj)), shape=(self.m, self.n))

        #leave-one-out operation
        beta=self.phi(beta)
        mask=beta.sum(axis=1).ravel()
        tmp=sparse.spdiags(mask, 0, self.m, self.m, 'csr').dot(self.H)
        beta=tmp-beta
        beta=self.phi(beta)

        return beta
        
    def ldpc_decode(self,Lc):
        # initialization
        L_mat = self.H.dot(sparse.spdiags(Lc, 0, self.n, self.n, 'csr'))
        k=0 #itr counter

        while k < self.max_iter:
            ##horizontal operation from L_mat to L_mat 
            #calcurate alpha
            alpha=self.make_alpha(L_mat)
            #culcurate beta
            beta=self.make_beta(L_mat)

            L_mat=alpha.multiply(beta)

            ##vertical operation
            stmp=L_mat.sum(axis=0).ravel()
            stmp+=Lc
            tmp=self.H.dot(sparse.spdiags(stmp, 0, self.n, self.n, 'csr'))
            L_mat=tmp-L_mat

            ##check operation
            EST_Lc=L_mat.sum(axis=0)
            EST_Lc+=Lc
            EST_codeword=(np.sign(EST_Lc)+1)/2

            #convert from matrix class to array class
            EST_codeword=(np.asarray(EST_codeword)).flatten()
            if np.all(self.H.dot(EST_codeword)%2 == 0):
                break
            k+=1
        
        return EST_codeword

    def decode(self,Lc):
        EST_codeword=self.ldpc_decode(Lc)
        #EST_information=EST_codeword[(self.N-self.K):] #systematicじゃないので、情報ビットだけで測れない
        return EST_codeword


# In[ ]:


class LDPC(encoding,decoding):

    def __init__(self,N):
        super().__init__(N) 
        
    def main_func(self,EbNodB):
        _,codeword=self.encode()
        Lc=ch.generate_LLR(codeword,EbNodB)
        EST_codeword=self.decode(Lc)
        return codeword,EST_codeword


# In[ ]:


if __name__=="__main__":
    ldpc=LDPC(20)
    print(ldpc.N)
    main_func=ldpc.main_func
    EbNodB=0
'''
    def output(EbNodB):
      count_err=0
      count_all=0
      count_berr=0
      count_ball=0
      MAX_ERR=8

      while count_err<MAX_ERR:
        
        information,EST_information=ldpc.LDPC(EbNodB)
      
        if np.any(information!=EST_information):#BLOCK error check
          count_err+=1
        
        count_all+=1

        #calculate bit error rate 
        count_berr+=np.sum(information!=EST_information)
        count_ball+=ldpc.N

        #print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="\
              #,count_ball,"count_berr=",count_berr,end="")

      #print("\n")
      #print("BER=",count_berr/count_ball)
      return  count_err,count_all,count_berr,count_all
    
    
    #results=[output.remote(EbNodB) for EbNodB in range(-3,5)]
    #print(ray.get(results))
'''

