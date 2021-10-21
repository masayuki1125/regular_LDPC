#!/usr/bin/env python
# coding: utf-8

# In[239]:


import os
from AWGN import _AWGN
import numpy as np
import scipy
from scipy import sparse 
ch=_AWGN()


# In[240]:


class coding():
    
  def __init__(self,N):
    super().__init__() 

    self.encoder_var=1 #0:regular_LDPC 1:NR_LDPC(quasi cyclic)

    self.N=N
    self.R=1/2
    self.K=int(self.N*self.R)
    self.max_iter=50
    
    if self.encoder_var==0:#regular_LDPC
      #prepere constants
      self.Wc=3
      self.Wr=6

      if (self.Wr-self.Wc)/self.Wr!=self.R:
        print("encoder rate error")

      self.H=self.generate_regular_H()  
      self.tG=self.HtotG()
      print(self.H.shape)    
      self.filename="regular_LDPC_code_{}_{}".format(self.N,self.K)
    
    elif self.encoder_var==1:#NR_LDPC
      self.Zc,filename,BG_num,Kb=self.generate_filename(self.K,self.R)
      print(filename)
      self.H=self.generate_NR_H(BG_num,Kb,self.Zc,filename)

      #redifine N and K
      self.N=self.H.shape[1]-2*self.Zc
      self.K=self.N-self.H.shape[0]
      print((self.N,self.K))
      print(self.Zc)
      #modify H 
      #self.H=self.H[:self.K,:(self.H.shape[1]-self.H.shape[0]+self.K)]
  
      self.filename="NR_LDPC_code_{}_{}".format(self.N,self.K)
    
    self.H=sparse.csr_matrix(self.H)

    #np.savetxt("tG",self.tG,fmt='%i')
    #np.savetxt("H",self.H.toarray(),fmt='%i')

  @staticmethod
  def generate_filename(K,R):

    #decide BG_num
    if K<=3824 and R<=0.67:
      BG_num=2
    elif K<=292:
      BG_num=2
    elif R<=0.25:
      BG_num=2
    else:
      BG_num=1

    #decide Kb
    if BG_num==1:
      Kb=22
    else:
      if K>640:
        Kb=10
      elif 560<K<=640:
        Kb=9
      elif 192<K<=560:
        Kb=8
      elif K<=192:
        Kb=6

    #decide Zc

    a=np.arange(2,16)
    j=np.arange(0,8)

    a,j=np.meshgrid(a,j)
    a=a.flatten()
    j=j.flatten()

    Zc_array=a*(2**j)
    MAX_Zc=384
    Zc_array=Zc_array[MAX_Zc>=Zc_array]
    Zc=np.min(Zc_array[Zc_array>=K/Kb])

    #decide iLS
    i=list()
    i0=np.array([2,4,8,16,32,64,128,256])
    i1=np.array([3,6,12,24,48,96,192,384])
    i2=np.array([5,10,20,40,80,160,320])
    i3=np.array([7,14,28,56,112,224])
    i4=np.array([9,18,36,72,144,288])
    i5=np.array([11,22,44,88,176,352])
    i6=np.array([13,26,52,104,208])
    i7=np.array([15,30,60,120,240])
    i_list=[i0,i1,i2,i3,i4,i5,i6,i7]

    for count,i in enumerate(i_list):
      if np.any(i==Zc):
        iLS=count

    filename='NR_'+str(BG_num)+'_'+str(iLS)+'_'+str(Zc)+'.txt'

    return Zc,filename,BG_num,Kb

  @staticmethod
  def permute(a,Zc): #n*n単位行列をaだけシフトさせる
    if a==-1:
      tmp=np.zeros([Zc,Zc],dtype=int)
    else:
      tmp=np.identity(Zc,dtype=int)
      tmp=np.roll(tmp,a,axis=1)
    return tmp

  def generate_NR_H(self,BG_num,Kb,Zc,filename):

    base_matrix=np.loadtxt(os.path.join('base_matrices', filename),dtype='int')
    
    if BG_num==1:
        tmp=22
    elif BG_num==2:
        tmp=10
    
    Mb=np.arange((self.N-self.K)//Zc+1)
    Nb=np.zeros(Kb+2+len(Mb),dtype='int')
    Nb[:Kb+2]=np.arange(Kb+2)
    Nb[Kb+2:]=np.arange(tmp,tmp+len(Mb))
    #print(Nb)
    #print(Mb)

    H=np.empty((0,Zc*len(Nb)),dtype=int)
    for i in Mb:
        matrix_row=np.empty((Zc,0),dtype=int)

        for j in Nb:
            tmp=self.permute(base_matrix[i,j],Zc)
            matrix_row=np.concatenate([matrix_row,tmp],axis=1)

        H=np.concatenate([H,matrix_row],axis=0)
        
    return H

  #interleave N sequence
  @staticmethod
  def interleave(N):

    interleaver_sequence=np.arange(N)
    np.random.shuffle(interleaver_sequence)
    return interleaver_sequence

  def generate_regular_H(self):
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


# In[246]:


class encoding(coding):
  
  def __init__(self,N):
    super().__init__(N) 

  def generate_information(self):
    #generate information
    information=np.random.randint(0,2,self.K)
    return information

  def encode(self):
    if self.encoder_var==0:
      information=self.generate_information()
      codeword=self.tG@information%2
      
    
    elif self.encoder_var==1:
      information=self.generate_information()
      codeword=self.NR_encode(information)
    
    return information,codeword

  def NR_encode(self,information):

    info_num=self.K+2*self.Zc
    cwd_num=self.N+2*self.Zc

    codeword=np.zeros(cwd_num,dtype=int)

    #0:2*Zc:shortened code
    codeword[2*self.Zc:info_num]=information

    #double diagonal structure
    matrix=np.zeros((self.Zc,cwd_num))
    for i in range(4):
      matrix=(matrix+self.H[i*self.Zc:(i+1)*self.Zc])%2
    #print(matrix.shape)

    #first base-matrix-row parity check 
    for i in range(info_num,info_num+self.Zc):
      count=0
      while count<self.Zc:        
        if matrix[count,i]==1:
          codeword[i]=codeword@np.transpose(matrix[count,:])%2
          #print(i,count)
          break
        count+=1
    #K-K+Zcまでのparitybitを生成

    #K+Zc-K+3Zcまでのparitybitを生成
    for i in range(info_num+self.Zc,info_num+3*self.Zc):
      j=i-info_num-self.Zc
      codeword[i]=codeword@np.transpose(self.H[j,:])%2
      #print(codeword[i])
      #print(i)
      #print(codeword[K:])
      #from IPython.core.debugger import Pdb; Pdb().set_trace()

    #nomal structure
    for i in range(info_num+3*self.Zc,cwd_num):
      j=i-info_num
      codeword[i]=codeword@np.transpose(self.H[j,:])%2

    if np.any(self.H@codeword%2!=0):
      print("cword err")
    
    return codeword


# In[247]:


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

  def sum_product(self,Lc):
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
    EST_codeword=self.sum_product(Lc)
    #EST_information=EST_codeword[(self.N-self.K):] #systematicじゃないので、情報ビットだけで測れない
    return EST_codeword


# In[250]:


class LDPC(encoding,decoding):
  
  def __init__(self,N):
    super().__init__(N) 
      
  def main_func(self,EbNodB):
    information,codeword=self.encode()
    Lc=ch.generate_LLR(codeword,EbNodB)

    if self.encoder_var==1:
      INF=10**10
      Lc[:2*self.Zc]==-1*INF

    EST_codeword=self.decode(Lc)

    if self.encoder_var==0:
      return codeword,EST_codeword
    
    elif self.encoder_var==1:
      return information, EST_codeword[2*self.Zc:self.K+2*self.Zc]

    


# In[253]:


if __name__=="__main__":
  #cd=coding(1024)
  ldpc=LDPC(1024)
  main_func=ldpc.main_func
  EbNodB=0

  def output(EbNodB):
    count_err=0
    count_all=0
    count_berr=0
    count_ball=0
    MAX_ERR=8

    while count_err<MAX_ERR:
      
      information,EST_information=ldpc.main_func(EbNodB)
    
      if np.any(information!=EST_information):#BLOCK error check
        count_err+=1
      
      count_all+=1

      #calculate bit error rate 
      count_berr+=np.sum(information!=EST_information)
      count_ball+=ldpc.N

      print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="        ,count_ball,"count_berr=",count_berr,end="")

    #print("\n")
    #print("BER=",count_berr/count_ball)
    return  count_err,count_all,count_berr,count_all
  
  output(0)

