#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
from scipy import sparse
import multiprocessing
import multiprocessing.pool


# In[ ]:


def generate_information(K):
  #generate information
  information=np.random.randint(0,2,K)
  return information


# In[ ]:


#interleave N sequence
def interleave(N):
  interleaver_sequence=np.arange(N)
  np.random.shuffle(interleaver_sequence)
  return interleaver_sequence

def generate_H(N,Wc,Wr):
  '''
  generate regular parity check matrix
  -----------
  Wr : row weight
  Wc : column weight
  N : length of codeword 
  '''

  if N*Wc%Wr!=0:
    print("constant err")
    exit()

  K=N*(Wr-Wc)//Wr

  #generate sub_H matrix(Wc=1)
  sub_H=np.zeros(((N-K)//Wc,N),dtype=int)
  for i in range((N-K)//Wc):
    sub_H[i][Wr*i:Wr*(i+1)]=1

  H=sub_H

  #generate other sub_H matrix(Wc=1)
  for i in range(Wc-1):
    sub_H2=sub_H[:,interleave(N)]
    H=np.concatenate((H,sub_H2))
  return H


# In[ ]:


#from https://github.com/hichamjanati/pyldpc/blob/master/pyldpc/code.py 

def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2

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

def HtotG(H, sparse=True):
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
    if type(H) == scipy.sparse.csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = gaussjordan(H.T, 1)

    Href_diag = gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q = scipy.sparse.csr_matrix(Q)
        Y = scipy.sparse.csr_matrix(Y)

    tG = binaryproduct(Q, Y)

    return tG


#check
#H=generate_H(400,3,4)
#tG=HtotG(H)
#np.savetxt("H,txt",H,fmt="%d")
#print(tG.shape)
#print(H.shape)
#print(H@tG%2)


# In[ ]:


#複数のアンテナ用に行列を作成
def generate_codeword_matrix(tG,TX_antenna):
  '''
  this function uses below functions:
    generate_information

  '''

  N,K=tG.shape

  info=generate_information(K)

  codeword=tG@info%2

  row_number=codeword.shape[0]//TX_antenna
  codeword_matrix=np.ndarray.copy(codeword.reshape(TX_antenna,row_number))

  #if codeword.shape[0]%TX_antenna!=0:
    #print("err generate_codeword_matrix!")
    #exit()
  return codeword,codeword_matrix


# In[ ]:


#マッピング用の行列に変更
def generate_mapping(codeword_matrix,modulation_symbol):
  modulation_bits=np.log2(modulation_symbol).astype(int)

  if codeword_matrix.shape[1] % modulation_bits !=0:
    print("err generate_mapping!")
    exit()
  
  for i in range(modulation_bits):
    codeword_matrix[:,i::modulation_bits]=codeword_matrix[:,i::modulation_bits]*2**(modulation_bits-i-1)

  mapping_matrix=np.zeros((codeword_matrix.shape[0],codeword_matrix.shape[1]//2),dtype=int)
  #print(mapping_matrix.shape)
  for i in range(0,codeword_matrix.shape[1],modulation_bits):
    mapping_matrix[:,i//2]=np.sum(codeword_matrix[:,i:i+modulation_bits],axis=1)

  return mapping_matrix


# In[ ]:


#マッピングの行列を実際の変調したコンスタレーションに変更
def modulate(mapping_matrix):
  constellation=np.zeros(mapping_matrix.shape,dtype=complex)
  constellation[mapping_matrix==0]=1+1j
  constellation[mapping_matrix==1]=-1+1j
  constellation[mapping_matrix==2]=1-1j
  constellation[mapping_matrix==3]=-1-1j
  return constellation


# In[ ]:


#only 1*1 antenna
#チャネルを通す
def channel(constellation,EbNodB,RX_antenna):

  # Additive Gaussian White Noiseの生成する際のパラメータ設定
  EbNo = 10 ** (EbNodB / 10)
  No=1/EbNo #Eb=1(fixed)

  # AWGN雑音の生成
  noise = np.random.normal(0, np.sqrt(No / 2), (RX_antenna, constellation.shape[1]))           + 1j * np.random.normal(0, np.sqrt(No / 2), (RX_antenna, constellation.shape[1]))

  # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
  RX_constellation = constellation + noise

  # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
  #print(np.dot(noise[0, :], np.conj(noise[0, :]))/bit_num)

  return RX_constellation


# In[ ]:


def demodulate(RX_constellation):
  return RX_constellation.real,RX_constellation.imag


# In[ ]:


#from https://github.com/willemolding/FastPythonLDPC
# function [x_hat, success, k, prob ] = ldpc_decode(f0,f1,H,max_iter)
# decoding of binary LDPC as in Elec. Letters by MacKay&Neal 13March1997
# For notations see the same reference.
# function [x_hat, success, k] = ldpc_decode(y,f0,f1,H)
# outputs the estimate x_hat of the ENCODED sequence for
# the received vector y with channel likelihoods of '0' and '1's
# in f0 and f1 and parity check matrix H. Success==1 signals
# successful decoding. Maximum number of iterations is set to 100.
# k returns number of iterations until convergence.
#
# Example:
# We assume G is systematic G=[A|I] and, obviously, mod(G*H',2)=0
# sigma = 1;                          # AWGN noise deviation
# x = (sign(randn(1,size(G,1)))+1)/2; # random bits
#         y = mod(x*G,2);                     # coding
#         z = 2*y-1;                          # BPSK modulation
#         z=z + sigma*randn(1,size(G,2));     # AWGN transmission
#
#         f1=1./(1+exp(-2*z/sigma^2));        # likelihoods
#         f0=1-f1;
#         [z_hat, success, k] = ldpc_decode(z,f0,f1,H);
#         x_hat = z_hat(size(G,2)+1-size(G,1):size(G,2));
#         x_hat = x_hat';

#   Copyright (c) 1999 by Igor Kozintsev igor@ifp.uiuc.edu
#   $Revision: 1.1 $  $Date: 1999/07/11 $
#   fixed high-SNR decoding


def ldpc_decode(f0, f1, H, max_iter=20):
    """
    A python port of the ldpc_decode matlab code.
    Parameters:
    ----------
    f0 : 1D numpy array
        see matlab docstring
    f1 : 1D numpy array
        see matlab docstring
    H : 2D scipy.sparse.csc_matrix
        Must be a scipy sparse array of the csc type. This is the same type that matlab used so we remain compatible.
    max_iter : integer
        maximum number of iterations
    Returns:
    --------
    x_hat : 1D numpy array
        Error corrected ENCODED sequence
    success : bool
        indicates successful convergence e.g. parity check passed
    k : integer
        number of iterations to converge
    prob :
    """

    # check the matrix is correctly orientated and transpose it if required
    [m, n] = H.shape
    if m > n:
        H = H.t
        [m, n] = H.shape

    # if ~issparse(H)  # make H sparse if it is not sparse yet
    #     [ii, jj, sH] = find(H);
    #     H = sparse(ii, jj, sH, m, n);

    # initialization
    ii, jj = H.nonzero()

    q0 = H.dot(sparse.spdiags(f0, 0, n, n, 'csc'))
    sq0 = q0[ii, jj].getA1()
    sff0 = sq0

    q1 = H.dot(sparse.spdiags(f1, 0, n, n, 'csc'))
    sq1 = q1[ii, jj].getA1()
    sff1 = sq1

    # iterations
    k = 0
    success = 0
    while success == 0 and k < max_iter:
        k += 1

        # horizontal step
        sdq = sq0 - sq1
        sdq[sdq == 0] = 1e-20  # if   f0 = f1 = .5
        dq = sparse.csc_matrix((sdq, (ii, jj)), shape=(m, n))

        dq.data = np.log(dq.data.astype(complex))
        Pdq_v = np.real(np.exp(dq.sum(axis=1)))

        Pdq = sparse.spdiags(Pdq_v.ravel(), 0, m, m, 'csc').dot(H)
        sPdq = Pdq[ii, jj].getA1()

        sr0 = (1 + sPdq / sdq) / 2.
        sr0[abs(sr0) < 1e-20] = 1e-20
        sr1 = (1 - sPdq / sdq) / 2.
        sr1[np.abs(sr1) < 1e-20] = 1e-20
        r0 = sparse.csc_matrix((sr0, (ii, jj)), shape=(m, n))
        r1 = sparse.csc_matrix((sr1, (ii, jj)), shape=(m, n))

        # vertical step
        r0.data = np.log(r0.data.astype(complex))
        Pr0_v = np.real(np.exp(r0.sum(axis=0)))

        Pr0 = H.dot(sparse.spdiags(Pr0_v.ravel(), 0, n, n, 'csc'))
        sPr0 = Pr0[ii, jj].getA1()
        Q0 = np.array(sparse.csc_matrix((sPr0 * sff0, (ii, jj)), shape=(m, n)).sum(axis=0)).T

        sq0 = sPr0 * sff0 / sr0

        r1.data = np.log(r1.data.astype(complex))
        Pr1_v = np.real(np.exp(r1.sum(axis=0)))

        Pr1 = H.dot(sparse.spdiags(Pr1_v.ravel(), 0, n, n, 'csc'))
        sPr1 = Pr1[ii, jj].getA1()

        Q1 = np.array(sparse.csc_matrix((sPr1 * sff1, (ii, jj)), shape=(m, n)).sum(axis=0)).T
        sq1 = sPr1 * sff1 / sr1

        sqq = sq0 + sq1
        sq0 = sq0 / sqq
        sq1 = sq1 / sqq

        # tentative decoding
        QQ = Q0 + Q1
        prob = Q1 / QQ
        Q0 = Q0 / QQ
        Q1 = Q1 / QQ

        tent = (Q1 - Q0)  # soft?
        x_hat = (np.sign(tent) + 1) / 2  # hard bits estimated

        if np.all(np.fmod(H.dot(x_hat), 2) == 0):
            success = 1

    return x_hat.flatten(), success, k


# In[ ]:


def monte_carlo(inputs):

  #prepare constants
  N,  Wc,  Wr,  TX_antenna,  RX_antenna,  modulation_symbol,  EbNodB,  L_MAX,  seed  =inputs

  #seed値の設定
  np.random.seed(seed)
  
  #prepare some constants
  MAX_ERR=1
  count_bitall=0
  count_biterr=0
  count_all=0
  count_err=0
  H=generate_H(N,Wc,Wr)
  tG=HtotG(H)
  H=scipy.sparse.csc_matrix(H)

  while count_err<MAX_ERR:

    codeword,codeword_matrix=generate_codeword_matrix(tG,TX_antenna) #generate word n*N/n
    mapping_matrix=generate_mapping(codeword_matrix,modulation_symbol) #generate constellation number 0~3(example 4-QAM
    constellation=modulate(mapping_matrix) #generate constellation Q-phase and I-phase
    RX_constellation=channel(constellation,EbNodB,RX_antenna) #generate recieve constellation
    y1,y2=demodulate(RX_constellation) #generate 1-d constellation which is decoded alteratively
    y=np.zeros(N)
    y[::2]=y2
    y[1::2]=y1
    #LLR
    EbNo = 10 ** (EbNodB / 10)
    No=1/EbNo
    Lc=4*y/No
    f0=1/(1+np.exp(-1*Lc))
    f1=1-f0
    EST_codeword,_,l=ldpc_decode(f0,f1,H,L_MAX)
    #calculate block error rate
    #print(codeword)
    #print(EST_codeword)
    if np.any(codeword!=EST_codeword):#BLOCK error check
      count_err+=1

    count_all+=1

    #calculate bit error rate 
    count_biterr+=np.sum(codeword!=EST_codeword)
    count_bitall+=len(codeword)

    #print("\r"+"l="+str(l)+",count_all="+str(count_all)+",count_err="+str(count_err)+",count_biterr="+str(count_biterr),end="")
  
    #print("\n",EbNodB,"BLER=",count_err/count_all,"BER=",count_biterr/count_bitall)

  return count_err,count_all,count_biterr,count_bitall


# In[ ]:


def monte_carlo_multi(one_codeword_inputs):

  N,  MAX_ERR,  Wc,  Wr,  TX_antenna,  RX_antenna,  modulation_symbol,  EbNodB_start,  EbNodB_end,  L_MAX  =one_codeword_inputs
  
  EbNodB_range=np.arange(EbNodB_start,EbNodB_end)

  #txtファイルへ書き込むための変数の準備

  EbNodB_range=np.arange(EbNodB_start,EbNodB_end)
  BLER=np.zeros(len(EbNodB_range))
  BER=np.zeros(len(EbNodB_range))

  print("N=",N)
  
  for i,EbNodB in enumerate(EbNodB_range):
    
    #ランダムシードの引数もランダムに生成する(tuple に変換)
    seed=np.random.randint(0, 2 ** 32 -1, (MAX_ERR,1))

    inputs=[]


    #argumentの定数部分(MAX_ERRがなくなってtumpleになってます)
    constants=(N,Wc,Wr,TX_antenna,RX_antenna,modulation_symbol,EbNodB,L_MAX)

    #argumentの作成
    
    for j in range(MAX_ERR):
        inputs.append(constants+(seed[j],))

    
    ##multiprocess
    
    pool = multiprocessing.Pool(MAX_ERR) # プロセス数を設定

    result=pool.map(monte_carlo, inputs)  # 並列演算
    #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
    
    pool.close()
    pool.join()
    
    count_err=0
    count_all=0
    count_biterr=0
    count_bitall=0
    
    for j in range(MAX_ERR):
        tmp1,tmp2,tmp3,tmp4=result[j]
        count_err+=tmp1
        count_all+=tmp2
        count_biterr+=tmp3
        count_bitall+=tmp4
        
        #print("result"+str(j)+"=",result[j])
    
    
    
    BLER[i]=count_err/count_all
    BER[i]=count_biterr/count_bitall

    print("\r"+"EbNodB="+str(EbNodB)+",BLER="+str(BLER[i])+",BER="+str(BER[i]),end="")

    if count_biterr/count_bitall<10**-5:
        print("finish")
        break

  
  #結果のEbNodB_range,BLER,BERをtxtファイルへ書き込み

  filename="regular_LDPC_{}_({},{})".format(N,Wc,Wr)

  with open(filename,'w') as f:

      print("#N="+str(N),file=f)
      print("#("+str(Wc)+","+str(Wr)+")",file=f)
      print("#TX_antenna="+str(TX_antenna),file=f)
      print("#RX_antenna="+str(RX_antenna),file=f)
      print("#modulation_symbol="+str(modulation_symbol),file=f)
      print("#MAX_BLERR="+str(MAX_ERR),file=f)
      print("#iteration number="+str(L_MAX),file=f)
      print("#EsNodB,BLER,BER",file=f)      #この説明はプログラムによって変えましょう！！！！！！！
      for i in range(len(EbNodB_range)):
          print(str(EbNodB_range[i]),str(BLER[i]),str(BER[i]),file=f)


# In[ ]:


#定数の設定
Wc=3
Wr=6
TX_antenna=1
RX_antenna=1
modulation_symbol=4
EbNodB_start=-5
EbNodB_end=2
L_MAX=20
inputs=[] #argument

#argumentの定数部分
constants=(Wc,Wr,TX_antenna,RX_antenna,modulation_symbol,EbNodB_start,EbNodB_end,L_MAX)

#argumentの変数部分
N=[1200,2400]
MAX_ERR=[6,6]

#argumentの作成
for i in range(len(N)):
  tmp=(N[i],MAX_ERR[i])
  tmp=tmp+constants
  inputs.append(tmp)

print(inputs)


#別符号長のmultiprocessing

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

if __name__ == "__main__":
    pool = MyPool(1)  # プロセス数を設定
    pool.map(monte_carlo_multi, inputs)  # 並列演算

    pool.close()
    pool.join()

