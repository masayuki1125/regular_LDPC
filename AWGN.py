import numpy as np
import time


#QAMは工事中

class _AWGN():
    def __init__(self):
        '''
        input constant about channel
        -----------
        M:変調多値数
        TX_antenna:送信側アンテナ数
        RX_antenna:受信側アンテナ数
        '''
        super().__init__()

        self.M=2
        self.M_bits = int(np.log2(self.M))
        self.TX_antenna=1
        self.RX_antenna=1

    '''
    @staticmethod
    def bi2de(binary):
        bin_temp = 0
        bin_res = np.zeros(len(binary), dtype=int)
        for i in range(len(binary)):
            for j in range(len(binary[i])):
                bin_temp = bin_temp + binary[i][j] * (2 ** j)
            bin_res[i] = bin_temp
            bin_temp = 0
        return bin_res

    def gray_code(self):
        for k in range(2 ** self.M_bits):
            yield k ^ (k >> 1)
    '''


    def generate_QAM(self,information):
        if self.M_bits==1:
            constellation=2*information-1
        


        elif self.M_bits==2:
            for i in range(len(information)//self.M_bits):
                constellation=np.array([],dtype=complex)
                tmp=(2*information[2*i]-1)+1j*(2*information[2*i+1]-1)
                constellation=np.append(constellation,tmp)
        return constellation

    def add_AWGN(self,constellation,No):

        # AWGN雑音の生成
        noise = np.random.normal(0, np.sqrt(No / 2), (len(constellation))) \
                + 1j * np.random.normal(0, np.sqrt(No / 2), (len(constellation)))

        # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
        RX_constellation = constellation + noise

        # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
        #print(np.dot(noise[0, :], np.conj(noise[0, :]))/bit_num)

        return RX_constellation

    def demodulate(self,RX_constellation,No):
        if self.M_bits==1:
            y=RX_constellation.real
        
        elif self.M_bits==2:
            y=np.zeros(K)
            y[::2]=RX_constellation.real
            y[1::2]=RX_constellation.imag
            #y = np.array([])
            #for i in range(len(RX_constellation)):
                #tmp=[RX_constellation[i].real,RX_constellation[i].imag]
                #y=np.append(y,tmp)
        #print(y)
        Lc=4*y/No
        return Lc

    def generate_LLR(self,information,EbNodB):
        '''
        information:1D sequence
        EbNodB:EsNodB
        --------
        output:LLR of channel output
        '''
        # Additive Gaussian White Noiseの生成する際のパラメータ設定
        EbNo = 10 ** (EbNodB / 10)
        No=1/EbNo #Eb=1(fixed)

        #tmp=self.bi2de(np.reshape(information, (len(information)//self.M_bits, self.M_bits), order='F'))
        constellation=self.generate_QAM(information)
        RX_constellation=self.add_AWGN(constellation,No)
        Lc=self.demodulate(RX_constellation,No)
        #print(Lc)
        return Lc

if __name__=="__main__":
  ch=_AWGN()
  time_start = time.time()  
  information=np.zeros(100)
  res=ch.generate_LLR(information,100)
  res=np.sign(res)
  EST_information=(res+1)//2
  print(EST_information)
  #print(information)
  print(np.sum(information!=EST_information))
  #print(ch.channel(information,100))
  
  K=100
  MAX_ERR=100
  
  for EbNodB in range(0,10):
    print(EbNodB)
    count_err=0
    count_all=0
    while count_err<MAX_ERR:
        information=np.random.randint(0,2,K)
        res=ch.generate_LLR(information,EbNodB)
        res=np.sign(res)
        EST_information=(res+1)//2
        #print(EST_information)
        #print(information)
        count_err+=np.sum(information!=EST_information)
        #print(count_err)
        count_all+=K

    print(count_err/count_all)
    
  

  time_end = time.time()  
  time_cost = time_end - time_start  
  print('time cost:', time_cost, 's')
  
  