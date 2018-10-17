import datetime as dt
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest
import uuid


class PandasChain:
    # 5 pts - Complete this constructor
    def __init__(self, name):
        self.__name = name.upper()
        self.__chain = []
        self.__id = hashlib.sha256(str(str(uuid.uuid4()) + self.__name + str(dt.datetime.now())).encode('utf-8')).hexdigest()
        self.__seq_id = 0
        self.__prev_hash =  None
        self.__current_block =  Block(self.__seq_id, self.__prev_hash)
        print(self.__name, 'PandasChain created with ID', self.__id, 'chain started.')

    def display_chain(self):
        index = 1
        for bloc in self.__chain:
            print('.........................')
            print('Block Number ' + str(index))
            print('-------------------------')
            bloc.display_transactions()
            index +=1
        print('.........................')
        print('Block Number ' + str(len(self.__chain)+1))
        print('-------------------------')
        self.__current_block.display_transactions()

    def add_transaction(self, s, r, v):
        if self.__current_block.get_size() >= 10:
            self.__commit_block(self.__current_block)
        self.__current_block.add_transaction(s, r, v)

    def __commit_block(self, block):
        self.__current_block.set_status()
        self.__current_block.get_simple_merkle_root()
        block_hash =  self.__current_block.set_block_hash(self.__prev_hash)
        self.__chain.append(self.__current_block)
        self.__prev_hash = block_hash
        self.__chain[self.__seq_id] = self.__current_block
        self.__seq_id += 1
        self.__current_block = Block(self.__seq_id, self.__prev_hash)
        print('Block committed')

    def display_block_headers(self):
        for bloc in self.__chain:
            bloc.display_header()
        self.__current_block.display_header()

    def get_number_of_blocks(self):
        return len(self.__chain) + 1

    # 10 pts - Returns all of the values (Pandas coins transferred) of all transactions from every block as a single list
    def get_values(self):
        value_return = []
        for bloc in self.__chain:
            value_return = value_return + bloc.get_values()
        value_return = value_return + self.__current_block.get_values()
        print(len(value_return))
        print(type(value_return[0][0]))
        return value_return


class Block:
    def __init__(self, seq_id, prev_hash):
        self.__seq_id = seq_id
        self.__prev_hash = prev_hash
        self.__col_names = ['Timestamp', 'Sender', 'Receiver', 'Value', 'TxHash','Transaction_time']
        self.__transactions = pd.DataFrame(columns=self.__col_names,index=None)
        self.__status = 'UNCOMMITTED'
        self.__block_hash = None
        self.__merkle_tx_hash = None

    def display_header(self):
        if self.__prev_hash == None:
            hash_entry = 'This is the first block. So there is no previous hash to report.'
        else:
            hash_entry = self.__prev_hash
        block_hash_entry = str(self.__block_hash).ljust(64)
        merkle_root_entry = str(self.__merkle_tx_hash).ljust(64)
        status_entry = str(self.__status).ljust(11)
        print(str(self.__seq_id+1) + ', '+ status_entry + ', '+ block_hash_entry + ', '+ hash_entry + ', '+ merkle_root_entry + ', ' + str(self.get_size()))

    def add_transaction(self, s, r, v):
        ts =  dt.datetime.now()
        s = s.upper()
        r = r.upper()
        ts_nice = ts.strftime('%m/%d/%Y at %I:%M %p')
        tx_hash = hashlib.sha256(str(str(ts)+s+r+str(v)).encode('utf-8')).hexdigest()# Hash of timestamp, sender, receiver, value
        new_transaction = [ts, s, r, v, tx_hash, ts_nice]
        self.__transactions.loc[len(self.__transactions)] = new_transaction
    def display_transactions(self):
        #print(self.__transactions[['Sender', 'Value','Receiver', 'Transaction_time']])
        display = self.__transactions.to_string(index=False,columns=['Sender', 'Value','Receiver', 'Transaction_time'])
        print(display)
    def get_size(self):
        return len(self.__transactions)
    def set_status(self):
        self.__status = 'COMMITTED' # I removed the parameter from this function because blocks aren't ever decommitted in our use.
    def set_block_hash(self, hash):
        if hash == None:
            hash = ''
        ts = dt.datetime.now()
        self.__block_hash = hashlib.sha256(str(str(self.__prev_hash)+str(hash)+str(ts)+str(self.__seq_id)+self.__merkle_tx_hash).encode('utf-8')).hexdigest()  # Hash of timestamp, sender, receiver, value
        return self.__block_hash

    def get_simple_merkle_root(self):
        self.__merkle_tx_hash = hashlib.sha256(str(pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()).encode('utf-8')).hexdigest()
        return self.__merkle_tx_hash

    def get_values(self):
        values_set  = self.__transactions[['Timestamp','Value']]
        tuples = [tuple(a) for a in values_set.values]
        return tuples

        # SEE PROF COMMENTS  RETURN TUPLE WITH TXN TIMESTAMP/VALUE PAIR
class TestAssignment4(unittest.TestCase):
    def test_chain(self):
        block = Block(1,"test")
        self.assertEqual(block.get_size(),0)
        block.add_transaction("Bob","Alice",50)
        self.assertEqual(block.get_size(),1)
        pandas_chain = PandasChain('testnet')
        self.assertEqual(pandas_chain.get_number_of_blocks(),1)
        pandas_chain.add_transaction("Bob","Alice",50)
        pandas_chain.add_transaction("Bob","Alice",51)
        pandas_chain.add_transaction("Bob","Alice",52)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        self.assertEqual(pandas_chain.get_number_of_blocks(),2)
        pandas_chain.add_transaction("Bob","Alice",50)
        pandas_chain.add_transaction("Bob","Alice",51)
        pandas_chain.add_transaction("Bob","Alice",52)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        pandas_chain.add_transaction("Bob","Alice",53)
        self.assertEqual(pandas_chain.get_number_of_blocks(),3)
        x,y = zip(*pandas_chain.get_values())
        plt.plot(x,y)
        plt.show()

if __name__ == '__main__':
    unittest.main()



fakem= PandasChain('FakeMoney')
print(fakem.get_number_of_blocks())
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','44.1')
fakem.add_transaction('FrAnK','oLiVeR','44.1')
fakem.add_transaction('FrAnK','oLiVeR','44.1')
fakem.add_transaction('FrAnK','oLiVeR','44.1')
fakem.add_transaction('FrAnK','oLiVeR','44.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','66.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
fakem.add_transaction('FrAnK','oLiVeR','22.1')
print('showing chain')


#fakem.display_block_headers()
g = None
g = '  '
print ('def'+ g + 'abc')
h = [[1,'a'],[2,'b'],[3,'c'],[4,'d']]
print(h[-1])

fakem.display_block_headers()
fakem.display_chain()
print(fakem.get_number_of_blocks())
print('dddddddddddddddddddddddddddddddddd')
print(fakem.get_values())
