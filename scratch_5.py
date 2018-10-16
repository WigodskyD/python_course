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
    def TEMPORARY_SHOW_CHAIN(self):
        print(self.__chain)
    # 5 pts - This method should loop through all committed and uncommitted blocks and display all transactions in them
    def TEMPORARY_HASHER(self):
        print('TEMPORARY you are here!')
        self.__current_block.TEMPORARY_MERKLE()
        self.__current_block.get_simple_merkle_root()
        self.__current_block.TEMPORARY_MERKLE()
        block_hash = self.__current_block.set_block_hash(self.__prev_hash)
        print('block hash is:')
        print(block_hash)
        print("it's set to:")
        self.__current_block.TEMPORARY_BLOCK_HASH()
        self.__chain.append(block_hash)
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
        self.__current_block.TEMPORARY_set_variables()
        for bloc in self.__chain:
            bloc.display_header()
        self.__current_block.display_header()

    # 5 pts - return int total number of blocks in this chain (committed and uncommitted blocks combined)
    def get_number_of_blocks(self):
        return len(self.__chain) +1

    # 10 pts - Returns all of the values (Pandas coins transferred) of all transactions from every block as a single list
    def get_values(self):
        pass

    def TEMPORARY_get_length(self):
        print(self.__current_block.display_transactions())
class Block:
    def __init__(self, seq_id, prev_hash):
        self.__seq_id = seq_id
        self.__prev_hash = prev_hash
        self.__col_names = ['Timestamp', 'Sender', 'Receiver', 'Value', 'TxHash','Transaction_time']
        self.__transactions = pd.DataFrame(columns=self.__col_names,index=None)
        self.__status = 'UNCOMMITTED'
        self.__block_hash = None
        self.__merkle_tx_hash = None

    # 5 pts -  Display on a single line the metadata of this block. You'll display the sequence Id, status,
    # block hash, previous block's hash, merkle hash and number of transactions in the block
    def TEMPORARY_BLOCK_HASH(self):
        print(self.__block_hash)
    def TEMPORARY_set_variables(self):
        temp1 =hashlib.sha256(str("I'm testing this with a fake hash").encode('utf-8')).hexdigest()
        self.__block_hash = temp1
        self.__merkle_tx_hash = temp1
        self.__prev_hash = temp1
    def TEMPORARY_SHOW_TRANSACTIONS(self):
        print(self.__transactions)
        print(self.__transactions[['Sender','Value']])
    def TEMPORARY_GET_STATUS(self):
        print(self.__status)
    def TEMPORARY_MERKLE(self):
        print(self.__merkle_tx_hash)
    def TEMPORARY_CHECK_TXHASH(self):
        print('UH OH')
        simple_merkle_root=hashlib.sha256(str(pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()).encode('utf-8')).hexdigest()
        #thinggg2 = pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()
        print('UH OH')
        print(simple_merkle_root)
        #print(type(pd.Series(thinggg)))
        print(str(pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()))
        #print(type(str(self.__transactions[['TxHash']].loc[3])))
    def display_header(self):
        if self.__prev_hash == None:
            hash_entry = 'This is the first block. So there is no previous hash to report.'
        else:
            hash_entry = self.__prev_hash
        print(str(self.__seq_id+1) + ', '+ self.__status + ', '+ self.__block_hash + ', '+ hash_entry + ', '+ self.__merkle_tx_hash + ', ' + str(self.get_size()))

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
        self.__status = 'COMMITTED  ' # I removed the parameter from this function because blocks aren't ever decommitted in our use.  Two spaces are at the end of the status
                                      #  to make it line up when display block headers is called.  If the program needs to use the status, the spaces may nedd to be removed.
    def set_block_hash(self, hash):
        if hash == None:
            hash = ''
        ts = dt.datetime.now()
        self.__block_hash = hashlib.sha256(str(str(self.__prev_hash)+str(hash)+str(ts)+str(self.__seq_id)+self.__merkle_tx_hash).encode('utf-8')).hexdigest()  # Hash of timestamp, sender, receiver, value
        return self.__block_hash
#Block hash: The hash of this block is created by the hash of the string concatenation of the previous block's
 #   hash, the chains hash id, current date time, sequence id of the block and the root Merkle hash.
  #  The block hash is generated when a block is full and is committed.
    def get_simple_merkle_root(self):
        self.__merkle_tx_hash = hashlib.sha256(str(pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()).encode('utf-8')).hexdigest()
        return self.__merkle_tx_hash

    def get_values(self):
        pass
        # SEE PROF COMMENTS  RETURN TUPLE WITH TXN TIMESTAMP/VALUE PAIR


# Append to the transactions data
a = Block(3,'fghij')
a.TEMPORARY_set_variables()
a.TEMPORARY_SHOW_TRANSACTIONS()
a.add_transaction('frank','tim',44.0)

a.add_transaction('laurie','tim',28.5)
a.add_transaction('tim','sheila',44.0)
a.add_transaction('sheila','tim',32.73)
a.add_transaction('maureen','tim',44.0)
a.TEMPORARY_SHOW_TRANSACTIONS()
a.display_transactions()
print('noooooo')
print(a.get_size())
a.set_status()
a.get_simple_merkle_root()

print(a.get_size())
print(a.get_simple_merkle_root())
#a.display_header()
a.set_block_hash('hash')
a.display_header()
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
print('showing chain')
fakem.TEMPORARY_SHOW_CHAIN()
#fakem.TEMPORARY_HASHER()
fakem.TEMPORARY_SHOW_CHAIN()
#fakem.display_block_headers()
g = None
g = '  '
print ('def'+ g + 'abc')
h = [[1,'a'],[2,'b'],[3,'c'],[4,'d']]
print(h[-1])
fakem.TEMPORARY_SHOW_CHAIN()
fakem.display_block_headers()
fakem.display_chain()
print(fakem.get_number_of_blocks())
