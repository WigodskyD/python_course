import datetime as dt
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest
import uuid




class Block:
    def __init__(self, seq_id, prev_hash):
        self.__seq_id = seq_id
        self.__prev_hash = prev_hash
        self.__col_names = ['Timestamp', 'Sender', 'Receiver', 'Value', 'TxHash','Timestamp_nice']
        self.__transactions = pd.DataFrame(columns=self.__col_names,index=None)
        self.__status = 'UNCOMMITTED'
        self.__block_hash = None
        self.__merkle_tx_hash = None

    # 5 pts -  Display on a single line the metadata of this block. You'll display the sequence Id, status,
    # block hash, previous block's hash, merkle hash and number of transactions in the block

    def TEMPORARY_set_variables(self):
        temp1 =hashlib.sha256(str("I'm testing this with a fake hash").encode('utf-8')).hexdigest()
        self.__block_hash = temp1
        self.__merkle_tx_hash = temp1
    def TEMPORARY_SHOW_TRANSACTIONS(self):
        print(self.__transactions)
        print(self.__transactions[['Sender','Value']])
    def TEMPORARY_GET_STATUS(self):
        print(self.__status)
    def TEMPORARY_CHECK_TXHASH(self):


        print('UH OH')
        simple_merkle_root=hashlib.sha256(str(pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()).encode('utf-8')).hexdigest()
        thinggg2 = pd.Series(self.__transactions[['TxHash']].transpose().iloc[0]).str.cat()
        print('UH OH')
        print(simple_merkle_root)
        #print(type(pd.Series(thinggg)))
        print('UH OH')
        #print(type(str(self.__transactions[['TxHash']].loc[3])))
    def display_header(self):
        print(str(self.__seq_id) + ', '+ self.__status + ', '+ self.__block_hash + ', '+ self.__prev_hash + ', '+ self.__merkle_tx_hash)

    def add_transaction(self, s, r, v):
        ts =  dt.datetime.now()
        ts_nice = ts.strftime('%m/%d/%Y at %I:%M %p')
        tx_hash = hashlib.sha256(str(str(ts)+s+r+str(v)).encode('utf-8')).hexdigest()# Hash of timestamp, sender, receiver, value
        new_transaction = [ts, s, r, v, tx_hash, ts_nice]
        self.__transactions.loc[len(self.__transactions)] = new_transaction
    def display_transactions(self):
        print(self.__transactions[['Sender', 'Value','Receiver', 'Timestamp_nice']])
    def get_size(self):
        return len(self.__transactions)
    def set_status(self, status):
        self.__status = status
    def set_block_hash(self, hash):
        #self.__prev_hash
        #hash
        #datetime
        #seq_id
        #rootmerkle hash
        pass

#Block hash: The hash of this block is created by the hash of the string concatenation of the previous block's
 #   hash, the chains hash id, current date time, sequence id of the block and the root Merkle hash.
  #  The block hash is generated when a block is full and is committed.

    def get_values(self):
        pass
        #SEE PROF COMMENTS  RETURN TUPLE WITH TXN TIMESTAMP/VALUE PAIR


# Append to the transactions data
a = Block(3,'fghij')
a.TEMPORARY_set_variables()
a.display_header()
a.TEMPORARY_SHOW_TRANSACTIONS()
a.add_transaction('frank','tim',44.0)
a.TEMPORARY_CHECK_TXHASH()
a.add_transaction('laurie','tim',44.0)
a.TEMPORARY_CHECK_TXHASH()
a.add_transaction('sheila','tim',44.0)
a.add_transaction('maureen','tim',44.0)
a.TEMPORARY_SHOW_TRANSACTIONS()
a.display_transactions()
print('noooooo')
print(a.get_size())
a.TEMPORARY_CHECK_TXHASH()


c = dt.datetime.now()
#print(dt.time(c))
print(c.strftime('%m/%d/%Y at %H:%M'))