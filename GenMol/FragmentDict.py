import re
import dbm
import pickle
import struct
from rdkit import Chem
from collections import defaultdict

def CleanFrags():

    all_counts = defaultdict(int)
    dbname = "example.dbm"
    with dbm.open(dbname, flag='r') as db:
        for key,val in db.items():
            all_counts[key]+=struct.unpack('I',val)[0]

    expr = re.compile(r'[0-9]+\*')
    clean_counts2 = defaultdict(int)
    nRejected=0
    for k,v in all_counts.items():
        k = k.decode('UTF-8')
        if k.find('*')<0:
            nRejected +=1
            continue
        k = Chem.MolToSmiles(Chem.MolFromSmiles(expr.sub('*',k)),True)
        clean_counts2[k]+=v
    clean_itms2 = sorted([(v,k) for k,v in clean_counts2.items()],reverse=True)
    pickle.dump(clean_counts2,open('example_counts.pkl','wb+'))

    multidict_fragment = defaultdict(list)
    for i in clean_counts2.keys():
        try:
            m = Chem.MolFromSmiles(i)
            heanum = m.GetNumHeavyAtoms()
            multidict_fragment[heanum].append(i)
        except:
            print("error: "+str(i))
            pass
    pickle.dump(multidict_fragment, open('example.pkl','wb+'))

def main():
    
    CleanFrags()

if __name__=="__main__":
    main() 