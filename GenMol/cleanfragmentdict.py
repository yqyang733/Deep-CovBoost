import re
import dbm
import pickle
import struct
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import BRICS
from collections import defaultdict

all_counts = defaultdict(int)
for i in range(100):
    dbname = "chembl_" + str(i).zfill(2) + '.dbm'
    with dbm.open(dbname, flag='r') as db:
        for key,val in db.items():
            all_counts[key]+=struct.unpack('I',val)[0]

# itms = sorted(((y,x) for x,y in all_counts.items()),reverse=True)

# len(itms)
# itms[:10]
# itms[-10:]
# tc,ts = zip(*itms[:10])
# Draw.MolsToGridImage([Chem.MolFromSmiles(x.decode('UTF-8')) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])
# tc,ts = zip(*itms[-10:])
# Draw.MolsToGridImage([Chem.MolFromSmiles(x.decode('UTF-8')) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])

# import re
# expr = re.compile(r'[0-9]+\*')
# clean_counts = defaultdict(int)
# nRejected=0
# for k,v in all_counts.items():
#     k = k.decode('UTF-8')
#     if k.find('*')<0:
#         nRejected +=1
#         continue
#     k = expr.sub('*',k)
#     clean_counts[k]+=v

# clean_itms = sorted([(v,k) for k,v in clean_counts.items()],reverse=True)
# print(len(clean_itms))
# clean_itms[:10]

# tc,ts = zip(*clean_itms[:10])
# Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])
# tc,ts = zip(*clean_itms[-10:])
# Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])

# tmp = [x for x in clean_itms if x[1].count('*')==4]
# tc,ts = zip(*tmp)
# Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])

# tmp = [x for x in clean_itms if x[1].count('*')==3]
# tc,ts = zip(*tmp)
# Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in ts],molsPerRow=4,legends=[str(x) for x in tc])

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
# print(len(clean_itms2),len(clean_itms))

pickle.dump(clean_counts2,open('chemical_words_clean_counts2.pkl','wb+'))

multidict_fragment = defaultdict(list)
for i in clean_counts2.keys():
    try:
        print(i)
        m = Chem.MolFromSmiles(i)
        heanum = m.GetNumHeavyAtoms()
        multidict_fragment[heanum].append(i)
    except:
        print("!!!!!!!!!!!!!"+str(i))
        pass

pickle.dump(multidict_fragment, open('multidict_fragment_chembl.pkl','wb+'))