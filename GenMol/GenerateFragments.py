import dbm
import struct
from rdkit import Chem
from rdkit.Chem import BRICS
from multiprocessing import Pool

def get_fragment(in_file):
    
    outfname = in_file + '.dbm'
    outlog = in_file + '.log'
    db =  dbm.open(outfname, flag='n')
    rt = open(outlog, "w")
    idx = 0
    with open(in_file) as f:
        for smi in f:
            idx += 1
            if len(smi.split(",")) == 3:
                m = Chem.MolFromSmiles(smi.split(",")[1])
                if m is None or m.GetNumHeavyAtoms()>60: continue
                s = BRICS.BRICSDecompose(m)
                for entry in s:
                    cnt = struct.unpack('I', db.get(entry,b'\0\0\0\0'))[0]+1
                    db[entry] = struct.pack('I',cnt)
                if idx%1000 == 0:
                    rt.write(str(idx)+"\n")
                    rt.flush()
            else:
                pass
    db.close()

def main():
    
    get_fragment("example")

if __name__=="__main__":
    main() 
