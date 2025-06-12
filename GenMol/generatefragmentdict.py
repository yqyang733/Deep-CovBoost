
def get_fragment_zinc(in_file):
    from rdkit import Chem
    from rdkit.Chem import BRICS
    import dbm
    import struct

    outfname = in_file + '.dbm'
    outlog = in_file + '.log'
    db =  dbm.open(outfname, flag='n')
    rt = open(outlog, "w")
    idx = 0
    with open(in_file) as f:
        for smi in f:
            idx += 1
            if len(smi.split()) == 2:
                m = Chem.MolFromSmiles(smi.split()[0])
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

def get_smis():
    all_f = []
    for i in range(100):
        all_f.append("chembl_"+str(i).zfill(2))
    return all_f

def main():

    from multiprocessing import Pool
    
    all_f = get_smis()
    with Pool(128) as p:
        p.map(get_fragment_zinc, all_f)

if __name__=="__main__":
    main() 
