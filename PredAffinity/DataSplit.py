import os
import math
import torch
import random
import rdkit.Chem as Chem

elelst = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
atmdim = len(elelst) + 16
bonddim = 11
nbmax = 6
atmmax = 400
bondmax = atmmax * 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class config:

    def __init__(self):

        self.data = os.path.join("example.csv")
        self.ratio = [0.9, 0.1]

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), elelst) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def smiles2mpnnfeature(smiles):
    flag = False
    try: 
        padding = torch.zeros(atmdim + bonddim)
        fatoms, fbonds = [], [padding] 
        in_bonds,all_bonds = [], [(-1,-1)] 
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() 
            y = a2.GetIdx()

            b = len(all_bonds)
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0) 
        fbonds = torch.stack(fbonds, 0) 
        agraph = torch.zeros(n_atoms,nbmax).long()
        bgraph = torch.zeros(total_bonds,nbmax).long()
        for a in range(n_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2
        # print("fatoms: ",fatoms)
        # print("fbonds: ",fbonds)
        # print("agraph: ",agraph)
        # print("bgraph: ",bgraph)
        flag = True

    except: 
        print('Molecules not found and change to zero vectors..')
        fatoms = torch.zeros(0,39)
        fbonds = torch.zeros(0,50)
        agraph = torch.zeros(0,6)
        bgraph = torch.zeros(0,6)
    #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
    Natom, Nbond = fatoms.shape[0], fbonds.shape[0]

    atoms_completion_num = atmmax - fatoms.shape[0]
    bonds_completion_num = bondmax - fbonds.shape[0]
    try:
        assert atoms_completion_num >= 0 and bonds_completion_num >= 0
    except:
        raise Exception("Please increasing atmmax.")

    fatoms_dim = fatoms.shape[1]
    fbonds_dim = fbonds.shape[1]
    fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
    fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
    agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, nbmax)], 0)
    bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, nbmax)], 0)
    # print("atom size", fatoms.shape[0], agraph.shape[0])
    # print("bond size", fbonds.shape[0], bgraph.shape[0])
    shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()], flag

def judge_smiles(infile):

    with open(infile) as f:
        f.readline()
        f1 = f.readlines()
    data = []
    for i in f1:
        _, flag = smiles2mpnnfeature(i.strip().split(",")[0])
        if flag:
            data.append(i)

    return data

def Cross_validation(idx, num):
    
    fold_rt = []
    num_individual = len(idx) // num + 1
    for i in range(num):
        start_idx = i*num_individual 
        end_idx = min((i+1)*num_individual, len(idx)) 
        test = idx[start_idx:end_idx]
        train = [ii for ii in idx if ii not in test]
        fold_rt.append([train, test])

    return fold_rt

def cross_data(lst):

    data_rt = []
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            a_smi = lst[i].split(",")[0]
            a_label = float(lst[i].split(",")[1].strip())
            b_smi = lst[j].split(",")[0]
            b_label = float(lst[j].split(",")[1].strip())
            ddg = 0.596*(math.log(b_label / a_label))
            if ddg > 5:
                data_rt.append((a_smi, b_smi, 5))
            elif ddg < -5:
                data_rt.append((a_smi, b_smi, -5))
            else:
                data_rt.append((a_smi, b_smi, ddg))

    return data_rt

def cross_test_train(train_lst, test_lst):

    data_rt = []
    for i in train_lst:
        for j in test_lst:
            a_smi = i.split(",")[0]
            a_label = float(i.split(",")[1].strip())
            b_smi = j.split(",")[0]
            b_label = float(j.split(",")[1].strip())
            ddg = 0.596*(math.log(b_label / a_label))
            if ddg > 5:
                data_rt.append((a_smi, b_smi, 5))
            elif ddg < -5:
                data_rt.append((a_smi, b_smi, -5))
            else:
                data_rt.append((a_smi, b_smi, ddg))

    return data_rt

def data_process(in_file, ratio):

    random.shuffle(in_file)
    origin_test = in_file[:int(len(in_file)*ratio[-1])]
    origin_train_valid = in_file[int(len(in_file)*ratio[-1]):]
    test_data = cross_test_train(origin_train_valid, origin_test)
    train_valid_data = cross_data(origin_train_valid)

    return train_valid_data, test_data

def run():

    settings = config()
    data = judge_smiles(settings.data)
    train_valid_data, test_data = data_process(data, settings.ratio)
    
    fold = Cross_validation(range(len(train_valid_data)), 5)
    for ii in range(len(fold)):
        if not os.path.exists(os.path.join(".", "Fold"+str(ii))):
            os.mkdir(os.path.join(".", "Fold"+str(ii)))
        train_csv = open(os.path.join(".", "Fold"+str(ii), "train.csv"), "w")
        valid_csv = open(os.path.join(".", "Fold"+str(ii), "valid.csv"), "w")
        test_csv = open(os.path.join(".", "Fold"+str(ii), "test.csv"), "w")
        train_data = [train_valid_data[i] for i in fold[ii][0]] 
        valid_data = [train_valid_data[i] for i in fold[ii][1]] 
        for i in train_data:
            train_csv.write(i[0] + "," + i[1] + "," + str(i[2]) + "\n")
        for i in valid_data:
            valid_csv.write(i[0] + "," + i[1] + "," + str(i[2]) + "\n")
        for i in test_data:
            test_csv.write(i[0] + "," + i[1] + "," + str(i[2]) + "\n")
        train_csv.close()
        valid_csv.close()
        test_csv.close()

def main():
    run()

if __name__=="__main__":
    main() 