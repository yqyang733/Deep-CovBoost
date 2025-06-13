import os
import pickle
from rdkit import Chem
from molvs import standardize_smiles

class config:

    def __init__(self):

        self.fragdict = os.path.join("example.pkl")
        self.backbone = list(["CNC(CN1C[C@H](c2c(C1=O)ccc([*])c2)C(Nc(cnc3)c4c3cccc4)=O)=O",])
        self.side = str("Cl[*]")

def get_neiid_bysymbol(mol,marker):
    try:
        for atom in mol.GetAtoms():
            if atom.GetSymbol()==marker:
                neighbors=atom.GetNeighbors()
                if len(neighbors)>1:
                    print ('Cannot process more than one neighbor, will only return one of them')
                atom_nb=neighbors[0]
                return atom_nb.GetIdx()
    except Exception as e:
        print (e)
        return None

def get_id_bysymbol(mol,marker):
    for atom in mol.GetAtoms():
        if atom.GetSymbol()==marker:
            return atom.GetIdx()
        
def combine2frags(mol_a,mol_b,maker_b='Cs',maker_a='Fr'):
    
    merged_mol = Chem.CombineMols(mol_a,mol_b)
    bind_pos_a=get_neiid_bysymbol(merged_mol,maker_a)
    bind_pos_b=get_neiid_bysymbol(merged_mol,maker_b)
    
    ed_merged_mol= Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a,bind_pos_b,order=Chem.rdchem.BondType.SINGLE)
    
    marker_a_idx=get_id_bysymbol(merged_mol,maker_a)
    ed_merged_mol.RemoveAtom(marker_a_idx)
    
    temp_mol = ed_merged_mol.GetMol()
    marker_b_idx=get_id_bysymbol(temp_mol,maker_b)
    ed_merged_mol=Chem.EditableMol(temp_mol)
    ed_merged_mol.RemoveAtom(marker_b_idx)
    final_mol = ed_merged_mol.GetMol()
    
    return final_mol

def replace_node(smi, num, atom):
    idx = 0
    new_smi = ""
    for i in smi:
        if i == "*":
            idx += 1
            if idx == num:
                new_smi += atom
            else:
                new_smi += i
        else:
            new_smi += i
    return new_smi

def de_degree(smi):
    from rdkit import Chem

    try:
        num = smi.count("*")
        while num > 0:
            ref_mol = Chem.MolFromSmiles(smi, False)
            marker_idx = get_id_bysymbol(ref_mol, "*")
            ed_merged_mol = Chem.EditableMol(ref_mol)
            ed_merged_mol.RemoveAtom(marker_idx)
            final_mol = ed_merged_mol.GetMol()
            smi = Chem.MolToSmiles(final_mol)
            num -= 1
        return smi
    except:
        print("de_degree")
        return smi

def assembly_pieces(com, dif):
    
    c_nodes = len(com)
    d_nodes = dif.count("*")
    dif_lst = [dif,]
    dif_lst_1 = []
    all_ = []
    
    while len(com) > 0:
        piece = com.pop()
        frag_1 = replace_node(piece, 1, "[Fr]")       
        for i in dif_lst:
            for j in range(1, d_nodes+1):
                frag_2 = replace_node(i, j, "[Cs]")
                dif_lst_1.append(Chem.MolToSmiles(combine2frags(Chem.MolFromSmiles(frag_1), Chem.MolFromSmiles(frag_2))))
                all_.append(Chem.MolToSmiles(combine2frags(Chem.MolFromSmiles(frag_1), Chem.MolFromSmiles(frag_2))))
        dif_lst = dif_lst_1
        dif_lst_1 = []
        d_nodes -= 1
    all_fin = []
    for ss in all_:
        if ss.count("*") == dif.count("*")-c_nodes:
            all_fin.append(ss)
    all_fin_1 = []
    if dif.count("*")-c_nodes == 0:
        pass
    else:
        for i in all_fin:
            all_fin_1.append(de_degree(i))
    return all_fin_1

def gen_range(num_heavy, degree, word_dict):
    key = [num_heavy-5, num_heavy-4, num_heavy-3, num_heavy-2, num_heavy-1, num_heavy, num_heavy+1, num_heavy+2, num_heavy+3, num_heavy+4, num_heavy+5]
    key = [i for i in key if min(word_dict.keys()) <= i <= max(word_dict.keys())]
    smi_all = []
    for i in key:
        for a in word_dict[i]:
            if a.count("*") >= degree:
                smi_all.append(a)
    return smi_all

def assembly_all(com, repla, word_dict):

    com_1 = []
    for i in com:
        com_1.append(standardize_smiles(i))
    com = com_1

    repla = standardize_smiles(repla)

    degree = len(com)
    m = Chem.MolFromSmiles(repla)
    num_heavy = m.GetNumHeavyAtoms()
    smi_all = gen_range(num_heavy, degree, word_dict)
    gene_smi = []
    for i in smi_all:
        com_1 = com.copy()
        gen_single = assembly_pieces(com_1, i)
        for a in gen_single:
            gene_smi.append(a)
    gene_smi_1 = []
    for i in gene_smi:
        try:
            gene_smi_1.append(standardize_smiles(i))
        except:
            pass
    gene_smi = set(gene_smi_1)

    rt = open("InhouseLib.csv", "w")
    for i in gene_smi:
        rt.write(i+"\n")
    rt.close()

def main():
    
    settings = config()

    fragdict = pickle.load(open(settings.fragdict,'rb+'))
    assembly_all(settings.fragdict, settings.side, fragdict)

if __name__=="__main__":
    main() 