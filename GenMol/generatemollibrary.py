import pickle
from rdkit import Chem
from molvs import standardize_smiles

#函数一，获取marker邻居原子的index, 注意marker只能是一个单键连接核上的原子，否则邻居会多于一个
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
#函数二，获取marker原子的index
def get_id_bysymbol(mol,marker):
    for atom in mol.GetAtoms():
        if atom.GetSymbol()==marker:
            return atom.GetIdx()
        
def combine2frags(mol_a,mol_b,maker_b='Cs',maker_a='Fr'):
    #将两个待连接分子置于同一个对象中
    merged_mol = Chem.CombineMols(mol_a,mol_b)
    bind_pos_a=get_neiid_bysymbol(merged_mol,maker_a)
    bind_pos_b=get_neiid_bysymbol(merged_mol,maker_b)
    #转换成可编辑分子，在两个待连接位点之间加入单键连接，特殊情形需要其他键类型的情况较少，需要时再修改
    ed_merged_mol= Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a,bind_pos_b,order=Chem.rdchem.BondType.SINGLE)
    #将图中多余的marker原子逐个移除，先移除marker a
    marker_a_idx=get_id_bysymbol(merged_mol,maker_a)
    ed_merged_mol.RemoveAtom(marker_a_idx)
    #marker a移除后原子序号变化了，所以又转换为普通分子后再次编辑，移除marker b
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
        # print(smi)
        return smi

def assembly_pieces(com, dif):
    # print(com)
    c_nodes = len(com)
    d_nodes = dif.count("*")
    dif_lst = [dif,]
    dif_lst_1 = []
    all_ = []
    while len(com) > 0:
        piece = com.pop()
        frag_1 = replace_node(piece, 1, "[Fr]")       
        # for a in dif_lst:
        #     if a.count("*") != d_nodes:
        #         dif_lst.remove(a)
        for i in dif_lst:
            for j in range(1, d_nodes+1):
                frag_2 = replace_node(i, j, "[Cs]")
                # print("frag_1", frag_1)
                # print("frag_2", frag_2)
                dif_lst_1.append(Chem.MolToSmiles(combine2frags(Chem.MolFromSmiles(frag_1), Chem.MolFromSmiles(frag_2))))
                all_.append(Chem.MolToSmiles(combine2frags(Chem.MolFromSmiles(frag_1), Chem.MolFromSmiles(frag_2))))
        dif_lst = dif_lst_1
        dif_lst_1 = []
        d_nodes -= 1
    # print(all_)
    all_fin = []
    for ss in all_:
        if ss.count("*") == dif.count("*")-c_nodes:
            all_fin.append(ss)
        # if ss.count("*") == 1:
            # all_.remove(ss)
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
    # print(smi_all)
    gene_smi = []
    for i in smi_all:
        # print(i)
        com_1 = com.copy()
        gen_single = assembly_pieces(com_1, i)
        # print(gen_single)
        for a in gen_single:
            gene_smi.append(a)
    print(len(gene_smi))
    gene_smi_1 = []
    for i in gene_smi:
        try:
            gene_smi_1.append(standardize_smiles(i))
        except:
            pass
    print(len(gene_smi_1))
    gene_smi = set(gene_smi_1)
    # print(gene_smi)
    print(len(gene_smi))

    rt = open("opt6.csv", "w")
    for i in gene_smi:
        rt.write(i+"\n")
    rt.close()

mul = pickle.load(open('../Chembl_fragment/multidict_fragment_chembl.pkl','rb+'))
assembly_all(["CNC(CN1C[C@H](c2c(C1=O)ccc([*])c2)C(Nc(cnc3)c4c3cccc4)=O)=O"], "Cl[*]", mul)