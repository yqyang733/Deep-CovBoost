import os
import copy
import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn 
import rdkit.Chem as Chem
from torch.utils import data
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.autograd import Variable
from lifelines.utils import concordance_index
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss


ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
MAX_ATOM = 400
MAX_BOND = MAX_ATOM * 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class config:

    def __init__(self):

        self.test = os.path.join("./test.csv")

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
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
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
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms, fbonds = [], [padding] 
        in_bonds,all_bonds = [], [(-1,-1)] 
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            #print(atom.GetSymbol())
            fatoms.append(atom_features(atom))
            in_bonds.append([])
        #print("fatoms: ",fatoms)

        for bond in mol.GetBonds():
            #print(bond)
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() 
            y = a2.GetIdx()
            #print(x,y)

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
        agraph = torch.zeros(n_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
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
    #fatoms, fbonds, agraph, bgraph = [], [], [], [] 
    #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
    Natom, Nbond = fatoms.shape[0], fbonds.shape[0]


    ''' 
    ## completion to make feature size equal. 
    MAX_ATOM = 100
    MAX_BOND = 200
    '''
    atoms_completion_num = MAX_ATOM - fatoms.shape[0]
    bonds_completion_num = MAX_BOND - fbonds.shape[0]
    try:
        assert atoms_completion_num >= 0 and bonds_completion_num >= 0
    except:
        raise Exception("Please increasing MAX_ATOM in line 26 utils.py, for example, MAX_ATOM=600 and reinstall it via 'python setup.py install'. The current setting is for small molecule. ")


    fatoms_dim = fatoms.shape[1]
    fbonds_dim = fbonds.shape[1]
    fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
    fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
    agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, MAX_NB)], 0)
    bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, MAX_NB)], 0)
    # print("atom size", fatoms.shape[0], agraph.shape[0])
    # print("bond size", fbonds.shape[0], bgraph.shape[0])
    shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()], flag

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

class MPNN(nn.Sequential):

    def __init__(self, mpnn_hidden_size, mpnn_depth):
        super(MPNN, self).__init__()
        self.mpnn_hidden_size = mpnn_hidden_size
        self.mpnn_depth = mpnn_depth 

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

    def forward(self, feature):
        '''
            fatoms: (x, 39)
            fbonds: (y, 50)
            agraph: (x, 6)
            bgraph: (y, 6)
        '''
        fatoms, fbonds, agraph, bgraph, N_atoms_bond = feature
        N_atoms_scope = []
                
        N_a, N_b = 0, 0 
        fatoms_lst, fbonds_lst, agraph_lst, bgraph_lst = [],[],[],[]
        for i in range(N_atoms_bond.shape[0]):
            # print(N_atoms_bond[i][0])
            atom_num = int(N_atoms_bond[i][0].item()) 
            bond_num = int(N_atoms_bond[i][1].item()) 

            fatoms_lst.append(fatoms[i,:atom_num,:])
            fbonds_lst.append(fbonds[i,:bond_num,:])
            agraph_lst.append(agraph[i,:atom_num,:] + N_a)
            bgraph_lst.append(bgraph[i,:bond_num,:] + N_b)

            N_atoms_scope.append((N_a, atom_num))
            N_a += atom_num 
            N_b += bond_num 

        fatoms = torch.cat(fatoms_lst, 0)
        fbonds = torch.cat(fbonds_lst, 0)
        agraph = torch.cat(agraph_lst, 0)
        bgraph = torch.cat(bgraph_lst, 0)

        agraph = agraph.long()
        bgraph = bgraph.long()    

        fatoms = create_var(fatoms).to(device)
        fbonds = create_var(fbonds).to(device)
        agraph = create_var(agraph).to(device)
        bgraph = create_var(bgraph).to(device)

        binput = self.W_i(fbonds) #### (y, d1)
        message = F.relu(binput)  #### (y, d1)        

        for i in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message) ### (y,d1) 

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))
        output = [torch.mean(atom_hiddens.narrow(0, sts,leng), 0) for sts,leng in N_atoms_scope]
        output = torch.stack(output, 0)
        return output 

class Classifier(nn.Sequential):
    def __init__(self, model_drug, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.model_drug = model_drug
        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_drug] + self.hidden_dims + [1]
        
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # each encoding
        # print("v_D:", print(len(v_D)))
        v_D = self.model_drug(v_D)
        # print("v_D_1:", v_D.shape)
        # print("v_P:", len(v_P))
        v_P = self.model_drug(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i==(len(self.predictor)-1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f

def mpnn_feature_collate_func(x):
    N_atoms_scope = torch.cat([i[4] for i in x], 0)
    f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
    f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
    agraph_lst, bgraph_lst = [], []
    for j in range(len(x)):
        agraph_lst.append(x[j][2].unsqueeze(0))
        bgraph_lst.append(x[j][3].unsqueeze(0))
    agraph = torch.cat(agraph_lst, 0)
    bgraph = torch.cat(bgraph_lst, 0)
    return [f_a, f_b, agraph, bgraph, N_atoms_scope]

def mpnn_collate_func(x):
    mpnn_feature = [i[0] for i in x]
    mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
    x_remain = [list(i[1:]) for i in x]
    x_remain_collated = default_collate(x_remain)
    return [mpnn_feature] + x_remain_collated

def save_dict(path, obj):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def model_initialize(**config):
    model = DD_Model(**config)
    return model

class data_process_DDI_loader(data.Dataset):

    def __init__(self, list_IDs, labels, df, **config):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df
        self.config = config

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['drug_encoding_1']        
        v_p = self.df.iloc[index]['drug_encoding_2']
        y = self.labels[index]
        return v_d, v_p, y

class DD_Model:

    def __init__(self, **config):
        
        drug_encoding = config['drug_encoding']
        self.model_drug = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])
        self.model = Classifier(self.model_drug, **config)
        self.config = config
        self.drug_encoding = drug_encoding
        self.result_folder = config['result_folder']
        self.device = device
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)  
        self.params = {'batch_size': self.config['batch_size'],
                'shuffle': False,
                'num_workers': self.config['num_workers'],
                'drop_last': False,
                'collate_fn': mpnn_collate_func}          

    def test(self, predict_data):
        y_pred = []
        self.model.eval()
        num_val = int(len(predict_data)) // 1000 + 1
        print("validate:")
        for j in range(num_val):
            print("j: ", j)
            start = j*1000
            end = min(int(len(predict_data)), (j+1)*1000)
            predict_split = predict_data[start: end]
            predict_split = lig_encoding(predict_split)
            predict_generator = data.DataLoader(data_process_DDI_loader(predict_split.index.values, predict_split.Label.values, predict_split, **self.config), **self.params)
            for i, (v_d, v_p, label) in enumerate(predict_generator):
                tmp = torch.flatten(v_p[4], 1)
                v_p[4] = tmp                           
                score = self.model(v_d, v_p)
                scores = torch.squeeze(score).detach().cpu().numpy()
                y_pred = y_pred + scores.flatten().tolist()
        return y_pred

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location = torch.device('cpu'))
        # to support training from multi-gpus data-parallel:
        
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config['binary']

def lig_encoding(data):

    data_encoded = []
    for i in range(len(data)):
        a_data, _ = smiles2mpnnfeature(data[i][0])
        b_data, _ = smiles2mpnnfeature(data[i][1])
        data_encoded.append([a_data, b_data, data[i][2]])

    data_encoded = pd.DataFrame(data_encoded)
    data_encoded.rename(columns={
                        0: 'drug_encoding_1',
                        1: 'drug_encoding_2',
                        2: 'Label'}, 
                        inplace=True)

    return data_encoded

def read_data(in_file):

    rt_data = []
    with open(in_file) as f:
        f1 = f.readlines()
    
    for i in f1:
        line = i.strip().split(",")
        rt_data.append([line[0], line[1], float(line[2])])

    return rt_data

def predict_out(test_data, predict):
    
    rt = open("./out/predict.csv", "w")
    rt.write("Mol_1,Mol_2,predict_ddg\n")
    for i in range(len(test_data)):
        rt.write(test_data[i][0]+","+test_data[i][1]+","+str(predict[i])+"\n")
    rt.close()

def run():

    settings = config()

    test = read_data(settings.test)

    conf = {"drug_encoding": "MPNN", 
            "cls_hidden_dims": [2048, 1024, 512], 
            "result_folder": os.path.join("./out/"), 
            "train_epoch": 60,
            "LR": 0.001, 
            "decay": 0.00001,
            "batch_size": 64,
            "hidden_dim_drug": 128,
            "mpnn_hidden_size": 128,
            "mpnn_depth": 3, 
            "num_workers": 0}

    model = model_initialize(**conf)
    model.load_pretrained()
    print("Starting predict.")
    y_predict = model.test(test)
    predict_out(test, y_predict)
    print("Predict done.")

def main():
    run()

if __name__=="__main__":
    main() 