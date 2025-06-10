import os
import math
import time
import shutil

class config:

    def __init__(self):

        self.pdb_a_path = os.path.join(".","state_a.pdb")
        self.pdb_b_path = os.path.join(".","state_b.pdb")
        self.rtf_a_path = os.path.join(".","state_a.rtf")
        self.rtf_b_path = os.path.join(".","state_b.rtf")
        self.prm_a_path = os.path.join(".","state_a.prm")
        self.prm_b_path = os.path.join(".","state_b.prm")
        self.receptor_pdb = os.path.join(".","receptor.pdb")
        self.receptor_psf = os.path.join(".","receptor.psf")
        self.vmd_path = os.path.join("vmd-1.9.4a55-installed/bin/vmd")
        self.ff_path = os.path.join(".","toppar")
        self.submit = int(2)
        self.duplicate = int(3)
        self.common_pairs = int(0)

class Lig_pdb:

    def __init__(self, lig_pdb_file):

        self.atomname_property = dict()   # key:atomname   value: resname, chain, resid, coordinate, bfactor, occupy, segname
        self.atomname_coord = list()   # 0:atomname   1:coordinate
        self.segname = set() # all segment names

        with open(lig_pdb_file) as f:
            f1 = f.readlines()
        for i in f1:
            if i.startswith("ATOM"):
                self.atomname_property[i[12:16].strip()] = i[16:]
                self.atomname_coord.append([i[12:16].strip(), [i[30:38].strip(), i[38:46].strip(), i[46:54].strip()]])
                self.segname.add(i[66:76].strip())

class Lig_rtf:

    def __init__(self, lig_rtf_file):

        self.atomname_type = dict()   # key:atomname   value:atomtypeq
        self.atomname_charge = dict()   # key:atomname   value:charge
        self.atomname_penalty = dict()   # key:atomname   value:penalty
        self.bonds = list()   # 0:atom1   1:atom2
        self.impr = list()   # 0:atom1   1:atom2   2:atom3   3:atom4
        self.atomnames = list() # atomname
        self.charge = ""

        with open(lig_rtf_file) as f:
            f1 = f.readlines()
        for i in f1:
            if i.startswith("RESI"):
                self.charge = i.split("!")[0].rsplit()[-1]
            elif i.startswith("ATOM"):
                self.atomname_type[i.split("!")[0].split()[1]] = i.split("!")[0].split()[2]
                self.atomname_charge[i.split("!")[0].split()[1]] = i.split("!")[0].split()[3]
                self.atomname_penalty[i.split("!")[0].split()[1]] = i.split("!")[1].split()[0]
                self.atomnames.append(i.split("!")[0].split()[1])
            elif i.startswith("BOND"):
                self.bonds.append([i.split()[1], i.split()[2]])
            elif i.startswith("IMPR"):
                self.impr.append([i.split()[1], i.split()[2], i.split()[3], i.split()[4]])

class Lig_prm:

    def __init__(self, lig_prm_file):
        self.title = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.end = []

        identifier = {"BONDS":1, "ANGLES":2, "DIHEDRALS":3, "IMPROPERS":4, "END":5}
        index = 0
        flag = 0

        with open(lig_prm_file) as f:
            lines = f.readlines()
        
        while index < len(lines):
            line = lines[index]
            
            if line.replace("\n", "") in identifier.keys():
                flag = identifier[line.replace("\n", "")]
            
            if flag == 0:
                self.title.append(line)
            elif flag == 1:
                if line.startswith("BONDS"):
                    pass
                elif line == "\n":
                    pass
                else:
                    bonds = line.split("!")[0]
                    self.bonds.append([(bonds[:7].strip(),bonds[7:14].strip()),(bonds[14:21].strip(),bonds[21:32].strip()),line.split("!")[1]])
            elif flag == 2:
                if line.startswith("ANGLES"):
                    pass
                elif line == "\n":
                    pass
                else:
                    angles = line.split("!")[0]
                    self.angles.append([(angles[:7].strip(),angles[7:14].strip(),angles[14:21].strip()),(angles[21:28],angles[28:38].strip()),line.split("!")[1]])
            elif flag == 3:
                if line.startswith("DIHEDRALS"):
                    pass
                elif line == "\n":
                    pass
                else:
                    dihedrals = line.split("!")[0]
                    self.dihedrals.append([(dihedrals[:7].strip(),dihedrals[7:14].strip(),dihedrals[14:21].strip(),dihedrals[21:28]),(dihedrals[28:38].strip(),dihedrals[38:41].strip(),dihedrals[41:50].strip()),line.split("!")[1]])            
            elif flag == 4:
                if line.startswith("IMPROPERS"):
                    pass
                elif line == "\n":
                    pass
                else:
                    impropers = line.split("!")[0]
                    self.impropers.append([(impropers[:7].strip(),impropers[7:14].strip(),impropers[14:21].strip(),impropers[21:28]),(impropers[28:38].strip(),impropers[38:41].strip(),impropers[41:50].strip()),line.split("!")[1]])
            elif flag == 5:
                self.end.append(line)

            index += 1

def distance_x_y_z(coord_a, coord_b):
    return math.sqrt((float(coord_b[0])-float(coord_a[0]))**2 + (float(coord_b[1])-float(coord_a[1]))**2 + (float(coord_b[2])-float(coord_a[2]))**2)

def find_common_atom_pairs(pdb_a, pdb_b):
    
    pdb_a = Lig_pdb(pdb_a)
    pdb_b = Lig_pdb(pdb_b)
    
    common_pairs = []
    for i in pdb_a.atomname_coord:
        dis = 10
        atomname_ref = i[0]
        coord_ref = i[1]
        for j in range(len(pdb_b.atomname_coord)):
            coord = pdb_b.atomname_coord[j][1]    
            if distance_x_y_z(coord_ref, coord) < dis:
                dis = distance_x_y_z(coord_ref, coord)
                index = j
        if dis < 0.5:
            common_pairs.append([atomname_ref, pdb_b.atomname_coord[index][0]])
    
    return common_pairs   # common atom pairs of two pdbs. 0:pdb_1 atomname   1:pdb_2 atomname

def remove_difftypes(common_pairs, rtf_a, rtf_b):

    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    remove_difftypes = []
    for i in common_pairs:
        if rtf_a.atomname_type[i[0]] == rtf_b.atomname_type[i[1]]:
            remove_difftypes.append(i)
    
    return remove_difftypes

def remove_diffcharges(common_pairs, rtf_a, rtf_b):
    
    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    remove_diffcharges = []
    for i in common_pairs:
        if abs(float(rtf_a.atomname_charge[i[0]]) - float(rtf_b.atomname_charge[i[1]])) <= 0.05:
            remove_diffcharges.append(i)
    
    return remove_diffcharges

def remove_diff_h(common_pairs, rtf_a, rtf_b):

    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    common_a = [i[0] for i in common_pairs]
    common_b = [i[1] for i in common_pairs]

    other_a_heavy = []
    for i in rtf_a.atomnames:
        if (i not in common_a) and (not i.startswith("H")):
            other_a_heavy.append(i)

    other_b_heavy = []
    for i in rtf_b.atomnames:
        if (i not in common_b) and (not i.startswith("H")):
            other_b_heavy.append(i)

    remove_h1 = []
    for i in rtf_a.bonds:
        if i[0] in other_a_heavy and i[1].startswith("H"):
            remove_h1.append(i[1])
        if i[1] in other_a_heavy and i[0].startswith("H"):
            remove_h1.append(i[0])
    common_pairs_deh1 = []
    for i in common_pairs:
        if i[0] not in remove_h1:
            common_pairs_deh1.append(i)

    remove_h2 = []
    for i in rtf_b.bonds:
        if i[0] in other_b_heavy and i[1].startswith("H"):
            remove_h2.append(i[1])
        if i[1] in other_b_heavy and i[0].startswith("H"):
            remove_h2.append(i[0])
    common_pairs_deh2 = []
    for i in common_pairs_deh1:
        if i[1] not in remove_h2:
            common_pairs_deh2.append(i)

    return common_pairs_deh2

def common_a_b(common_pairs, rtf_a, rtf_b):
    
    dict_a_c = dict()
    dict_a_a = dict()
    dict_b_c = dict()
    dict_b_b = dict()

    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    common_a = [i[0] for i in common_pairs]
    common_b = [i[1] for i in common_pairs]

    for i in rtf_a.atomnames:
        if i in common_a:
            dict_a_c[i] = str(i+"C")
        else:
            dict_a_a[i] = str(i+"A")

    for i in rtf_b.atomnames:
        if i in common_b:
            dict_b_c[i] = str(i+"C")
        else:
            dict_b_b[i] = str(i+"B")

    return dict_a_c, dict_a_a, dict_b_c, dict_b_b

def charge_distribute(common_pairs, rtf_a, rtf_b):

    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    dict_newatomname_charge = dict()

    common_charge_all = 0
    for i in common_pairs:
        dict_newatomname_charge[str(i[0]+"C")] = (float(rtf_a.atomname_charge[i[0]]) + float(rtf_b.atomname_charge[i[1]]))/2
        common_charge_all += (float(rtf_a.atomname_charge[i[0]]) + float(rtf_b.atomname_charge[i[1]]))/2

    common_a = [i[0] for i in common_pairs]
    other_a = [i for i in rtf_a.atomnames if i not in common_a]
    charge_plus = (float(rtf_a.charge) - common_charge_all)/len(other_a)
    for i in other_a:
        dict_newatomname_charge[str(i+"A")] = float(rtf_a.atomname_charge[i]) + charge_plus

    common_b = [i[1] for i in common_pairs]
    other_b = [i for i in rtf_b.atomnames if i not in common_b]
    charge_plus = (float(rtf_b.charge) - common_charge_all)/len(other_b)
    for i in other_b:
        dict_newatomname_charge[str(i+"B")] = float(rtf_b.atomname_charge[i]) + charge_plus

    charge_other = float(rtf_a.charge) + float(rtf_b.charge) - sum(dict_newatomname_charge.values())

    tmp = 0
    for i in dict_newatomname_charge.keys():
        if abs(dict_newatomname_charge[i]) > tmp:
            tmp = abs(dict_newatomname_charge[i])
            max_atom = i

    dict_newatomname_charge[max_atom] += charge_other

    print("charge:", sum(dict_newatomname_charge.values()))

    return dict_newatomname_charge

def out_hybrid_pdb(common_pairs, pdb_a, pdb_b, rtf_a, rtf_b):

    dict_a_c, dict_a_a, dict_b_c, dict_b_b = common_a_b(common_pairs, rtf_a, rtf_b)
    pdb_a = Lig_pdb(pdb_a)
    pdb_b = Lig_pdb(pdb_b)
    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)
    
    rt = open("hybrid.pdb", "w")
    rt.write("REMARK  created by CHARMM-GUI\n")
    rt.write("REMARK  DATE: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    atomid = 0
    
    for i in rtf_a.atomnames:
        atomid += 1
        if i in dict_a_c.keys():
            rt.write("{:6s}{:5d} {:^4s}".format("HETATM", atomid, dict_a_c[i]) + pdb_a.atomname_property[i].replace("LIG", "MUT"))
        else:
            rt.write("{:6s}{:5d} {:^4s}".format("HETATM", atomid, dict_a_a[i]) + pdb_a.atomname_property[i].replace("LIG", "MUT"))

    for i in rtf_b.atomnames:
        if i in dict_b_c.keys():
            pass
        else:
            atomid += 1
            rt.write("{:6s}{:5d} {:^4s}".format("HETATM", atomid, dict_b_b[i]) + pdb_b.atomname_property[i].replace("LIG", "MUT"))
    
    rt.write("END")
    rt.close()

def judege_lsts(lst1, lst2):

    lst1 = sorted(lst1)
    lst2 = sorted(lst2)
    count = 0
    if len(lst1) == len(lst2):
        for i in range(0, len(lst1)):
            if lst1[i] == lst2[i]:
                count+=1
            else:
                return False
        if count == len(lst1):
            return True
        else:
            pass  
    else:
        return False

def out_hybrid_rtf(common_pairs, rtf_a, rtf_b):

    dict_a_c, dict_a_a, dict_b_c, dict_b_b = common_a_b(common_pairs, rtf_a, rtf_b)
    dict_newatomname_charge = charge_distribute(common_pairs, rtf_a, rtf_b)
    common_a = [i[0] for i in common_pairs]
    common_b = [i[1] for i in common_pairs]
    dict_common = dict(zip(common_b, common_a))
    rtf_a = Lig_rtf(rtf_a)
    rtf_b = Lig_rtf(rtf_b)

    rt = open("hybrid.rtf", "w")
    rt.write("* Topologies generated by CHARMM-GUI\n")
    rt.write("* DATE: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    rt.write("* \n")
    rt.write("36 1\n\n")
    rt.write("! \"penalty\" is the highest penalty score of the associated parameters.\n! Penalties lower than 10 indicate the analogy is fair; penalties between 10\n! and 50 mean some basic validation is recommended; penalties higher than\n! 50 indicate poor analogy and mandate extensive validation/optimization.\n\n")
    rt.write("{:5s}{:14s}{:>6.3f}".format("RESI", "MUT", (float(rtf_a.charge)+float(rtf_b.charge))) + "\n")
    rt.write("GROUP            ! CHARGE   CH_PENALTY\n")
    atomid = 0

    for i in rtf_a.atomnames:
        atomid += 1
        if i in dict_a_c.keys():
            rt.write("{:5s}{:7s}{:7s}{:7.3f} ! {:>8s}\n".format("ATOM", dict_a_c[i], rtf_a.atomname_type[i], dict_newatomname_charge[dict_a_c[i]], rtf_a.atomname_penalty[i]))
        else:
            rt.write("{:5s}{:7s}{:7s}{:7.3f} ! {:>8s}\n".format("ATOM", dict_a_a[i], rtf_a.atomname_type[i], dict_newatomname_charge[dict_a_a[i]], rtf_a.atomname_penalty[i]))

    for i in rtf_b.atomnames:
        if i in dict_b_c.keys():
            pass
        else:
            atomid += 1
            rt.write("{:5s}{:7s}{:7s}{:7.3f} ! {:>8s}\n".format("ATOM", dict_b_b[i], rtf_b.atomname_type[i], dict_newatomname_charge[dict_b_b[i]], rtf_b.atomname_penalty[i]))
    rt.write("\n")

    dict_a_atomname = dict()
    dict_a_atomname.update(dict_a_c)
    dict_a_atomname.update(dict_a_a)
    for i in rtf_a.bonds:
        rt.write("{:5s}{:5s}{:>4s}\n".format("BOND", dict_a_atomname[i[0]], dict_a_atomname[i[1]]))

    dict_b_atomname = dict()
    dict_b_a_c = dict()
    for i in dict_b_c.keys():
        dict_b_a_c[i] = dict_a_c[dict_common[i]]
    dict_b_atomname.update(dict_b_a_c)
    dict_b_atomname.update(dict_b_b)
    for i in rtf_b.bonds:
        if (i[0] in common_b) and (i[1] in common_b):
            if ([dict_common[i[0]], dict_common[i[1]]] in rtf_a.bonds) or ([dict_common[i[1]], dict_common[i[0]]] in rtf_a.bonds):
                pass
            else:
                rt.write("{:5s}{:5s}{:>4s}\n".format("BOND", dict_b_atomname[i[0]], dict_b_atomname[i[1]]))
        else:
            rt.write("{:5s}{:5s}{:>4s}\n".format("BOND", dict_b_atomname[i[0]], dict_b_atomname[i[1]]))
    rt.write("\n")
    
    for i in rtf_a.impr:
        rt.write("{:5s}{:7s}{:7s}{:7s}{:6s}\n".format("IMPR", dict_a_atomname[i[0]], dict_a_atomname[i[1]], dict_a_atomname[i[2]], dict_a_atomname[i[3]]))

    for i in rtf_b.impr:
        if (i[0] in common_b) and (i[1] in common_b) and (i[2] in common_b) and (i[3] in common_b):
            map_a = [dict_common[i[0]], dict_common[i[1]],dict_common[i[2]], dict_common[i[3]]]
            a = 0
            while a < len(rtf_a.impr):
                if dict_common[i[0]] == rtf_a.impr[a][0] and judege_lsts(rtf_a.impr[a], map_a):
                    break
                else:
                    a += 1
            if a == len(rtf_a.impr):
                rt.write("{:5s}{:7s}{:7s}{:7s}{:6s}\n".format("IMPR", dict_b_atomname[i[0]], dict_b_atomname[i[1]], dict_b_atomname[i[2]], dict_b_atomname[i[3]]))
        else:
            rt.write("{:5s}{:7s}{:7s}{:7s}{:6s}\n".format("IMPR", dict_b_atomname[i[0]], dict_b_atomname[i[1]], dict_b_atomname[i[2]], dict_b_atomname[i[3]]))
    rt.write("\nEND\n")

def out_hybrid_prm(prm_1, prm_2):
    prm_1 = Lig_prm(prm_1)
    prm_2 = Lig_prm(prm_2)

    rt = open("hybrid.prm", "w")
    rt.write("".join(prm_1.title))
    rt.write("BONDS\n")
    if len(prm_1.bonds) == 0 and len(prm_2.bonds) == 0:
        rt.write("\n")
    elif len(prm_1.bonds) != 0 and len(prm_2.bonds) == 0:
        for i in prm_1.bonds:
            rt.write("{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")
    elif len(prm_1.bonds) == 0 and len(prm_2.bonds) != 0:
        for i in prm_2.bonds:
            rt.write("{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")
    else:
        for i in prm_1.bonds:
            print(i)
            rt.write("{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], float(i[1][0]), float(i[1][1]), i[2]))
        for i in prm_2.bonds:
            j = len(prm_1.bonds)
            while j > 0:
                if i[0] == prm_1.bonds[j-1][0]:
                    break
                else:
                    j -= 1
            if j == 0:
                rt.write("{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")
    
    rt.write("ANGLES\n")
    if len(prm_1.angles) == 0 and len(prm_2.angles) == 0:
        rt.write("\n")
    elif len(prm_1.angles) != 0 and len(prm_2.angles) == 0:
        for i in prm_1.angles:
            rt.write("{:7s}{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], i[0][2], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")
    elif len(prm_1.angles) == 0 and len(prm_2.angles) != 0:
        for i in prm_2.angles:
            rt.write("{:7s}{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], i[0][2], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")
    else:
        for i in prm_1.angles:
            rt.write("{:7s}{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], i[0][2], float(i[1][0]), float(i[1][1]), i[2]))
        for i in prm_2.angles:
            j = len(prm_1.angles)
            while j > 0:
                if i[0] == prm_1.angles[j-1][0]:
                    break
                else:
                    j -= 1
            if j == 0:
                rt.write("{:7s}{:7s}{:7s}{:>7.2f}{:>11.4f} !{}".format(i[0][0], i[0][1], i[0][2], float(i[1][0]), float(i[1][1]), i[2]))
        rt.write("\n")

    rt.write("DIHEDRALS\n")
    if len(prm_1.dihedrals) == 0 and len(prm_2.dihedrals) == 0:
        rt.write("\n")
    elif len(prm_1.dihedrals) != 0 and len(prm_2.dihedrals) == 0:
        for i in prm_1.dihedrals:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")
    elif len(prm_1.dihedrals) == 0 and len(prm_2.dihedrals) != 0:
        for i in prm_2.dihedrals:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")
    else:
        for i in prm_1.dihedrals:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        for i in prm_2.dihedrals:
            j = len(prm_1.dihedrals)
            while j > 0:
                if i[0] == prm_1.dihedrals[j-1][0]:
                    break
                else:
                    j -= 1
            if j == 0:
                rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")

    rt.write("IMPROPERS\n")
    if len(prm_1.impropers) == 0 and len(prm_2.impropers) == 0:
        rt.write("\n")
    elif len(prm_1.impropers) != 0 and len(prm_2.impropers) == 0:
        for i in prm_1.impropers:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")
    elif len(prm_1.impropers) == 0 and len(prm_2.impropers) != 0:
        for i in prm_2.impropers:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")
    else:
        for i in prm_1.impropers:
            rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float([1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        for i in prm_2.impropers:
            j = len(prm_1.impropers)
            while j > 0:
                if i[0] == prm_1.impropers[j-1][0]:
                    break
                else:
                    j -= 1
            if j == 0:
                rt.write("{:7s}{:7s}{:7s}{:7s}{:>10.4f}{:>3d}{:>9.2f} !{}".format(i[0][0], i[0][1], i[0][2], i[0][3], float(i[1][0]), int(i[1][1]), float(i[1][2]), i[2]))
        rt.write("\n")
    rt.write("".join(prm_1.end))

def generate_lig_pdbpsf(ff_path, vmd_path):

    mklig = open("mk_lig.tcl", "w")
    mklig.write(
'''
package require psfgen
psfcontext reset

topology {0}/top_all36_prot.rtf
topology {0}/top_all36_na.rtf
topology {0}/top_all36_carb.rtf
topology {0}/top_all36_lipid_ljpme.rtf
topology {0}/top_all36_cgenff.rtf
topology {0}/toppar_water_ions.str
topology hybrid.rtf

segment MUT {{
    first none
    last none
    pdb hybrid.pdb
    }}
coordpdb hybrid.pdb MUT

writepdb ligand.pdb
writepsf ligand.psf

exit
'''.format(os.path.join(ff_path))
    )
    mklig.close()
    cmd = vmd_path +" -dispdev text -e mk_lig.tcl"
    os.system(cmd)
    time.sleep(1)

def generate_complex_pdbpsf(vmd_path, receptor_pdb, receptor_psf):

    segname_delig = [i for i in Lig_pdb(receptor_pdb).segname if i != "LIG"]
    segname = " ".join(segname_delig)
    mk_complex = open("mk_complex.tcl", "w")
    mk_complex.write(
"""
package require topotools

mol new ligand.psf
mol addfile ligand.pdb
mol new {0}
mol addfile {1}
set sel1 [atomselect 0 all]
set sel2 [atomselect 1 "segname {2}"]
set mol [::TopoTools::selections2mol "$sel1 $sel2"]
animate write psf complex-fep.psf $mol
animate write pdb complex.pdb $mol

exit
""".format(receptor_psf, receptor_pdb, segname)
    )
    mk_complex.close()
    cmd = vmd_path + " -dispdev text -e mk_complex.tcl"
    os.system(cmd)
    time.sleep(1)

def mark_fep(common_pairs, rtf_1, rtf_2):
    dict_a_c, dict_a_a, _, dict_b_b = common_a_b(common_pairs, rtf_1, rtf_2)
    c_atom_lst = list(dict_a_c.values())
    a_atom_lst = list(dict_a_a.values())
    b_atom_lst = list(dict_b_b.values())
    with open("complex.pdb") as f:
        f1 = f.readlines()
    rt = open("complex-fep.pdb", "w")
    for i in f1:
        if i[66:76].strip() == "MUT":
            if i[12:16].strip() in c_atom_lst:
                rt.write(i)
            elif i[12:16].strip() in a_atom_lst:
                rt.write(i[:60] + "{:6.2f}".format(-1) + i[66:])
            elif i[12:16].strip() in b_atom_lst:
                rt.write(i[:60] + "{:6.2f}".format(1) + i[66:])
        else:
            rt.write(i)

def md_pbc_box(vmd_path):
    mk_pbcbox = open("mk_pbcbox.tcl", "w")
    mk_pbcbox.write(
"""
#!/bin/bash
# vmd -dispdev text -e mk_pbcbox.tcl

package require psfgen
psfcontext reset
mol load psf complex-fep.psf pdb complex-fep.pdb
set everyone [atomselect top all]
set minmax [measure minmax $everyone]
foreach {min max} $minmax { break }
foreach {xmin ymin zmin} $min { break }
foreach {xmax ymax zmax} $max { break }

set file [open "PBCBOX.dat" w]
puts $file "cellBasisVector1 [ expr $xmax - $xmin ] 0 0 "
puts $file "cellBasisVector2 0 [ expr $ymax - $ymin ] 0 "
puts $file "cellBasisVector3 0 0 [ expr $zmax - $zmin ] "
puts $file "cellOrigin [ expr ($xmax + $xmin)/2 ] [ expr ($ymax + $ymin)/2 ] [ expr ($zmax + $zmin)/2 ] "

exit
"""
    )
    mk_pbcbox.close()
    cmd = vmd_path + " -dispdev text -e mk_pbcbox.tcl"
    os.system(cmd)
    time.sleep(1)

def position_constraints(vmd_path):
    seg_fep = Lig_pdb("complex-fep.pdb")
    seg_fep_new = []
    for i in seg_fep.segname:
        if not i.startswith("WT") and (not i.startswith("ION")):
            seg_fep_new.append(i)
    if not (len(seg_fep_new) == 1 and seg_fep_new[0] == "MUT"):
        mk_bonded = open("bonded_constraints.tcl", "w")
        mk_bonded.write(
"""
mol new complex-fep.pdb type pdb waitfor all
set all [atomselect top "all"]

$all set beta 0
set sel [atomselect top "(((segname PRO) and backbone) or ((segname MUT) and noh))"]
$sel set beta 1
$all writepdb constraints.pdb

quit
"""
    )
        mk_bonded.close()
        cmd = vmd_path + " -dispdev text -e bonded_constraints.tcl"
        os.system(cmd)
        time.sleep(1)
    else:
        mk_free = open("free_constraints.tcl", "w")
        mk_free.write(
"""
mol new complex-fep.pdb type pdb waitfor all
set all [atomselect top "all"]

$all set beta 0
set sel [atomselect top "((segname MUT) and noh)"]
$sel set beta 1
$all writepdb constraints.pdb

quit
"""
    )
        mk_free.close()
        cmd = vmd_path + " -dispdev text -e free_constraints.tcl"
        os.system(cmd)
        time.sleep(1)

def fep_tcl(fep_tcl):

    mk_feptcl = open(fep_tcl, "w")
    mk_feptcl.write(
"""
##############################################################
# FEP SCRIPT
# Jerome Henin <jhenin@ifr88.cnrs-mrs.fr>
#
# Changes:
# 2010-04-24: added runFEPmin
# 2009-11-17: changed for NAMD 2.7 keywords
# 2008-06-25: added TI routines
# 2007-11-01: fixed runFEP to handle backwards transformations
#             (i.e. dLambda < 0)
##############################################################

##############################################################
# Example NAMD input:
#
# source fep.tcl
#
# alch                  on
# alchFile              system.fep
# alchCol               B
# alchOutFreq           10
# alchOutFile           system.fepout
# alchEquilSteps        500
#
# set nSteps      5000
# set init {0 0.05 0.1}
# set end {0.9 0.95 1.0}
#
# runFEPlist $init $nSteps
# runFEP 0.1 0.9 0.1 $nSteps
# runFEPlist $end $nSteps
##############################################################

##############################################################
# proc runFEPlist { lambdaList nSteps }
#
# Run n FEP windows joining (n + 1) lambda-points
##############################################################

proc runFEPlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
      set win 1
    }

    set l1 [lindex $lambdaList 0]
    foreach l2 [lrange $lambdaList 1 end] {
      print [format "Running FEP window %3s: Lambda1 %-6s Lambda2 %-6s \[dLambda %-6s\]"\
        $win $l1 $l2 [expr $l2 - $l1]]
      firsttimestep    0
      alchLambda       $l1
      alchLambda2      $l2
      run              $nSteps

      set l1 $l2
      incr win
    }
}

proc runFEPlist_restart { lambdaList nSteps starting timestep } {
    # Keep track of window number
    global win
    if {![info exists win]} {
      set win $starting
    }

    set l1 [lindex $lambdaList $starting]
    foreach l2 [lrange $lambdaList [expr $starting + 1] end] {
      print [format "Running FEP window %3s: Lambda1 %-6s Lambda2 %-6s \[dLambda %-6s\]"\
        $win $l1 $l2 [expr $l2 - $l1]]
      if { $l1 == [lindex $lambdaList $starting] } {
        set firsttimestep $timestep
        alchEquilSteps    0
      } else {
        set firsttimestep 0
        alchEquilSteps    10000
      }
      firsttimestep    $firsttimestep
      alchLambda       $l1
      alchLambda2      $l2
      run              [expr $nSteps - $firsttimestep]

      set l1 $l2
      incr win
    }
}


##############################################################
# proc runFEP { start stop dLambda nSteps }
#
# FEP windows of width dLambda between values start and stop
##############################################################

proc runFEP { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runFEPlist $ll $nSteps
}


##############################################################
##############################################################

proc runFEPmin { start stop dLambda nSteps nMinSteps temp} {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    if { $nMinSteps > 0 } {
      alchLambda       $start
      alchLambda2      $start
      minimize $nMinSteps
      reinitvels $temp
    }

    runFEPlist $ll $nSteps
}

##############################################################
##############################################################

proc runTIlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
            set win 1
    }

    foreach l $lambdaList {
            print [format "Running TI window %3s: Lambda %-6s " $win $l ]
            firsttimestep 0
            alchLambda       $l
            run $nSteps
            incr win
    }
}


##############################################################
##############################################################

proc runTI { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runTIlist $ll $nSteps
}

##############################################################
# Increment lambda and try to correct truncation errors around
# 0 and 1
##############################################################

proc increment { lambda dLambda } {
    set epsilon 1e-15
    set new [expr { $lambda + $dLambda }]

    if { [expr $new > - $epsilon && $new < $epsilon] } {
      return 0.0
    }
    if { [expr ($new - 1) > - $epsilon && ($new - 1) < $epsilon] } {
      return 1.0
    }
    return $new
}
"""
    )

def submit_divide(do_fep, dup):
    mk_submit = open(do_fep, "w")
    mk_submit.write(
"""
#PBS -N {0}
#PBS -q fep
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -S /bin/bash
#PBS -j oe
#PBS -l walltime=168:00:00

date
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

NAMD="/public/software/apps/NAMD_3.0alpha9/namd3 +p1 +devices 0"

cd equil
base=fep-equil.conf
$NAMD $base > $base.log
cd ../prod
base=fep-prod.conf
$NAMD $base > $base.log
""".format(dup)
    )
    mk_submit.close()

def submit_all(do_fep, duplicate):
    mk_submit = open(do_fep, "w")
    mk_submit.write(
"""
#PBS -N dups
#PBS -q fep
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -S /bin/bash
#PBS -j oe
#PBS -l walltime=168:00:00

date
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

NAMD="/public/software/apps/NAMD_3.0alpha9/namd3 +p1 +devices 0"

for i in `seq 1 {0}`
do
    cd dup${{i}}
    cd equil
    base=fep-equil.conf
    $NAMD $base > $base.log
    cd ../prod
    base=fep-prod.conf
    $NAMD $base > $base.log
    cd ../../
done
""".format(str(duplicate))
    )
    mk_submit.close()

def fep_equil_config(equil_config, ff_path):
    mk_equil_config = open(equil_config, "w")
    mk_equil_config.write(
"""
#############################################################
## JOB DESCRIPTION                                         ##
#############################################################

# FEP: Minimization and Equilibration (NPT) of
# protein-ligand complex in a Water Box
# namd3 +p1 fep-com-equil.conf > fep-com-equil.log


#############################################################
## ADJUSTABLE PARAMETERS                                   ##
#############################################################

set  temp           310
set  outputbase     complex
set  outputName     $outputbase-equil
firsttimestep       0
# if you do not want to open this option, assign 0
set INPUTNAME       0                      ;# use the former outputName, for restarting a simulation
set CONSPDB     ../../common/constraints
set CONSSCALE   1                      ;# default; initial value if you want to change
set parpath     {0}

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

structure           ../../common/complex-fep.psf
coordinates         ../../common/complex-fep.pdb

# Input
paraTypeCharmm      on
parameters          ../../common/hybrid.prm  
parameters          ${{parpath}}/par_all36m_prot.prm
parameters          ${{parpath}}/par_all36_na.prm
parameters          ${{parpath}}/par_all36_carb.prm
parameters          ${{parpath}}/par_all36_lipid_ljpme.prm
parameters          ${{parpath}}/par_all36_cgenff.prm
parameters          ${{parpath}}/toppar_water_ions_namd.str
mergeCrossterms     yes

# restart or PBC
if {{ $INPUTNAME != 0 }} {{
    # restart
    BinVelocities $INPUTNAME.restart.vel.old
    BinCoordinates $INPUTNAME.restart.coor.old
    ExtendedSystem $INPUTNAME.restart.xsc.old
}} else {{
    # Periodic Boundary Conditions
    temperature $temp
    source ../../common/PBCBOX.dat
}}


## Force-Field Parameters
exclude             scaled1-4;         # non-bonded exclusion policy to use "none,1-2,1-3,1-4,or scaled1-4"
                                       # 1-2: all atoms pairs that are bonded are going to be ignored
                                       # 1-3: 3 consecutively bonded are excluded
                                       # scaled1-4: include all the 1-3, and modified 1-4 interactions
                                       # electrostatic scaled by 1-4scaling factor 1.0
                                       # vdW special 1-4 parameters in charmm parameter file.
1-4scaling          1.0

# CONSTANT-T
langevin                on
langevinTemp            $temp
langevinDamping         10.0

# CONSTANT-P, not in tutorial
useGroupPressure        yes;           # use a hydrogen-group based pseudo-molecular viral to calcualte pressure and
                                        # has less fluctuation, is needed for rigid bonds (rigidBonds/SHAKE)
useFlexibleCell         no;            # yes for anisotropic system like membrane
useConstantRatio        no;            # keeps the ratio of the unit cell in the x-y plane constant A=B
#    useConstatntArea     yes;
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    100;         # 100? 2000?
langevinPistonDecay     50;         # 50?
langevinPistonTemp      $temp
StrainRate              0.0 0.0 0.0

# CUT-OFFS
switching                on
switchdist              10.0
cutoff                  12.0
pairlistdist            13.5

PME                     yes
PMEGridSpacing          1.0
PMETolerance            10e-6
PMEInterpOrder          4

wrapWater               on;                # wrap water to central cell
wrapAll                 on;                # wrap other molecules too
wrapNearest             off;               # use for non-rectangular cells (wrap to the nearest image)

# SPACE PARTITIONING
splitpatch              hydrogen
hgroupcutoff            2.8
stepspercycle           20
margin                  2
longSplitting           C2

# RESPA PROPAGATOR
# timestep                1.0
timestep                2.0
useSettle               on
fullElectFrequency      2
nonbondedFreq           1

# SHAKE
rigidbonds              all
rigidtolerance          0.000001
rigiditerations         400

# COM
ComMotion               no

# vdw
vdwForceSwitching       on

#############################################################
## EXECUTION SCRIPT                                        ##
#############################################################

# Output

outputname              $outputName

# 500steps = every 1ps
# not important?
computeEnergies         50
outputenergies          1000
outputtiming            1000
outputpressure          1000
restartfreq             1000
XSTFreq                 1000
binaryoutput            yes
binaryrestart           yes

# Positional restraints
# Write out a separate pdb file in which the B values for
# the backbone, the non-hydrogen nucleotide atoms, the ion,
# and the water oxygens within 2.5 A of magnesium are set to 2
if {{ $CONSPDB != 0 }} {{
    Constraints          yes
    ConsRef              $CONSPDB.pdb
    ConsKFile            $CONSPDB.pdb
    ConskCol             B
    constraintScaling    $CONSSCALE
}}

source                  ../../common/fep.tcl

alch                    on
alchType                FEP
alchFile                ../../common/complex-fep.pdb
alchCol                 B
alchOutFile             $outputName.fepout
alchOutFreq             50

alchVdwLambdaEnd        1.0
alchElecLambdaStart     0.1
alchVdWShiftCoeff       1.0
alchDecouple            on

alchEquilSteps          10000
set numSteps            50000   ;#250000

set numMinSteps         5000

runFEPmin 0.0 0.0 0.0 $numSteps $numMinSteps $temp
""".format(os.path.join(ff_path))
    )
    mk_equil_config.close()

def fep_prod_config(prod_config, ff_path):
    mk_prod_config = open(prod_config, "w")
    mk_prod_config.write(
"""
#############################################################
## JOB DESCRIPTION                                         ##
#############################################################

# FEP: Forward run of
# protein-ligand complex in a Water Box
# namd3 +p1 +devices 0 fep-prod.conf > fep-prod.log

#############################################################
## ADJUSTABLE PARAMETERS                                   ##
#############################################################

set  temp           310
set  outputbase     complex
set  outputName     $outputbase-prod
set  INPUTNAME       0
# use the former outputName, for restarting a simulation
set parpath     {0}

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

structure           ../../common/complex-fep.psf
coordinates         ../../common/complex-fep.pdb

# Input
paraTypeCharmm      on
parameters          ../../common/hybrid.prm  
parameters          ${{parpath}}/par_all36m_prot.prm
parameters          ${{parpath}}/par_all36_na.prm
parameters          ${{parpath}}/par_all36_carb.prm
parameters          ${{parpath}}/par_all36_lipid_ljpme.prm
parameters          ${{parpath}}/par_all36_cgenff.prm
parameters          ${{parpath}}/toppar_water_ions_namd.str
mergeCrossterms yes

# restart or PBC
if {{ $INPUTNAME != 0 }} {{
    # restart
    BinVelocities $INPUTNAME.restart.vel.old
    BinCoordinates $INPUTNAME.restart.coor.old
    ExtendedSystem $INPUTNAME.restart.xsc.old
}} else {{
    # from equil. use the former outputName
    bincoordinates          ../equil/$outputbase-equil.coor
    binvelocities           ../equil/$outputbase-equil.vel
    extendedSystem          ../equil/$outputbase-equil.xsc
}}


## Force-Field Parameters
exclude             scaled1-4;         # non-bonded exclusion policy to use "none,1-2,1-3,1-4,or scaled1-4"
                                       # 1-2: all atoms pairs that are bonded are going to be ignored
                                       # 1-3: 3 consecutively bonded are excluded
                                       # scaled1-4: include all the 1-3, and modified 1-4 interactions
                                       # electrostatic scaled by 1-4scaling factor 1.0
                                       # vdW special 1-4 parameters in charmm parameter file.
1-4scaling              1.0

# CONSTANT-T
langevin                on
langevinTemp            $temp
langevinDamping         1.0

# CONSTANT-P, not in tutorial
useGroupPressure        yes;           # use a hydrogen-group based pseudo-molecular viral to calcualte pressure and
                                        # has less fluctuation, is needed for rigid bonds (rigidBonds/SHAKE)
useFlexibleCell         no;            # yes for anisotropic system like membrane
useConstantRatio        no;            # keeps the ratio of the unit cell in the x-y plane constant A=B
#    useConstatntArea     yes;
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    100;         # 100? 2000?
langevinPistonDecay     50;         # 50?
langevinPistonTemp      $temp
StrainRate              0.0 0.0 0.0

# CUT-OFFS
switching               on
switchdist              10.0
cutoff                  12.0
pairlistdist            13.5

PME                     yes
PMEGridSpacing          1.0
PMETolerance            10e-6
PMEInterpOrder          4

wrapWater               on;                # wrap water to central cell
wrapAll                 on;                # wrap other molecules too
wrapNearest             off;               # use for non-rectangular cells (wrap to the nearest image)

# SPACE PARTITIONING
splitpatch              hydrogen
hgroupcutoff            2.8
stepspercycle           20
margin                  2
longSplitting           C2

# RESPA PROPAGATOR
# timestep                1.0
timestep                2.0
useSettle               on
fullElectFrequency      2
nonbondedFreq           1

# SHAKE
rigidbonds              all
rigidtolerance          0.000001
rigiditerations         400

# COM
# according to P. Blood use "no" for first NPT run
# then use "yes" for all NPT runs afterward
COMmotion               yes

# vdw
vdwForceSwitching       on

CUDASOAintegrate         on


#############################################################
## EXECUTION SCRIPT                                        ##
#############################################################

# Output

outputname              $outputName

# 500steps = every 1ps
# not important?
computeEnergies         50
outputenergies          10200
outputtiming            10200
outputpressure          10200
restartfreq             10200
XSTFreq                 10200
dcdfreq                 202000  # steps. 10 frames/per window
binaryoutput            yes
binaryrestart           yes

source                  ../../common/fep.tcl

alch                    on
alchType                FEP
alchFile                ../../common/complex-fep.pdb
alchCol                 B
alchOutFile             $outputName.fepout
alchOutFreq             50  # 10

alchVdwLambdaEnd        1.0
alchElecLambdaStart     0.1
alchVdWShiftCoeff       1.0
alchDecouple            on

alchEquilSteps          20000
set numSteps            1020000  ;# 2ns a window

set all {{0.00 0.00001 0.0001 0.001 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99 0.999 0.9999 0.99999 1.00}}

runFEPlist $all $numSteps
""".format(os.path.join(ff_path))
    )
    mk_prod_config.close()


def run_fep():

    settings = config()
    if int(settings.common_pairs) == 0:
        common_pairs = find_common_atom_pairs(settings.pdb_a_path, settings.pdb_b_path)
    else:
        common_pairs = settings.common_pairs
    common_pairs_type = remove_difftypes(common_pairs, settings.rtf_a_path, settings.rtf_b_path)
    common_pairs_charge = remove_diffcharges(common_pairs_type, settings.rtf_a_path, settings.rtf_b_path)
    common_pairs_charge_dh = remove_diff_h(common_pairs_charge, settings.rtf_a_path, settings.rtf_b_path)
    out_hybrid_pdb(common_pairs_charge_dh, settings.pdb_a_path, settings.pdb_b_path, settings.rtf_a_path, settings.rtf_b_path)
    out_hybrid_rtf(common_pairs_charge_dh, settings.rtf_a_path, settings.rtf_b_path)
    out_hybrid_prm(settings.prm_a_path, settings.prm_b_path)
    generate_lig_pdbpsf(settings.ff_path, settings.vmd_path)
    generate_complex_pdbpsf(settings.vmd_path, settings.receptor_pdb, settings.receptor_psf)
    mark_fep(common_pairs_charge_dh, settings.rtf_a_path, settings.rtf_b_path)
    md_pbc_box(settings.vmd_path)
    position_constraints(settings.vmd_path)
    if os.path.exists(os.path.join(".", "common")):
        shutil.rmtree(os.path.join(".", "common"))
        os.makedirs(os.path.join(".", "common"))
    else:
        os.makedirs(os.path.join(".", "common"))
    shutil.move("hybrid.pdb", os.path.join(".", "common", "hybrid.pdb"))
    shutil.move("hybrid.rtf", os.path.join(".", "common", "hybrid.rtf"))
    shutil.move("hybrid.prm", os.path.join(".", "common", "hybrid.prm"))
    shutil.move("ligand.pdb", os.path.join(".", "common", "ligand.pdb"))
    shutil.move("ligand.psf", os.path.join(".", "common", "ligand.psf"))
    shutil.move("complex.pdb", os.path.join(".", "common", "complex.pdb"))
    shutil.move("complex-fep.pdb", os.path.join(".", "common", "complex-fep.pdb"))
    shutil.move("complex-fep.psf", os.path.join(".", "common", "complex-fep.psf"))
    shutil.move("mk_lig.tcl", os.path.join(".", "common", "mk_lig.tcl"))
    shutil.move("mk_complex.tcl", os.path.join(".", "common", "mk_complex.tcl"))
    shutil.move("mk_pbcbox.tcl", os.path.join(".", "common", "mk_pbcbox.tcl"))
    shutil.move("PBCBOX.dat", os.path.join(".", "common", "PBCBOX.dat"))
    try:
        shutil.move("bonded_constraints.tcl", os.path.join(".", "common", "bonded_constraints.tcl"))
    except:
        shutil.move("free_constraints.tcl", os.path.join(".", "common", "free_constraints.tcl"))
    shutil.move("constraints.pdb", os.path.join(".", "common", "constraints.pdb"))
    fep_tcl(os.path.join(".", "common", "fep.tcl"))
    for i in range(1, int(settings.duplicate)+1):
        if os.path.exists(os.path.join(".", "dup"+str(i))):
            shutil.rmtree(os.path.join(".", "dup"+str(i)))
            os.makedirs(os.path.join(".", "dup"+str(i), "equil"))
            os.makedirs(os.path.join(".", "dup"+str(i), "prod"))
        else:
            os.makedirs(os.path.join(".", "dup"+str(i), "equil"))
            os.makedirs(os.path.join(".", "dup"+str(i), "prod"))
        fep_equil_config(os.path.join(".", "dup"+str(i), "equil", "fep-equil.conf"), settings.ff_path)
        fep_prod_config(os.path.join(".", "dup"+str(i), "prod", "fep-prod.conf"), settings.ff_path)
        if int(settings.submit) == 0 or (int(settings.submit) == 2):
            submit_divide(os.path.join(".", "dup"+str(i), "do_fep.sh"), "dup"+str(i))
    if int(settings.submit) == 1 or (int(settings.submit) == 2):
        submit_all(os.path.join(".", "do_fep.sh"), settings.duplicate)

def main():
    run_fep()

if __name__=="__main__":
    main() 
