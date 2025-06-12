# Deep-CovBoost：Integrating physics-based simulations with data-driven deep learning represents a robust strategy for developing inhibitors targeting the main protease

![](images/TOC.png)

## Molecular Assembly

<img src="images/GenMol.png" alt="GenMol" width="500" />  

程序主体见GenMol文件夹。  
（1）分子碎片化。使用 [generatefragmentdict.py](GenMol/generatefragmentdict.py) 将小分子SMILES集合进行碎片化。  
（2）分子碎片整理。使用 [cleanfragmentdict.py](GenMol/cleanfragmentdict.py) 对分子碎片进行整理方便后续使用。  
（3）组装生成新分子。使用 [generatemollibrary.py](GenMol/generatemollibrary.py) 将先导化合物骨架与分子碎片组装生成新分子。

## AI-based Binding Affinity Prediction

<img src="images/PredAffinity.png" alt="PredAffinity" width="500" />  

程序主体见PredAffinity文件夹。 
（1）数据准备。使用 [DataSplit.py](PredAffinity/DataSplit.py) 对输入数据进行规范化处理并生成五折交叉验证的输入数据。  
（2）五折交叉验证。使用 [DDGmpnnTrain.py](PredAffinity/DDGmpnnTrain.py) 进行五折交叉验证。  
（3）预测。使用 [DDGmpnnPredict.py](PredAffinity/DDGmpnnPredict.py) 进行。 

## Free Energy Perturbation (FEP) 

<img src="images/FEP.png" alt="FEP" width="300" />  

**​​(1) Prepare input files.​​**  
Prepare mol2 files of the ligand molecules representing State A and State B for FEP calculations. Use CHARMM-GUI's ​Ligand Reader & Modeler module​ to generate force field parameters for both ligand mol2 files.  

**(2) Generate dual-topology file.**  
Use [FEPMolSetup.py](FEP/FEPMolSetup.py) to create the dual-topology file required for FEP calculations:  
```python
python FEPMolSetup.py
```

**(3) Run FEP calculation.​**
```shell
sbatch do_fep.sh
```