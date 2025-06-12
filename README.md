# Deep-CovBoost：Integrating physics-based simulations with data-driven deep learning represents a robust strategy for developing inhibitors targeting the main protease

![](images/TOC.png)

## Molecular Assembly
程序主体见GenMol文件夹。  
（1）分子碎片化。使用 [generatefragmentdict.py](GenMol/generatefragmentdict.py) 将小分子SMILES集合进行碎片化。  
（2）分子碎片整理。使用 [cleanfragmentdict.py](GenMol/cleanfragmentdict.py) 对分子碎片进行整理方便后续使用。  
（3）组装生成新分子。使用 [generatemollibrary.py](GenMol/generatemollibrary.py) 将先导化合物骨架与分子碎片组装生成新分子。

## AI-based Binding Affinity Prediction
程序主体见PredAffinity文件夹。 
（1）数据准备。使用 [DataSplit.py](PredAffinity/DataSplit.py) 对输入数据进行规范化处理并生成五折交叉验证的输入数据。  
（2）五折交叉验证。使用 [DDGmpnnTrain.py](PredAffinity/DDGmpnnTrain.py) 进行五折交叉验证。  
（3）预测。使用 [DDGmpnnPredict.py](PredAffinity/DDGmpnnPredict.py) 进行。 

## Free Energy Perturbation (FEP) 

![](images/FEP.png)

（1）小分子mol2文件准备。准备FEP计算所需的A状态和B状态的小分子配体mol2文件。  
（2）小分子力场文件准备。将上述A和B两个状态的小分子mol2文件使用 CHARMM-GUI 的 Ligand Reader & Modeler 模块生成力场文件。  
（3）FEP计算文件准备。基于上述力场文件使用 [FEPMolSetup.py](FEP/FEPMolSetup.py) 生成FEP计算所需文件。
（4）提交FEP计算。使用上述生成的运行脚本提交FEP计算即可。  