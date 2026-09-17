import csv
import os
import re
import pandas as pd
from Bio import SeqIO
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from Bio import SeqIO
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


current_dir = os.path.dirname(os.path.abspath(__file__))
def get_drug_info(drug_file):
    drug_list = []
    drug_dict = {}
    supplier = Chem.SDMolSupplier(drug_file)

    for mol in supplier:
        if mol is not None:  
            mol_id = mol.GetProp('DRUGBANK_ID')  
            drug_name = mol.GetProp('COMMON_NAME')  
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            # smiles = standardize_smiles(smiles)
            drug_list.append([mol_id, drug_name, smiles])
            drug_dict[mol_id] = smiles
    
    drug_df = pd.DataFrame(drug_list, columns=['DRUGID', 'DRUGNAME', 'SMILES'])
    drug_df.to_csv(current_dir + '/raw_data/DrugInfo.csv', index=False)

sdf_file = current_dir + '/raw_data/open structures.sdf'
get_drug_info(sdf_file)
