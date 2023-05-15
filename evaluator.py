import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif as MI
import datetime
import random
import torch
from model import *
import optuna
from optuna.samplers import TPESampler
import os
import tqdm
import tqdm
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as px
import torch.nn as nn
from loss import *

def main(seed):
    n_iter = 10
    q = 11
    # CREATE STUDY
    pathway = ['CTLA4']
    #pathway = ['CTLA4', 'CD8A', 'CD8B']
    #pathway = ['CTLA4','CD2']
    #pathway = ['CTLA4','CD2','CD48','CD53','CD58','CD84']
    study_name = 'q_{}__pathway_{}'.format(q,'-'.join(pathway))
    study_storage = "sqlite:///{}/{}.db".format(study_name,study_name)
    try:
        os.mkdir(study_name)
        os.mkdir("{}/s_genes".format(study_name))
        os.mkdir("{}/figs".format(study_name))
    except:
        pass
    # LOAD INPUT DATA
    s = datetime.datetime.now()
    in_dirc = 'data'
    GeneExp = pd.read_csv('{}/RNA.csv'.format(in_dirc)).iloc[:,:2048]
    k_g = [x for x in GeneExp.columns if "?" not in x]
    GeneExp = GeneExp[k_g].iloc[:,:2048]
    ALLGenes = GeneExp.columns
    TargPaths = pd.read_csv('{}/MUTATION.csv'.format(in_dirc))
    print(GeneExp)
    print(TargPaths)
    # SAMPLING FROM QUANTUM SAMPLER
    def Sampling(a, b, n, tau):
        m = quantum_sampler(a, b, n)
        prob = compute_modulo(m)
        m = (prob - prob.min())/(prob.max()-prob.min())
        m = threshold(m, tau)
        return m, prob
    def objective(trial):
        n = trial.suggest_int('n_layer', 4, 16)
        alpha = trial.suggest_float('alpha', -2*np.pi, 2*np.pi)
        beta = trial.suggest_float('beta', -2*np.pi, 2*np.pi)
        tau = trial.suggest_float('tau', 0.5, 0.7)
        temp = trial.suggest_float('temp', 1e3, 2e3)
        eps = trial.suggest_float('eps', np.pi/96, np.pi/24)
        a = np.random.uniform(alpha-eps,alpha+eps,size = (n,q))
        b = np.random.uniform(beta-eps,alpha+eps,size = (n,q))
        m, p_x_y = Sampling(a, b, n, tau)
        #print(m.shape, p_x_y.shape)
        print(qml.draw(quantum_sampler, show_matrices=True)(a,b, n))
        masks = m.bool()
        df = GeneExp.T
        df = df.to_numpy()
        df = torch.tensor(df)
        df = df[masks]
        selected_genes = GeneExp.T.index[masks]
        p_x_y = p_x_y[masks]
        selected_genes = pd.DataFrame(selected_genes)
        ## p_x_y: SAMPLING DISTRIBUTION FROM QUANTUM SAMPLER
        selected_genes['Score'] = p_x_y
        y = TargPaths[pathway]
        x = df.detach().numpy()
        selected_genes = pd.concat([selected_genes, pd.DataFrame(x)], axis = 1)
        selected_genes = selected_genes.sort_values(by = ['Score'],ascending=False).reset_index(drop=True)
        x = selected_genes.iloc[:,2:].to_numpy()
        x = torch.tensor(x)
        y = torch.tensor(y.values)
        px = torch.tensor(selected_genes['Score'].to_numpy())
        px = nn.Softmax(dim = 0)(px*temp)
        score = loss(x, y, px)
        it = trial.number
        selected_genes.to_csv('{}/s_genes/{}_geneset.csv'.format(study_name,str(it).zfill(5)), index = True)
        return score
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction = 'minimize',
            study_name=study_name,
            storage=study_storage,
            load_if_exists=True)
    study.optimize(objective, n_trials=n_iter)
    print('Elapsed Time: {}'.format(datetime.datetime.now() - s))
