from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import sys, os,re,platform,math
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import orfipy_core
import Bio
from Bio import SeqIO
from Bio.Seq import Seq
import FrameKmer
from Bio.SeqUtils import ProtParam
import math
import random
import itertools
import re
from itertools import product
from collections import Counter
import pickle

def read_data2(path):
    sequence = []
    rx = SeqIO.parse(path, "fasta")
    for x in list(rx):
        seq = str(x.seq)
        # 如果序列长度大于20000，则截断
        if len(seq) > 20000:
            seq = seq[:20000]
        sequence.append(seq)
    return sequence

def read_data(path):
    sequence = []
    rx  = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        sequence.append(seq)
    return sequence

def read_data1(path):
    pseq = []
    nseq = []
    rx  = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        if "Positive" in id:
            pseq.append(seq)
        if "Negative" in id:
            nseq.append(seq)
    return pseq,nseq

def check_for_orf(sequence):
    start_codon = "ATG"  # 起始密码子
    stop_codons = ["TAA", "TAG", "TGA"]  # 终止密码子

    # 在序列中查找起始密码子
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        if codon == start_codon:
            # 找到起始密码子，现在检查是否有终止密码子
            for j in range(i + 3, len(sequence) - 2, 3):
                codon = sequence[j:j + 3]
                if codon in stop_codons:
                    # 找到终止密码子，认为有ORF
                    return True

    # 如果没有找到ORF，返回False
    return False

def find_longest_sublist_length(my_list):
    if not my_list:
        return 0  # 如果列表为空，返回0

    longest_length = len(max(my_list, key=len))
    return longest_length

def pad_features(features, max_length):
    padded_features = []

    for feature_list in features:
        if len(feature_list) < max_length:
            # 计算需要填充的数量
            padding_length = max_length - len(feature_list)
            # 使用零进行填充，你也可以使用其他值
            padded_feature = feature_list + [0] * padding_length
        else:
            padded_feature = feature_list
        padded_features.append(padded_feature)

    return padded_features


start_codons = 'ATG'
stop_codons = 'TAG,TAA,TGA'
Coverage = 0

maxlen = 100000

def get_length(seq):
    return np.log(len(seq)+1)

def binary(seq):
    std = {"A": np.array([1, 0, 0, 0]),
           "T": np.array([0, 1, 0, 0]),
           "C": np.array([0, 0, 1, 0]),
           "G": np.array([0, 0, 0, 1])
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def BINARY_my(seq):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    code = []

    for aa in seq:
        if aa == '-':
            code.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)

    encodings.append(code)
    return encodings

def EIIP(seq):
    std = {"A": np.array([0.12601,0,0,0]),
           "T": np.array([0,0.13400,0,0]),
           "C": np.array([0,0,0.08060,0]),
           "G": np.array([0,0,0,0.13350])
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def entroy(seq):
    a = seq.count("A") / len(seq)
    t = seq.count("T") / len(seq)
    c = seq.count("C") / len(seq)
    g = seq.count("G") / len(seq)

    a = -a * np.log(a)
    t = -t * np.log(t)
    c = -c * np.log(c)
    g = -g * np.log(g)
    return [a,t,c,g]

'''
sum 
'''
def SNCP(seq):
    "NCP, Nucleotide chemical property"
    std = {"A": np.array([1, 1, 1]),
           "T": np.array([0, 0, 1]),
           "C": np.array([0, 1, 0]),
           "G": np.array([1, 0, 0]),
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

'''
sum
'''
def SPCP(seq):
    std = {
        "A": np.array([37.03, 83.8, 279.9, 122.7, 14.68]),
        "T": np.array([29.71, 102.7, 251.3, 35.7, 11.77]),
        "C": np.array([27.30, 71.5, 206.3, 69.2, 10.82]),
        "G": np.array([35.46, 68.8, 229.6, 124.0, 14.06])
    }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def z_curve(seq):
    '''
        x = (A+G)-(C+T)
        y = (A+C)-(G+T)
        z = (A+T)-(C+G)
    '''
    A = seq.count("A")/len(seq)
    T = seq.count("T")/len(seq)
    C = seq.count("C")/len(seq)
    G = seq.count("G")/len(seq)

    x = (A + G) - (C + T)
    y = (A + C) - (G + T)
    z = (A + T) - (C + G)
    return [x,y,z]



def mer2(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmp = x+y
            mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res


def mer3(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for z in "ATCG":
                tmp = x+y+z
                mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res


def mer4(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for z in "ATCG":
                for k in "ATCG":
                    tmp = x+y+z+k
                    mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res

def mers1(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmps = []
            for z in "ATCG":
                tmp = x+z+y
                tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum +seq.count(mer)
            sum = sum/len(seq)
            mers.append(sum)
    return mers

def mers2(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmps = []
            for z in "ATCG":
                for k in "ATCG":
                    tmp = x+z+k+y
                    tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum +seq.count(mer)
            sum = sum/len(seq)
            mers.append(sum)
    return mers

def mers3(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmps = []
            for z in "ATCG":
                for k in "ATCG":
                    for l in "ATCG":
                        tmp = x + z+k+l +y
                        tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum +seq.count(mer)
            sum = sum/len(seq)
            mers.append(sum)
    return mers

def mers4(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmps = []
            for z in "ATCG":
                for k in "ATCG":
                    for l in "ATCG":
                        for m in "ATCG":
                            tmp = x+z+k+l+m+y
                            tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum +seq.count(mer)
            sum = sum/len(seq)
            mers.append(sum)
    return mers


def mmers1(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for p in "ATCG":
                for q in "ATCG":
                    tmps = []
                    for z in "ATCG":
                        tmp = x+y+z+p+q
                        tmps.append(tmp)
                    sum = 0
                    for mer in tmps:
                        sum = sum +seq.count(mer)
                    sum = sum/len(seq)
                    mers.append(sum)
    return mers

def mmers2(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for p in "ATCG":
                for q in "ATCG":
                    tmps = []
                    for z in "ATCG":
                        for k in "ATCG":
                            tmp = x+y+z+k+p+q
                            tmps.append(tmp)
                    sum = 0
                    for mer in tmps:
                        sum = sum +seq.count(mer)
                    sum = sum/len(seq)
                    mers.append(sum)
    return mers

def mmers3(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for p in "ATCG":
                for q in "ATCG":
                    tmps = []
                    for z in "ATCG":
                        for k in "ATCG":
                            for l in "ATCG":
                                tmp = x+y+z+k+l+p+q
                                tmps.append(tmp)
                    sum = 0
                    for mer in tmps:
                        sum = sum +seq.count(mer)
                    sum = sum/len(seq)
                    mers.append(sum)
    return mers

def mmers4(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            for p in "ATCG":
                for q in "ATCG":
                    tmps = []
                    for z in "ATCG":
                        for k in "ATCG":
                            for l in "ATCG":
                                for m in "ATCG":
                                    tmp = x+y+z+k+l+m+p+q
                                    tmps.append(tmp)
                    sum = 0
                    for mer in tmps:
                        sum = sum +seq.count(mer)
                    sum = sum/len(seq)
                    mers.append(sum)
    return mers



def CTD(seq):
    n = float(len(seq))
    num_A, num_T, num_G, num_C = 0.0, 0.0, 0.0, 0.0
    AT_trans, AG_trans, AC_trans, TG_trans, TC_trans, GC_trans = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(seq) - 1):
        if seq[i] == "A":
            num_A = num_A + 1
        if seq[i] == "T":
            num_T = num_T + 1
        if seq[i] == "G":
            num_G = num_G + 1
        if seq[i] == "C":
            num_C = num_C + 1
        if (seq[i] == "A" and seq[i + 1] == "T") or (seq[i] == "T" and seq[i + 1] == "A"):
            AT_trans = AT_trans + 1
        if (seq[i] == "A" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "A"):
            AG_trans = AG_trans + 1
        if (seq[i] == "A" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "A"):
            AC_trans = AC_trans + 1
        if (seq[i] == "T" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "T"):
            TG_trans = TG_trans + 1
        if (seq[i] == "T" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "T"):
            TC_trans = TC_trans + 1
        if (seq[i] == "G" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "G"):
            GC_trans = GC_trans + 1

    a, t, g, c = 0, 0, 0, 0
    A0_dis, A1_dis, A2_dis, A3_dis, A4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    T0_dis, T1_dis, T2_dis, T3_dis, T4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    G0_dis, G1_dis, G2_dis, G3_dis, G4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    C0_dis, C1_dis, C2_dis, C3_dis, C4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(seq) - 1):
        if seq[i] == "A":
            a = a + 1
            if a == 1:
                A0_dis = ((i * 1.0) + 1) / n
            if a == int(round(num_A / 4.0)):
                A1_dis = ((i * 1.0) + 1) / n
            if a == int(round(num_A / 2.0)):
                A2_dis = ((i * 1.0) + 1) / n
            if a == int(round((num_A * 3 / 4.0))):
                A3_dis = ((i * 1.0) + 1) / n
            if a == num_A:
                A4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "T":
            t = t + 1
            if t == 1:
                T0_dis = ((i * 1.0) + 1) / n
            if t == int(round(num_T / 4.0)):
                T1_dis = ((i * 1.0) + 1) / n
            if t == int(round((num_T / 2.0))):
                T2_dis = ((i * 1.0) + 1) / n
            if t == int(round((num_T * 3 / 4.0))):
                T3_dis = ((i * 1.0) + 1) / n
            if t == num_T:
                T4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "G":
            g = g + 1
            if g == 1:
                G0_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G / 4.0)):
                G1_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G / 2.0)):
                G2_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G * 3 / 4.0)):
                G3_dis = ((i * 1.0) + 1) / n
            if g == num_G:
                G4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "C":
            c = c + 1
            if c == 1:
                C0_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C / 4.0)):
                C1_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C / 2.0)):
                C2_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C * 3 / 4.0)):
                C3_dis = ((i * 1.0) + 1) / n
            if c == num_C:
                C4_dis = ((i * 1.0) + 1) / n
    return [num_A / n, num_T / n, num_G / n, num_C / n,
            AT_trans / n - 1, AG_trans / (n - 1), AC_trans / (n - 1),
            TG_trans / n - 1, TC_trans / (n - 1), GC_trans / (n - 1),
            A0_dis, A1_dis, A2_dis, A3_dis, A4_dis,
            T0_dis, T1_dis, T2_dis, T3_dis, T4_dis,
            G0_dis, G1_dis, G2_dis, G3_dis, G4_dis,
            C0_dis, C1_dis, C2_dis, C3_dis, C4_dis]

import sys
sys.path.append("/home/ys/work/ch/lncloc/lncloc/code/feu/repDNA")
from repDNA.psenac import PseKNC, PseDNC
def psednc(seq):
    res = PseDNC(lamada=3).make_psednc_vec([seq])[0]
    return res

def pseknc(seq):
    res = PseKNC(k=3,lamada=3).make_pseknc_vec([seq])[0]
    return res

def psetranc(seq):
    res = PseKNC(k=4,lamada=3).make_pseknc_vec([seq])[0]
    return res

def psepentanc(seq):
    res = PseKNC(k=5,lamada=3).make_pseknc_vec([seq])[0]
    return res


def cumulativeSkew(seq):
    '''
        x = (G-C)/((G+C)+0.1)
        y = (A-T)/((A+T)+0.1)
    '''
    A = seq.count("A")/len(seq)
    T = seq.count("T")/len(seq)
    C = seq.count("C")/len(seq)
    G = seq.count("G")/len(seq)

    x = (G - C) / (G + C)
    y = (A - T) / (A + T)
    return [x,y]


def calculate_nonamer_composition_as_list(sequence):
    # 初始化一个列表来存储每种碱基的相对频率
    base_composition = [0, 0, 0, 0, 0]  # 顺序为 A, C, G, T, U

    # 遍历序列并计算 nonamer 中每种碱基的出现次数
    for i in range(len(sequence) - 8):  # 遍历九个连续碱基的窗口
        nonamer = sequence[i:i + 9]
        for j, base in enumerate("ACGTU"):
            base_composition[j] += nonamer.count(base)

    # 计算每种碱基的相对频率
    total_bases = sum(base_composition)
    composition = [count / total_bases for count in base_composition]

    return composition

class Fickett:
    def __init__(self):
        self.content_parameter = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0]
        self.position_parameter = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
        '''
        newly calculated lookup table for RNA full length
        '''
        self.position_probability = {
            "A": [0.51, 0.55, 0.57, 0.52, 0.48, 0.58, 0.57, 0.54, 0.50, 0.36],
            "C": [0.29, 0.44, 0.55, 0.49, 0.52, 0.60, 0.60, 0.56, 0.51, 0.38],
            "G": [0.62, 0.67, 0.74, 0.65, 0.61, 0.62, 0.52, 0.41, 0.31, 0.17],
            "T": [0.51, 0.60, 0.69, 0.64, 0.62, 0.67, 0.58, 0.48, 0.39, 0.24],
        }
        self.position_weight = {"A": 0.062, "C": 0.093, "G": 0.205, "T": 0.154}
        self.content_probability = {
            "A": [0.40, 0.55, 0.58, 0.58, 0.52, 0.48, 0.45, 0.45, 0.38, 0.19],
            "C": [0.50, 0.63, 0.59, 0.50, 0.46, 0.45, 0.47, 0.56, 0.59, 0.33],
            "G": [0.21, 0.40, 0.47, 0.50, 0.52, 0.56, 0.57, 0.52, 0.44, 0.23],
            "T": [0.30, 0.49, 0.56, 0.53, 0.48, 0.48, 0.52, 0.57, 0.60, 0.51]
        }
        self.content_weight = {"A": 0.084, "C": 0.076, "G": 0.081, "T": 0.055}

    def look_up_position_probability(self, value, base):
        '''
        look up positional probability by base and value
        '''
        if float(value) < 0:
            return None
        for idx, val in enumerate(self.position_parameter):
            if (float(value) >= val):
                return float(self.position_probability[base][idx]) * float(self.position_weight[base])

    def look_up_content_probability(self, value, base):
        '''
        look up content probability by base and value
        '''
        if float(value) < 0:
            return None
        for idx, val in enumerate(self.content_parameter):
            if (float(value) >= val):
                return float(self.content_probability[base][idx]) * float(self.content_weight[base])

    def fickett_value(self, dna):
        '''
        calculate Fickett value from full RNA transcript sequence
        '''
        if len(dna) < 2:
            return 0
        fickett_score = 0
        dna = dna
        total_base = len(dna)
        A_content = float(dna.count("A")) / total_base
        C_content = float(dna.count("C")) / total_base
        G_content = float(dna.count("G")) / total_base
        T_content = float(dna.count("T")) / total_base

        phase_0 = dna[::3]
        phase_1 = dna[1::3]
        phase_2 = dna[2::3]

        phase_0_A = phase_0.count("A")
        phase_1_A = phase_1.count("A")
        phase_2_A = phase_2.count("A")
        phase_0_C = phase_0.count("C")
        phase_1_C = phase_1.count("C")
        phase_2_C = phase_2.count("C")
        phase_0_G = phase_0.count("G")
        phase_1_G = phase_1.count("G")
        phase_2_G = phase_2.count("G")
        phase_0_T = phase_0.count("T")
        phase_1_T = phase_1.count("T")
        phase_2_T = phase_2.count("T")

        A_content = float(phase_0_A + phase_1_A + phase_2_A) / total_base
        C_content = float(phase_0_C + phase_1_C + phase_2_C) / total_base
        G_content = float(phase_0_G + phase_1_G + phase_2_G) / total_base
        T_content = float(phase_0_T + phase_1_T + phase_2_T) / total_base
        A_position = np.max([phase_0_A, phase_1_A, phase_2_A]) / (np.min([phase_0_A, phase_1_A, phase_2_A]) + 1.0)
        C_position = np.max([phase_0_C, phase_1_C, phase_2_C]) / (np.min([phase_0_C, phase_1_C, phase_2_C]) + 1.0)
        G_position = np.max([phase_0_G, phase_1_G, phase_2_G]) / (np.min([phase_0_G, phase_1_G, phase_2_G]) + 1.0)
        T_position = np.max([phase_0_T, phase_1_T, phase_2_T]) / (np.min([phase_0_T, phase_1_T, phase_2_T]) + 1.0)

        fickett_score += self.look_up_content_probability(A_content, "A")
        fickett_score += self.look_up_content_probability(C_content, "C")
        fickett_score += self.look_up_content_probability(G_content, "G")
        fickett_score += self.look_up_content_probability(T_content, "T")

        fickett_score += self.look_up_position_probability(A_position, "A")
        fickett_score += self.look_up_position_probability(C_position, "C")
        fickett_score += self.look_up_position_probability(G_position, "G")
        fickett_score += self.look_up_position_probability(T_position, "T")

        return fickett_score

def eiip(seq):
    eiip_dict = {
        "A": 0.12601,
        "T": 0.13400,
        "C": 0.08060,
        "G": 0.13350
    }
    res = []
    for x in seq:
        res.append(eiip_dict[x])
    res = np.abs(np.fft.fft(np.array(res)))
    res = np.percentile(res,q=[0,25,50,75,100]).tolist()
    return res

def find_max_orf(seq):
    seq = seq.upper()
    res = []
    for start,stop,strand,description in orfipy_core.orfs(seq,minlen=10,maxlen=maxlen):
        tmp = [int(start),int(stop),int(description.split(";")[2].split("=")[-1])]
        res.append(tmp)
    if res:
        max_index = max(res,key=lambda x:x[2])
    else:
        max_index = [0,0,0]
    return max_index

class FindCDS:
    '''
    Find the most like CDS in a given sequence
    The most like CDS is the longest ORF found in the sequence
    When having same length, the upstream ORF is printed
    modified from source code of CPAT 1.2.1 downloaded from https://sourceforge.net/projects/rna-cpat/files/?source=navbar
    '''

    def __init__(self, seq):
        self.seq = seq
        self.result = (0, 0, 0, 0, 0)
        self.longest = 0
        self.basepair = {"A": "T", "T": "A", "U": "A", "C": "G", "G": "C", "N": "N", "X": "X"}

    def _reversecompliment(self):
        return "".join(self.basepair[base] for base in self.seq)[::-1]

    def get_codons(self, frame_number):
        coordinate = frame_number
        while coordinate + 3 <= len(self.seq):
            yield (self.seq[coordinate:coordinate + 3], coordinate)
            coordinate += 3

    def find_longest_in_one(self, myframe, direction, start_codon, stop_codon):
        triplet_got = self.get_codons(myframe)
        starts = start_codon
        stops = stop_codon
        while True:
            try:
                codon, index = triplet_got.__next__()
            except StopIteration:
                break
            if codon in starts and codon not in stops:
                orf_start = index
                end_extension = False
                while True:
                    try:
                        codon, index = triplet_got.__next__()
                    except StopIteration:
                        end_extension = True
                        integrity = -1
                    if codon in stops:
                        integrity = 1
                        end_extension = True
                    if end_extension:
                        orf_end = index + 3
                        Length = (orf_end - orf_start)
                        if Length > self.longest:
                            self.longest = Length
                            self.result = [direction, orf_start, orf_end, Length, integrity]
                        if Length == self.longest and orf_start < self.result[1]:
                            self.result = [direction, orf_start, orf_end, Length, integrity]
                        break

    def longest_orf(self, direction, start_codon={"ATG": None}, stop_codon={"TAG": None, "TAA": None, "TGA": None}):
        return_orf = ""
        for frame in range(3):
            self.find_longest_in_one(frame, "+", start_codon, stop_codon)
        return_orf = self.seq[self.result[1]:self.result[2]][:]
        start_coordinate = self.result[1]
        strand_direction = "+"
        orf_integrity = self.result[4]
        if direction == "-":
            self.seq = self._reversecompliment()
            for frame in range(3):
                self.find_longest_in_one(frame, "-", start_codon, stop_codon)
            if self.result[0] == "-":
                return_orf = self.seq[self.result[1]:self.result[2]][:]
                start_coordinate = self.result[1]
                strand_direction = "-"
                orf_integrity = self.result[4]
        return return_orf, orf_integrity

def get_orf(seq):
    findCDS = FindCDS(seq)
    return_orf, orf_integrity = findCDS.longest_orf(seq)
    return return_orf, orf_integrity

def get_orf_coverge(seq):
    transript_length = get_length(seq)
    orf, _ = get_orf(seq)
    orf_length = len(orf)
    ORF_coverage = float(orf_length) / transript_length
    return ORF_coverage

def get_orf_frame_score(seq):
    ORF_length_in_frame1, _ = get_orf(seq)
    ORF_length_in_frame2, _ = get_orf(seq[1:])
    ORF_length_in_frame3, _ = get_orf(seq[2:])

    ORF_length_in_frame1 = len(ORF_length_in_frame1)
    ORF_length_in_frame2 = len(ORF_length_in_frame2)
    ORF_length_in_frame3 = len(ORF_length_in_frame3)

    ORF_len = [ORF_length_in_frame1, ORF_length_in_frame2, ORF_length_in_frame3]
    ORF_frame = ((ORF_len[0] - ORF_len[1]) ** 2 + (ORF_len[0] - ORF_len[2]) ** 2 + (ORF_len[1] - ORF_len[2]) ** 2) / 2
    return ORF_frame

def get_GC1(mRNA):
    if len(mRNA) < 3:
        numGC = 0
        mRNA = 'ATG'
    else:
        numGC = mRNA[0::3].count("C") + mRNA[0::3].count("G")
    return numGC * 1.0 / len(mRNA) * 3

# 0
def get_GC2(mRNA):
    if len(mRNA) < 3:
        numGC = 0
        mRNA = 'ATG'
    else:
        numGC = mRNA[1::3].count("C") + mRNA[1::3].count("G")
    return numGC * 1.0 / len(mRNA) * 3

def get_GC3(mRNA):
    if len(mRNA) < 3:
        numGC = 0
        mRNA = 'ATG'
    else:
        numGC = mRNA[2::3].count("C") + mRNA[2::3].count("G")
    return numGC * 1.0 / len(mRNA) * 3

def gc(seq):
    seq0 = seq[0::3]
    seq1 = seq[1::3]
    seq2 = seq[2::3]
    res = []
    for seqx in [seq0,seq1,seq2]:
        gcx = (seqx.count("G")+seqx.count("C"))/len(seq)
        res.append(gcx)
    return res

def get_gc1_frame_score(seq):
    GC1_in_frame1 = get_GC1(seq)
    GC1_in_frame2 = get_GC1(seq[1:])
    GC1_in_frame3 = get_GC1(seq[2:])
    GC1_all = [GC1_in_frame1, GC1_in_frame2, GC1_in_frame3]
    GC1_frame = ((GC1_all[0] - GC1_all[1]) ** 2 + (GC1_all[0] - GC1_all[2]) ** 2 + (GC1_all[1] - GC1_all[2]) ** 2) / 2
    return GC1_frame

def get_gc2_frame_score(seq):
    GC2_in_frame1 = get_GC2(seq)
    GC2_in_frame2 = get_GC2(seq[1:])
    GC2_in_frame3 = get_GC2(seq[2:])
    GC2_all = [GC2_in_frame1, GC2_in_frame2, GC2_in_frame3]
    GC2_frame = ((GC2_all[0] - GC2_all[1]) ** 2 + (GC2_all[0] - GC2_all[2]) ** 2 + (GC2_all[1] - GC2_all[2]) ** 2) / 2
    return GC2_frame

def get_gc3_frame_score(seq):
    GC3_in_frame1 = get_GC3(seq)
    GC3_in_frame2 = get_GC3(seq[1:])
    GC3_in_frame3 = get_GC3(seq[2:])
    GC3_all = [GC3_in_frame1, GC3_in_frame2, GC3_in_frame3]
    GC3_frame = ((GC3_all[0] - GC3_all[1]) ** 2 + (GC3_all[0] - GC3_all[2]) ** 2 + (GC3_all[1] - GC3_all[2]) ** 2) / 2
    return GC3_frame

def get_stop_frame_score(seq):
    stop_num_in_frame1 = get_stop_codon_num(seq)
    stop_num_in_frame2 = get_stop_codon_num(seq[1:])
    stop_num_in_frame3 = get_stop_codon_num(seq[2:])
    stop_num_all = [stop_num_in_frame1, stop_num_in_frame2, stop_num_in_frame3]
    stop_num_frame = ((stop_num_all[0] - stop_num_all[1]) ** 2 + (stop_num_all[0] - stop_num_all[2]) ** 2 + (
            stop_num_all[1] - stop_num_all[2]) ** 2) / 2
    return stop_num_frame

def find_top_orf(seq,top=2):
    seq = seq.upper()
    res = []
    for start,stop,strand,description in orfipy_core.orfs(seq,minlen=10,maxlen=maxlen):
        tmp = [int(start),int(stop),int(description.split(";")[2].split("=")[-1])]
        res.append(tmp)
    if res:
        max_index = list(sorted(res,key=lambda x:x[2],reverse=True))[0:top]
        if len(max_index)<top:
            max_index = max_index+[[0,0,0]]*(top-len(max_index))
    else:
        max_index = [[0,0,0]]*top
    return max_index

def orf_num(seq):
    num = len(orfipy_core.orfs(seq, minlen=10, maxlen=len(seq)))/len(seq)
    return num

def AAC_my(seq):
    std = 'ACDEFGHIKLMNPQRSTVWY'
    res =[]
    for x in std:
        num = seq.count(x)/(len(seq)+1)
        res.append(num)
    return res

def AAINDEX_my(AAindex,seq):
    # 填充或截断序列到长度为300
    if len(seq) < 300:
        seq = seq.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(seq) > 300:
        seq = seq[:300]  # 截断到前300个字符
    AA = 'ARNDCQEGHILKMFPSTWYV'
    #
    # fileAAindex = r"/home/ys/work/ch/lncloc/lncloc/code/feu/data/AAindex.txt"
    # with open(fileAAindex) as f:
    #     records = f.readlines()[1:]
    #
    # AAindex = []
    # AAindexName = []
    # for i in records:
    #     AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
    #     AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []

    for sequence in seq:
        code = []
        for aa in sequence:
            if aa == '-':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)

    return encodings

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def CKSAAGP_my(seq, gap=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()
    encodings = []

    if isinstance(seq, str):
        seq = [seq]  # 将单个字符串转换为字符串列表

    for s in seq:
        code = []
        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key
        gPairIndex = []
        for key1 in groupKey:
            for key2 in groupKey:
                gPairIndex.append(key1 + '.' + key2)
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(s)):
                p2 = p1 + g + 1
                if p2 < len(s) and s[p1] in AA and s[p2] in AA:
                    gPair[index[s[p1]] + '.' + index[s[p2]]] = gPair[index[s[p1]] + '.' + index[s[p2]]] + 1
                    sum = sum + 1
            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)
        encodings.extend(code)

    return encodings


def CKSAAP_my(seq, gap=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []

    if isinstance(seq, str):
        seq = [seq]  # 将单个字符串转换为字符串列表

    for s in seq:
        code = []
        for aa1 in AA:
            for aa2 in AA:
                aaPairs.append(aa1 + aa2)
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(s)):
                index2 = index1 + g + 1
                if index1 < len(s) and index2 < len(s) and s[index1] in AA and s[index2] in AA:
                    myDict[s[index1] + s[index2]] = myDict[s[index1] + s[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                if sum == 0:
                    code.append(0)
                else:
                    code.append(myDict[pair] / sum)
        encodings.extend(code)

    return encodings


def BLOSUM62(seq):
    # 填充或截断序列到长度为300
    if len(seq) < 300:
        seq = seq.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(seq) > 300:
        seq = seq[:300]  # 截断到前300个字符
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    # header = ['#']
    # for i in range(1, len(fastas[0][1]) * 20 + 1):
    #     header.append('blosum62.F'+str(i))
    # encodings.append(header)

    for i in seq:
        code = []
        for aa in i:
            code.extend(blosum62[aa])
        encodings.extend(code)
    return encodings

import re
from collections import Counter

def CPSR_my(protein_sequence):
    # Define hydrophobic, electronic, and other amino acid groups
    hydrophobic_amino_acids = ['A', 'C', 'F', 'I', 'L', 'M', 'V', 'W']
    electronic_amino_acids = ['D', 'E', 'H', 'K', 'R']

    # Define hydrophobicity, hydrophilicity, rigidity, flexibility, and irreplaceability values
    hydrophobicity_values = {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48,
                            'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18,
                            'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08}
    hydrophilicity_values = {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0,
                            'H': -0.5, 'I': -1.8, 'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3,
                            'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5}
    rigidity_values = {'A': 1.0, 'C': 1.0, 'D': 0.6, 'E': 0.6, 'F': 0.8, 'G': 0.5, 'H': 0.7, 'I': 1.0, 'K': 0.5,
                     'L': 1.0, 'M': 0.9, 'N': 0.6, 'P': 0.5, 'Q': 0.6, 'R': 0.5, 'S': 0.7, 'T': 1.0, 'V': 1.0,
                     'W': 0.7, 'Y': 0.7}
    flexibility_values = {'A': 0.984, 'C': 0.906, 'D': 1.068, 'E': 1.094, 'F': 0.915, 'G': 1.031, 'H': 0.950, 'I': 0.927,
                        'K': 1.102, 'L': 0.935, 'M': 0.958, 'N': 1.048, 'P': 1.049, 'Q': 1.037, 'R': 1.008, 'S': 1.046,
                        'T': 1.051, 'V': 0.931, 'W': 0.904, 'Y': 0.929}
    irreplaceability_values = {'A': 0.9, 'C': 0.9, 'D': 1.0, 'E': 1.0, 'F': 0.3, 'G': 0.6, 'H': 0.3, 'I': 0.1, 'K': 0.1,
                           'L': 0.1, 'M': 0.3, 'N': 1.0, 'P': 0.6, 'Q': 0.6, 'R': 0.1, 'S': 0.9, 'T': 0.9, 'V': 0.3,
                           'W': 0.3, 'Y': 0.3}

    # Function to calculate 2G exchange group frequency
    def calculate_2G_exchange_group_frequency(protein_sequence):
        count = 0
        seq_len = len(protein_sequence)
        for i in range(seq_len - 2):
            if protein_sequence[i] != 'G' and protein_sequence[i+2] == 'G':
                count += 1
        frequency = count / (seq_len - 2)
        return frequency

    # Function to calculate the frequency of a group of amino acids
    def calculate_group_frequency(group, protein_sequence):
        count = 0
        seq_len = len(protein_sequence)
        for amino_acid in protein_sequence:
            if amino_acid in group:
                count += 1
        frequency = count / seq_len
        return frequency

    # Calculate features
    length = len(protein_sequence)
    group_frequency1 = calculate_group_frequency(hydrophobic_amino_acids, protein_sequence)
    group_frequency2 = calculate_group_frequency(electronic_amino_acids, protein_sequence)

    sum_hydrophobicity = sum(hydrophobicity_values.get(aa, 0) for aa in protein_sequence)
    sum_hydrophilicity = sum(hydrophilicity_values.get(aa, 0) for aa in protein_sequence)
    sum_rigidity = sum(rigidity_values.get(aa, 0) for aa in protein_sequence)
    sum_flexibility = sum(flexibility_values.get(aa, 0) for aa in protein_sequence)
    sum_irreplaceability = sum(irreplaceability_values.get(aa, 0) for aa in protein_sequence)

    return [length, calculate_2G_exchange_group_frequency(protein_sequence), group_frequency1, group_frequency2,
            sum_hydrophobicity, sum_hydrophilicity, sum_rigidity, sum_flexibility, sum_irreplaceability]


def Count(seq, group):
    sum = 0
    for aa in seq:
        sum = sum + group.count(aa)
    return sum

def CTDC_my(seq):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    encodings = []
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
    code = []

    for p in property:
        c1 = Count(seq, group1[p]) / len(seq)
        c2 = Count(seq, group2[p]) / len(seq)
        c3 = 1 - c1 - c2
        code.extend([c1, c2, c3])

    encodings.extend(code)
    return encodings


def Count1(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD_my(seq):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    code = []
    for p in property:
        code = code + Count1(group1[p], seq) + Count1(group2[p], seq) + Count1(group3[p], seq)
    encodings.extend(code)
    return encodings

def CTDT_my(seq):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    code = []
    aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
    for p in property:
        c1221, c1331, c2332 = 0, 0, 0
        for pair in aaPair:
            if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                c1221 = c1221 + 1
                continue
            if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                c1331 = c1331 + 1
                continue
            if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                c2332 = c2332 + 1
        code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
    return code

def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap+1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i+gap+1 < len(sequence) and i+2*gap+2<len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i+gap+1]]+'.'+AADict[sequence[i+2*gap+2]]
                myDict[fea] = myDict[fea] + 1

        max_value, min_value = max(myDict.values()), min(myDict.values())
        for f in features:
            if max_value != 0:
                res.append((myDict[f] - min_value) / max_value)
            else:
                res.append(0)
    return res

def CTriad(seq):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    code = []
    if len(seq) < 3:
        print('Error: for "CTriad" encoding, the input sequence should be greater than 3. \n\n')
        return 0
    code = code + CalculateKSCTriad(seq, 0, features, AADict)

    return code

def DDE(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    sequence = re.sub('-', '', seq)
    code = []
    tmpCode = [0] * 400
    for j in range(len(sequence) - 2 + 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i/sum(tmpCode) for i in tmpCode]

    myTV = []
    for j in range(len(myTM)):
        myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

    for j in range(len(tmpCode)):
        tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

    code = code + tmpCode

    return code

def DPC(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    code = []
    tmpCode = [0] * 400

    for j in range(len(sequence) - 2 + 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] + 1

    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]

    code = code + tmpCode

    return code

def EAAC_my(sequence, window=5):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    code = []

    for j in range(len(sequence)):
        if j < len(sequence) and j + window <= len(sequence):
            count = Counter(re.sub('-', '', sequence[j:j+window]))
            for key in count:
                count[key] = count[key] / len(sequence[j:j + window])
            for aa in AA:
                code.append(count[aa])

    encodings.extend(code)
    return encodings

def EGAAC_my(sequence, window=5):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()
    code = []
    # Remove '-' characters from the sequence
    sequence = re.sub('-', '', sequence)

    for j in range(len(sequence) - window + 1):
        subsequence = sequence[j:j + window]
        myDict = {}

        for key in groupKey:
            count = sum(subsequence.count(aa) for aa in group[key])
            myDict[key] = count / window

        for key in groupKey:
            code.append(myDict[key])
    return code

def GAAC_my(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()
    encodings = []
    code = []

    count = Counter(sequence)
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            myDict[key] = myDict.get(key, 0) + count[aa]

    for key in groupKey:
        code.append(myDict[key] / len(sequence))

    encodings.extend(code)
    return encodings

def GDPC_my(sequence):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    myDict = {t: 0 for t in dipeptide}
    sum = 0

    sequence = re.sub('-', '', sequence)

    for j in range(len(sequence) - 2 + 1):
        myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] += 1
        sum += 1

    if sum == 0:
        code = [0] * len(dipeptide)
    else:
        code = [myDict[t] / sum for t in dipeptide]

    return code

def GTPC_my(sequence):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    myDict = {t: 0 for t in triple}
    sum = 0

    sequence = re.sub('-', '', sequence)

    for j in range(len(sequence) - 3 + 1):
        myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] += 1
        sum += 1

    if sum == 0:
        code = [0] * len(triple)
    else:
        code = [myDict[t] / sum for t in triple]

    return code


def calculate_hybrid_aac(sequence, k=2):
    aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
               'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    ss_dict = {'H': 1, 'E': 2, 'C': 3}

    aac_vector = [0] * len(aa_dict)
    ss_vector = [0] * len(ss_dict)

    for aa in sequence:
        if aa in aa_dict:
            aac_vector[aa_dict[aa] - 1] += 1

    for i in range(len(sequence) - k + 1):
        ss = sequence[i:i+k][-1]
        if ss in ss_dict:
            ss_vector[ss_dict[ss] - 1] += 1

    aac_vector = np.array(aac_vector) / len(sequence)
    ss_vector = np.array(ss_vector) / (len(sequence) - k + 1)

    hybrid_aac_vector = list(aac_vector) + list(ss_vector)
    return hybrid_aac_vector

def calculate_hybrid_pseaac(sequence, k=3, lamda=3):
    aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
               'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

    aac_vector = [0] * len(aa_dict)
    for aa in sequence:
        if aa in aa_dict:
            aac_vector[aa_dict[aa] - 1] += 1
    aac_vector = [x / len(sequence) for x in aac_vector]

    pseaac_vector = []
    for i in range(1, k + 1):
        kmer_dict = {}
        for j in range(len(sequence) - i + 1):
            kmer = sequence[j:j + i]
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1

        weight = lamda ** (k - i)

        kmer_vector = [0] * len(aa_dict)
        for kmer, freq in kmer_dict.items():
            for j, aa in enumerate(kmer):
                kmer_vector[aa_dict[aa] - 1] += freq * weight

        denominator = len(sequence) - i + 1
        kmer_vector = [x / denominator if denominator != 0 else 0 for x in kmer_vector]
        pseaac_vector += kmer_vector

    hybrid_pseaac_vector = aac_vector + pseaac_vector
    return hybrid_pseaac_vector

def ksc_encoding(sequence, k=1, stride=1):
    # 定义20个氨基酸的编码
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    num = 20 ** k

    # 初始化特征向量
    feature_vector = np.zeros(num)

    # 对序列进行编码
    for i in range(0, len(sequence) - k * stride + 1, stride):
        kmer = sequence[i:i + k * stride:stride]
        if 'X' not in kmer and len(kmer) == k:
            index = sum([amino_acids.index(kmer[j]) * 20 ** (k - j - 1) for j in range(k)])
            feature_vector[index] += 1

    # 归一化特征向量
    feature_vector /= np.linalg.norm(feature_vector)
    # 将 NumPy 数组转换为 Python 列表
    feature_vector1 = feature_vector.tolist()
    return feature_vector1

def KSCTriad(sequence, gap=5):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    code = CalculateKSCTriad(sequence, gap, features, AADict)
    return code


def calculate_pcp(sequence):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    properties_table = {
        'A': [1.8, -0.5, 0.2, 0.62, 0.83],
        'R': [-4.5, 3.0, -3.5, 2.5, -1.4],
        'N': [-3.5, 0.2, -3.5, -1.03, -0.92],
        'D': [-3.5, 3.0, -3.5, -0.58, -0.90],
        'C': [2.5, -1.0, 0.2, 0.29, -1.3],
        'Q': [-3.5, 0.2, -3.5, -0.71, -1.1],
        'E': [-3.5, 3.0, -3.5, -0.74, -0.75],
        'G': [-0.4, 0.0, 0.0, 0.48, 1.15],
        'H': [-3.2, 0.5, -3.5, 0.11, -0.40],
        'I': [4.5, -1.8, 0.2, 1.38, 1.17],
        'L': [3.8, -1.8, 0.2, 1.06, -1.10],
        'K': [-3.9, 3.0, -3.5, 2.80, -1.50],
        'M': [1.9, -1.3, 0.2, 0.64, -0.81],
        'F': [2.8, -2.5, 0.2, 1.19, -1.71],
        'P': [-1.6, -0.2, 1.6, 0.12, 0.45],
        'S': [-0.8, 0.3, 0.4, -0.18, 0.65],
        'T': [-0.7, -0.4, 0.4, -0.05, 0.52],
        'W': [-0.9, -3.4, 0.2, 0.81, -2.09],
        'Y': [-1.3, -2.3, 0.2, 0.26, -0.68],
        'V': [4.2, -1.5, 0.2, 1.08, 1.00]
    }
    sequence = sequence.replace('-', '')
    # 初始化PCP特征列表
    pcp_features = []

    for aa in sequence:
        properties = properties_table.get(aa, None)
        if properties:
            pcp_features.extend(properties)
    return pcp_features


def calculate_pps(sequence):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    sequence = sequence.replace('-', '')  # 删除连字符
    properties = {
        'A': [1.8, -0.5, 0.2, -0.1],  # Alanine
        'C': [2.5, -1.0, 4.1, 0.3],  # Cysteine
        'D': [-3.5, 3.0, -3.1, -0.9],  # Aspartic Acid
        'E': [-3.5, 3.0, -1.8, -0.8],  # Glutamic Acid
        'F': [2.8, -2.5, 1.5, -1.6],  # Phenylalanine
        'G': [-0.4, 0.0, 0.0, 0.0],  # Glycine
        'H': [-3.2, -0.5, -0.5, 0.4],  # Histidine
        'I': [4.5, -1.8, -1.3, -1.5],  # Isoleucine
        'K': [-3.9, 3.0, -0.8, -1.3],  # Lysine
        'L': [3.8, -1.8, -1.3, -1.8],  # Leucine
        'M': [1.9, -1.3, -0.4, -0.9],  # Methionine
        'N': [-3.5, 0.2, 2.8, 0.2],  # Asparagine
        'P': [-1.6, 0.0, 0.0, 0.0],  # Proline
        'Q': [-3.5, 0.2, -2.4, -0.7],  # Glutamine
        'R': [-4.5, 3.0, 1.4, -1.4],  # Arginine
        'S': [-0.8, 0.3, 0.6, -0.6],  # Serine
        'T': [-0.7, -0.4, -0.7, -0.1],  # Threonine
        'V': [4.2, -1.5, -1.5, -1.8],  # Valine
        'W': [-0.9, -3.4, -0.9, -2.5],  # Tryptophan
        'Y': [-1.3, -2.3, 0.7, -0.7]  # Tyrosine
    }

    pps_vector = []

    for aa in sequence:
        if aa in properties:
            pps_vector.extend(properties[aa])
        else:
            raise ValueError(f"未知的氨基酸: {aa}")

    return pps_vector

def calculate_pse_aac(sequence, k=2, lamda=3):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    sequence = sequence.replace('-', '')  # 删除连字符
    # 构建k-mer字典
    k_mer_dict = {}
    k_mer_list = []
    for i in range(len(sequence) - k + 1):
        k_mer = sequence[i:i + k]
        if k_mer not in k_mer_dict:
            k_mer_dict[k_mer] = len(k_mer_dict) + 1
        k_mer_list.append(k_mer_dict[k_mer])

    # 计算lamda权重
    weights = []
    for i in range(1, lamda + 1):
        weights.append(0.5 * (1 - np.cos(2 * np.pi * i / lamda)))

    # 初始化PseAAC特征向量
    pse_aac_vector = [0] * len(k_mer_dict)

    # 计算PseAAC特征向量
    for j in range(1, len(k_mer_dict) + 1):
        pse_aac = 0
        for i in range(len(k_mer_list) - lamda):
            if k_mer_list[i] == j:
                for l in range(lamda):
                    if k_mer_list[i + l + 1] == j:
                        pse_aac += weights[l]
        pse_aac_vector[j - 1] = pse_aac

    return pse_aac_vector

def calculate_pssm_feature_vector(sequence):
    # 构建氨基酸索引
    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    # 初始化PSSM特征向量
    pssm_feature_vector = np.zeros(len(amino_acids))
    # 计算PSSM
    for aa in sequence:
        pssm_feature_vector[aa_index[aa]] += 1
    # 归一化PSSM特征向量
    pssm_feature_vector = pssm_feature_vector / len(sequence)
    # 将NumPy数组转换为列表
    pssm_feature_list = pssm_feature_vector.tolist()
    return pssm_feature_list

def calculate_saac_feature_vector(protein_sequence):
    # 将蛋白质序列转化为大写字母形式
    protein_sequence = protein_sequence.upper()

    # 定义氨基酸的分割规则
    groups = {
        'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'],
        'polar': ['S', 'T', 'C', 'N', 'Q'],
        'charged': ['D', 'E', 'K', 'R', 'H'],
        'special': ['G', 'P']
    }

    # 统计每个分组中的氨基酸数量
    saac_features = []
    for group_name, group_amino_acids in groups.items():
        count = 0
        for amino_acid in protein_sequence:
            if amino_acid in group_amino_acids:
                count += 1
        saac_features.append(count)

    # 计算每个分组所占比例
    total_count = len(protein_sequence)
    saac_features = [count / total_count for count in saac_features]

    return saac_features

def calculate_TPC_feature_vector(sequence):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    sequence = sequence.replace('-', '')  # 删除连字符
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    AADict = {}
    for i, aa in enumerate(AA):
        AADict[aa] = i

    feature_vector = [0] * 8000
    for j in range(len(sequence) - 3 + 1):
        feature_vector[AADict[sequence[j]] * 400 + AADict[sequence[j+1]] * 20 + AADict[sequence[j+2]]] += 1

    if sum(feature_vector) != 0:
        feature_vector = [i / sum(feature_vector) for i in feature_vector]

    return feature_vector

def compute_zscale_feature_vector(sequence):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
        '-': [0.00, 0.00, 0.00, 0.00, 0.00],
    }

    feature_vector = []
    for aa in sequence:
        feature_vector.extend(zscale[aa])

    return feature_vector

def calculate_BPF_feature_vector(sequence):
    # 填充或截断序列到长度为300
    if len(sequence) < 300:
        sequence = sequence.ljust(300, '-')  # 填充-字符直到长度为300
    elif len(sequence) > 300:
        sequence = sequence[:300]  # 截断到前300个字符
    sequence = sequence.replace('-', '')  # 删除连字符
    # 定义20个氨基酸的二元剖面编码
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.eye(20)
    encoding_dict = {amino_acids[i]: encoding[i] for i in range(20)}

    # 初始化二元剖面特征向量
    BPF_feature_vector = []

    # 计算二元剖面特征
    for aa in sequence:
        if aa in encoding_dict:
            BPF_feature_vector.extend(encoding_dict[aa])
            BPF_feature_vector.extend(encoding_dict[aa])

    return BPF_feature_vector

def cks_encoding_feature_vector(sequence, k=1, stride=1):
    # 定义20个氨基酸的编码
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.eye(20)
    encoding_dict = {amino_acids[i]: encoding[i] for i in range(20)}

    # 初始化特征向量
    feature_vector = np.zeros(400)  # 20 * 20

    # 对序列进行编码
    for i in range(0, len(sequence) - (k + 1) * stride, stride):
        pair = sequence[i:i + (k + 1) * stride:stride]
        if 'X' not in pair:  # 排除含有非标准氨基酸的氨基酸对
            index = amino_acids.index(pair[0]) * 20 + amino_acids.index(pair[k])
            feature_vector[index] += 1

    # 归一化特征向量
    feature_vector /= np.linalg.norm(feature_vector)

    return feature_vector.tolist()

def get_fs(seq):
    f1 = cumulativeSkew(seq)
    f2 = CTD(seq)
    f3 = psednc(seq)
    f4 = mer4(seq)
    f5 = pseknc(seq)
    f6 = psetranc(seq)
    f7 = psepentanc(seq)
    res = f1 + f2 + f3 + f4 + f5 + f6 + f7
    return res

def orf_fs(seq):
    orfs = []
    seq = seq.upper()
    s_e_l_s = find_top_orf(seq,top=1)

    orf_seq = ""
    for start,end,length in s_e_l_s:
        if length:
            orf_seq = seq[start:end]
        else:
            orf_seq = seq
        orfs.append(orf_seq)

    ress = []
    for orf_seq in orfs:
        seq = orf_seq

        seq_pro = str(Seq(seq).translate()).replace("*","")
        # aac = AAC_my(seq_pro)
        # aaindex = AAINDEX_my(AAindex,seq_pro)
        # bi = BINARY_my(seq_pro)
        # blousum = BLOSUM62(seq_pro)
        # bpf = calculate_BPF_feature_vector(seq_pro)
        # cks = cks_encoding_feature_vector(seq_pro)
        # cksaagp = CKSAAGP_my(seq_pro)
        cksaap = CKSAAP_my(seq_pro)
        # cpsr = CPSR_my(seq_pro)
        # ctdc = CTDC_my(seq_pro)
        # ctdd = CTDD_my(seq_pro)
        ctdt = CTDT_my(seq_pro)
        # ctriad = CTriad(seq_pro)
        # dde = DDE(seq_pro)
        # dpc = DPC(seq_pro)
        # eaac = EAAC_my(seq_pro)
        # egaac = EGAAC_my(seq_pro)
        # gaac = GAAC_my(seq_pro)
        # gdpc = GDPC_my(seq_pro)
        # gtpc = GTPC_my(seq_pro)
        # hybrid_aac = calculate_hybrid_aac(seq_pro)
        # hybeid_pseaac = calculate_hybrid_pseaac(seq_pro)
        # hybeid_pseaac1 = calculate_hybrid_pseaac(seq_pro,k=2)
        # hybeid_pseaac2 = calculate_hybrid_pseaac(seq_pro, k=4)
        # hybeid_pseaac3 = calculate_hybrid_pseaac(seq_pro, k=5)
        # ksc = ksc_encoding(seq_pro)
        # ksct = KSCTriad(seq_pro)
        # pcp = calculate_pcp(seq_pro)
        # pps = calculate_pps(seq_pro)
        # pseaac = calculate_pse_aac(seq_pro)
        # pseaac1 = calculate_pse_aac(seq_pro,k=3)
        # pseaac2 = calculate_pse_aac(seq_pro, k=4)
        # pseaac3 = calculate_pse_aac(seq_pro, k=5)
        # pssm = calculate_pssm_feature_vector(seq_pro)
        # qsorder = QSOrder(seq_pro)
        # saac = calculate_saac_feature_vector(seq_pro)
        tpc = calculate_TPC_feature_vector(seq_pro)
        # zscale = compute_zscale_feature_vector(seq_pro)
        # mw = Bio.SeqUtils.ProtParam.ProteinAnalysis(seq_pro).molecular_weight()
        # gravy = Bio.SeqUtils.ProtParam.ProteinAnalysis(seq_pro).gravy()
        # cu = cumulativeSkew(seq)
        # len_p = Bio.SeqUtils.ProtParam.ProteinAnalysis(seq_pro).length
        # ss = list(Bio.SeqUtils.ProtParam.ProteinAnalysis(seq_pro).secondary_structure_fraction())
        # gc1f = get_gc1_frame_score(seq)
        # gc2f = get_gc2_frame_score(seq)
        # gc3f = get_gc3_frame_score(seq)
        # gc_list = [gc1f,gc2f,gc3f]
        # fickett_val = Fickett().fickett_value(seq)
        # coding, noncoding = coding_nocoding_potential()
        # hexamer = FrameKmer.kmer_ratio(seq, 6, 3, coding, noncoding)
        tmp = cksaap + ctdt + tpc
        ress.extend(tmp)
    # res = ress + orf_list
    res = ress
    return res

def rh_fs(seq):
    res = orf_fs(seq) + get_fs(seq)
    return res

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from multiprocessing import Pool,cpu_count
pool = Pool(processes=cpu_count())
from sklearn.model_selection import train_test_split
print("Start extracting")
train_p_path = r"/your/data/path/data_p.txt"
train_n_path  = r"/your/data/path/data_n.txt"
print(train_p_path)
print(train_n_path)

pseq = read_data1(train_p_path)
nseq = read_data1(train_n_path)
print("train pseq",len(pseq))
print("train nseq",len(nseq))
train_p_data = pool.map(rh_fs,pseq)
train_n_data = pool.map(rh_fs,nseq)
train_p_data = np.array(train_p_data)
train_n_data = np.array(train_n_data)
data_p_path = f'/your/data/path/data_p.npy'  # 修改为你的路径
data_n_path = f'/your/data/path/data_n.npy'  # 修改为你的路径
np.save(data_p_path, train_p_data)
np.save(data_n_path, train_n_data)