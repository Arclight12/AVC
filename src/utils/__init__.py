import numpy as np
import torch

# -----------------------
# CONFIG
# -----------------------
MAX_SEQUENCE_LENGTH = 400
RAW_FEATURE_DIM = 5
FEATURE_DIM = 7

# -----------------------
# ARPAbet Dictionary (English)
# -----------------------
ARPABET = {
    "AA":0, "AE":1, "AH":2, "AO":3, "AW":4, "AY":5, "B":6, "CH":7, "D":8, "DH":9,
    "EH":10, "ER":11, "EY":12, "F":13, "G":14, "HH":15, "IH":16, "IY":17, "JH":18,
    "K":19, "L":20, "M":21, "N":22, "NG":23, "OW":24, "OY":25, "P":26, "R":27,
    "S":28, "SH":29, "T":30, "TH":31, "UH":32, "UW":33, "V":34, "W":35, "Y":36,
    "Z":37, "ZH":38, "<BLANK>":39, "<SOS>": 40, "<EOS>": 41
}

# Added SOS/EOS for Transformer
INDEX_TO_PH = {v: k for k, v in ARPABET.items()}

# -----------------------
# EMG-UKA Phonemes
# -----------------------
# Based on Supplementary/phoneList
EMG_UKA_PHONEMES = {
    "IY": 0, "IH": 1, "EH": 2, "AE": 3, "IX": 4, "AX": 5, "AH": 6, "UW": 7, 
    "UH": 8, "AO": 9, "AA": 10, "EY": 11, "AY": 12, "OY": 13, "AW": 14, "OW": 15, 
    "L": 16, "R": 17, "Y": 18, "W": 19, "ER": 20, "AXR": 21, "M": 22, "N": 23, 
    "NG": 24, "CH": 25, "JH": 26, "DH": 27, "B": 28, "D": 29, "G": 30, "P": 31, 
    "T": 32, "K": 33, "Z": 34, "ZH": 35, "V": 36, "F": 37, "TH": 38, "S": 39, 
    "SH": 40, "HH": 41, "XL": 42, "XM": 43, "XN": 44, "SIL": 45, 
    "<BLANK>": 46, "<SOS>": 47, "<EOS>": 48
}
INDEX_TO_PH_UKA = {v: k for k, v in EMG_UKA_PHONEMES.items()}

NUM_PHONEMES = len(ARPABET)
BLANK_INDEX = ARPABET["<BLANK>"]
SOS_INDEX = ARPABET["<SOS>"]
EOS_INDEX = ARPABET["<EOS>"]

# -----------------------
# Sentences for Synthetic Data
# -----------------------
SENTENCES = [
    "WHAT IS YOUR NAME", "WHERE ARE YOU FROM", "HOW ARE YOU", "NICE TO MEET YOU",
    "CAN YOU HELP ME", "THANK YOU VERY MUCH", "GOOD MORNING", "GOOD NIGHT",
    "SEE YOU LATER", "I AM FINE", "I LOVE YOU", "PLEASE WAIT", "EXCUSE ME",
    "SORRY ABOUT THAT", "I DON'T KNOW", "CAN YOU REPEAT", "TURN LEFT", "TURN RIGHT",
    "GO STRAIGHT", "STOP HERE", "I NEED WATER", "I AM HUNGRY", "I AM TIRED",
    "CALL THE DOCTOR", "OPEN THE DOOR", "CLOSE THE WINDOW", "TURN ON THE LIGHT",
    "TURN OFF THE FAN", "PLAY MUSIC", "STOP THE MUSIC", "WHAT TIME IS IT",
    "TODAY IS MONDAY", "TOMORROW IS TUESDAY", "I WANT COFFEE", "THIS IS DELICIOUS",
    "HOW MUCH IS THIS", "I WILL PAY", "KEEP THE CHANGE", "SEE YOU TOMORROW",
    "HAVE A GOOD DAY", "TAKE CARE", "BE CAREFUL", "DON'T WORRY", "I UNDERSTAND",
    "I DON'T UNDERSTAND", "SPEAK SLOWLY", "WRITE IT DOWN", "SHOW ME AGAIN",
    "ONE MORE TIME", "YES PLEASE", "NO THANKS"
]

SENTENCE_PHONEMES = {
    "HELLO WORLD": ["HH","EH","L","OW","W","ER","L","D"],
    "WHAT IS YOUR NAME": ["W","AH","T","IH","Z","Y","AO","R","N","EY","M"],
    "WHERE ARE YOU FROM": ["W","EH","R","AA","R","Y","UW","F","R","AH","M"],
    "HOW ARE YOU": ["HH","AW","AA","R","Y","UW"],
    "HELLO HOW ARE YOU": ["HH","EH","L","OW","HH","AW","AA","R","Y","UW"],
    "NICE TO MEET YOU": ["N","AY","S","T","UW","M","IY","T","Y","UW"],
    "CAN YOU HELP ME": ["K","AE","N","Y","UW","HH","EH","L","P","M","IY"],
    "THANK YOU VERY MUCH": ["TH","AE","NG","K","Y","UW","V","EH","R","IY","M","AH","CH"],
    "GOOD MORNING": ["G","UH","D","M","AO","R","N","IH","NG"],
    "GOOD NIGHT": ["G","UH","D","N","AY","T"],
    "SEE YOU LATER": ["S","IY","Y","UW","L","EY","T","ER"],
    "I AM FINE": ["AY","AE","M","F","AY","N"],
    "I LOVE YOU": ["AY","L","AH","V","Y","UW"],
    "PLEASE WAIT": ["P","L","IY","Z","W","EY","T"],
    "EXCUSE ME": ["IH","K","S","K","Y","UW","Z","M","IY"],
    "SORRY ABOUT THAT": ["S","AO","R","IY","AH","B","AW","T","DH","AE","T"],
    "I DON'T KNOW": ["AY","D","OW","N","T","N","OW"],
    "CAN YOU REPEAT": ["K","AE","N","Y","UW","R","IH","P","IY","T"],
    "TURN LEFT": ["T","ER","N","L","EH","F","T"],
    "TURN RIGHT": ["T","ER","N","R","AY","T"],
    "GO STRAIGHT": ["G","OW","S","T","R","EY","T"],
    "STOP HERE": ["S","T","AA","P","HH","IY","R"],
    "I NEED WATER": ["AY","N","IY","D","W","AO","T","ER"],
    "I AM HUNGRY": ["AY","AE","M","HH","AH","NG","G","R","IY"],
    "I AM TIRED": ["AY","AE","M","T","AY","ER","D"],
    "CALL THE DOCTOR": ["K","AO","L","DH","AH","D","AA","K","T","ER"],
    "OPEN THE DOOR": ["OW","P","AH","N","DH","AH","D","AO","R"],
    "CLOSE THE WINDOW": ["K","L","OW","Z","DH","AH","W","IH","N","D","OW"],
    "TURN ON THE LIGHT": ["T","ER","N","AA","N","DH","AH","L","AY","T"],
    "TURN OFF THE FAN": ["T","ER","N","AO","F","DH","AH","F","AE","N"],
    "PLAY MUSIC": ["P","L","EY","M","Y","UW","Z","IH","K"],
    "STOP THE MUSIC": ["S","T","AA","P","DH","AH","M","Y","UW","Z","IH","K"],
    "WHAT TIME IS IT": ["W","AH","T","T","AY","M","IH","Z","IH","T"],
    "TODAY IS MONDAY": ["T","UW","D","EY","IH","Z","M","AH","N","D","EY"],
    "TOMORROW IS TUESDAY": ["T","AH","M","AA","R","OW","IH","Z","T","UW","Z","D","EY"],
    "I WANT COFFEE": ["AY","W","AA","N","T","K","AO","F","IY"],
    "THIS IS DELICIOUS": ["DH","IH","S","IH","Z","D","IH","L","IH","SH","AH","S"],
    "HOW MUCH IS THIS": ["HH","AW","M","AH","CH","IH","Z","DH","IH","S"],
    "I WILL PAY": ["AY","W","IH","L","P","EY"],
    "KEEP THE CHANGE": ["K","IY","P","DH","AH","CH","EY","N","JH"],
    "SEE YOU TOMORROW": ["S","IY","Y","UW","T","AH","M","AA","R","OW"],
    "HAVE A GOOD DAY": ["HH","AE","V","AH","G","UH","D","D","EY"],
    "TAKE CARE": ["T","EY","K","K","EH","R"],
    "BE CAREFUL": ["B","IY","K","EH","R","F","AH","L"],
    "DON'T WORRY": ["D","OW","N","T","W","ER","IY"],
    "I UNDERSTAND": ["AY","AH","N","D","ER","S","T","AE","N","D"],
    "I DON'T UNDERSTAND": ["AY","D","OW","N","T","AH","N","D","ER","S","T","AE","N","D"],
    "SPEAK SLOWLY": ["S","P","IY","K","S","L","OW","L","IY"],
    "WRITE IT DOWN": ["R","AY","T","IH","T","D","AW","N"],
    "SHOW ME AGAIN": ["SH","OW","M","IY","AH","G","EH","N"],
    "ONE MORE TIME": ["W","AH","N","M","AO","R","T","AY","M"],
    "YES PLEASE": ["Y","EH","S","P","L","IY","Z"],
    "NO THANKS": ["N","OW","TH","AE","NG","K","S"],
}

# -----------------------
# Helper Functions
# -----------------------
def greedy_decode(log_probs, input_lens_reduced):
    """
    Decodes CTC output greedily.
    log_probs: (T, N, C)
    input_lens_reduced: (N,)
    """
    with torch.no_grad():
        preds = log_probs.argmax(dim=2)  # (T, N)
        preds = preds.transpose(0, 1).cpu().numpy()  # (N, T)
    
    results = []
    for i, Lr in enumerate(input_lens_reduced.cpu().numpy()):
        seq = preds[i, :Lr].tolist()
        out = []
        prev = -1
        for s in seq:
            if s != prev and s != BLANK_INDEX:
                out.append(INDEX_TO_PH.get(int(s), "?"))
            prev = s
        results.append(out)
    return results

def phoneme_error_rate(pred, true):
    """
    Computes Phoneme Error Rate (PER) using Levenshtein distance.
    """
    m, n = len(pred), len(true)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1): dp[i, 0] = i
    for j in range(n + 1): dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == true[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[m, n] / max(n, 1)
