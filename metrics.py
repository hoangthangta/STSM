import re

import datasets
from datasets import load_metric
#from evaluate import load as load_metric2
import sacrebleu

bleu = load_metric('bleu')
bertscore = load_metric('bertscore')
rouge = load_metric('rouge')
meteor = load_metric('meteor')

from utils import convert_datetime, match_single_value

from nltk import ngrams
from collections import Counter


#import spacy
#nlp = spacy.load('en_core_web_md')

'''import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        
from nltk.tokenize import wordpunct_tokenize, word_tokenize'''

def compute_osf_batch(preds, targets):
    """
        "Overall Slot Filling" in https://arxiv.org/pdf/1809.01797v2.pdf
    """

    f1 = []
    re = []
    pre = []
    for pred, target in zip(preds, targets):
        f1.append(compute_osf_single(pred, target)['osf_f1'])
        pre.append(compute_osf_single(pred, target)['osf_precision'])
        re.append(compute_osf_single(pred, target)['osf_recall'])
        
    f1 = sum(f1)/len(f1)
    re = sum(re)/len(re)
    pre = sum(pre)/len(pre)
    
    return {'osf_f1': f1, 'osf_precision': pre, 'osf_recall': re}
        

def compute_osf_single(pred, target):
    """
        "Overall Slot Filling" in https://arxiv.org/pdf/1809.01797v2.pdf
    """

    # split to pairs/triples
    pred_list, target_list = [], []
    
    import re
    pred_list = re.split('\| |>', pred)
    pred_list = [x.strip() for x in pred_list]
    
    target_list = re.split('\| |>', target)
    target_list = [x.strip() for x in target_list]
    
    if (len(pred_list) == 0 or len(target_list) == 0): return {'osf_f1': 0, 'osf_precision': 0, 'osf_recall': 0}

    # get overlap items
    items = set(pred_list).intersection(set(target_list))
    if (len(items) == 0): return {'osf_f1': 0, 'osf_precision': 0, 'osf_recall': 0}
    
    pre = len(items)/len(set(pred_list))
    re = len(items)/len(set(target_list))

    return {'osf_f1': (2*pre*re)/(pre + re), 'osf_precision': pre, 'osf_recall': re}

def compute_sf_batch(preds, targets):
    """
        "Overall Slot Filling" in https://arxiv.org/pdf/1809.01797v2.pdf
    """

    f1 = []
    re = []
    pre = []
    for pred, target in zip(preds, targets):
        f1.append(compute_sf_single(pred, target)['sf_f1'])
        pre.append(compute_sf_single(pred, target)['sf_precision'])
        re.append(compute_sf_single(pred, target)['sf_recall'])
        
    f1 = sum(f1)/len(f1)
    re = sum(re)/len(re)
    pre = sum(pre)/len(pre)
    
    return {'sf_f1': f1, 'sf_precision': pre, 'sf_recall': re}
    
def compute_sf_single(pred, target):
    """
        "Slot Filling" 
    """

    # split to pairs/triples
    pred_list, target_list = [], []
    
    import re
    pred_list = re.split('\| |> |:', pred)
    pred_list = [x.strip() for x in pred_list]

    target_list = re.split('\| |> |:', target)
    target_list = [x.strip() for x in target_list]
    
    if (len(pred_list) == 0 or len(target_list) == 0): return {'sf_f1': 0, 'sf_precision': 0, 'sf_recall': 0}

    # get overlap items
    items = set(pred_list).intersection(set(target_list))
    if (len(items) == 0): return {'sf_f1': 0, 'sf_precision': 0, 'sf_recall': 0}
    
    pre = len(items)/len(set(pred_list))
    re = len(items)/len(set(target_list))

    return {'sf_f1': (2*pre*re)/(pre + re), 'sf_precision': pre, 'sf_recall': re}
    
    
def compute_repetition_batch(sentences):

    rep_score = []

    for sentence in sentences:
        score = compute_repetition_single(sentence)
        rep_score.append(score['rep'])

    rep_score = sum(rep_score)/len(rep_score)
    return {'rep': rep_score}
                     
def compute_repetition_single(sentence):

    #doc = nlp(sentence)
    #token_list = [token.text for token in doc]
    token_list = sentence.split()

    grams = list(ngrams(token_list, 2))
    grams += list(ngrams(token_list, 3))
    grams += list(ngrams(token_list, 4))
    grams = ['_'.join(x for x in gram) for gram in grams]

    counter = Counter(grams)
    rep_count = 0
    for _, freq in counter.items():
        if (freq > 2): rep_count += 1
    
    try: return {'rep': rep_count/len(counter)}
    except: return {'rep': 0}

def compute_spm_batch(target_list, values_list, lib = 'spacy'):

    #recall, precision, f1 = [], [], []
    spm = []
    for values, target in zip(values_list, target_list):

        score = compute_spm_single(target, values, lib = lib)
        '''print('target: ', target)
        print('values: ', values)
        print('score: ', score)
        print('----------------------')'''
        
        spm.append(score['spm'])
        #recall.append(score['spm_recall'])
        #precision.append(score['spm_precision'])
        #f1.append(score['spm_f1'])

    #recall = sum(recall)/len(recall)
    #precision = sum(precision)/len(precision)
    #f1 = sum(f1)/len(f1)
    spm = sum(spm)/len(spm)

    result_dict = {}
    result_dict['spm'] = spm
    #result_dict['spm_recall'] = recall
    #result_dict['spm_precision'] = precision
    #result_dict['spm_f1'] = f1

    return result_dict

def compute_spm_single(target, values, lib = 'spacy'):

    count = 0

    #doc = nlp(target)
    #target = ' '.join(token.text for token in doc) # more exact with datetime

    # NLTK
    #doc = word_tokenize(target)
    #target = ' '.join(token for token in doc) # more exact with datetime

    for value in values:
        try:
            datetime = convert_datetime(value) # '+2005-01-01T00:00:00Z'
            if (datetime != ''):
                value = str(datetime.year) # get "year" only for easier comparing
        except: pass

        flag, sen_id = match_single_value(value, [target])
        if (flag == True): count = count + 1
        
    '''nouns = []
    if (lib == 'spacy'):
        doc = nlp(target)
        nouns = [np.text.lstrip('a ').lstrip('an ').lstrip('the ') for np in doc.noun_chunks]
        
    else: # nltk
        nouns = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(target)) if pos[0] == 'N']
        nouns = list(set(nouns))

    # lowercase
    nouns = [n.lower() for n in nouns]
    values = [v.lower() for v in values]
    commons = list(set(nouns).intersection(values))

    recall = count/len(values)
    if (len(nouns) == 0): precision = 0
    else: precision = len(commons)/len(nouns)

    f1 = 0
    if (recall != 0 and precision != 0): f1 = (2*recall*precision)/(recall + precision)'''

    result_dict = {}
    try:
        result_dict['spm'] = count/len(values)
    except:
        result_dict['spm'] = 0
    #result_dict['spm_recall'] = recall
    #result_dict['spm_precision'] = precision
    #result_dict['spm_f1'] = f1

    return result_dict


def compute_rouge_batch(predictions, references):

    """
        predictions: list
        references: list
    """

    result_dict = {}

    predictions = list(predictions)
    references = list(references)
    
    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict
        
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []

    i =  0
    for pre, ref in zip(predictions, references):
        output = compute_rouge_single(pre, ref)
        #print('pre: ', pre)
        #print('ref: ', ref)
        #print(i, len(predictions))
        i = i + 1
        #print('----------------')
        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])
    
    result_dict['rouge1_precision'] = sum(r1_pre)/len(r1_pre)
    result_dict['rouge2_precision'] = sum(r2_pre)/len(r2_pre)
    result_dict['rougeL_precision'] = sum(rL_pre)/len(rL_pre)
        
    result_dict['rouge1_recall'] = sum(r1_re)/len(r1_re)
    result_dict['rouge2_recall'] = sum(r2_re)/len(r2_re)
    result_dict['rougeL_recall'] = sum(rL_re)/len(rL_re)
        
    result_dict['rouge1_fmeasure'] = sum(r1_fm)/len(r1_fm)
    result_dict['rouge2_fmeasure'] = sum(r2_fm)/len(r2_fm)
    result_dict['rougeL_fmeasure'] = sum(rL_fm)/len(rL_fm)

    # memory error
    '''output = rouge.compute(predictions=[predictions], references=[references])
    result_dict['rouge1_precision'] = output['rouge1'].mid.precision
    result_dict['rouge2_precision'] = output['rouge2'].mid.precision
    result_dict['rougeL_precision'] = output['rougeL'].mid.precision
        
    result_dict['rouge1_recall'] = output['rouge1'].mid.recall
    result_dict['rouge2_recall'] = output['rouge2'].mid.recall
    result_dict['rougeL_recall'] = output['rougeL'].mid.recall
        
    result_dict['rouge1_fmeasure'] = output['rouge1'].mid.fmeasure
    result_dict['rouge2_fmeasure'] = output['rouge2'].mid.fmeasure
    result_dict['rougeL_fmeasure'] = output['rougeL'].mid.fmeasure'''
   
    return result_dict
    

def compute_rouge_single(prediction, reference):
    """
        predictions: single string
        references: single string
    """

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict
    
    output = rouge.compute(predictions=[prediction], references=[reference])
    
    result_dict['rouge1_precision'] = output['rouge1'].mid.precision
    result_dict['rouge2_precision'] = output['rouge2'].mid.precision
    result_dict['rougeL_precision'] = output['rougeL'].mid.precision
        
    result_dict['rouge1_recall'] = output['rouge1'].mid.recall
    result_dict['rouge2_recall'] = output['rouge2'].mid.recall
    result_dict['rougeL_recall'] = output['rougeL'].mid.recall
        
    result_dict['rouge1_fmeasure'] = output['rouge1'].mid.fmeasure
    result_dict['rouge2_fmeasure'] = output['rouge2'].mid.fmeasure
    result_dict['rougeL_fmeasure'] = output['rougeL'].mid.fmeasure
   
    return result_dict  


def compute_bleu_single(prediction, reference):

    """
        BLEU should be measured in the corpus level, use compute_bleu_batch() instead
    """

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict

    #prediction = [prediction.split()] # two brackets in total
    #reference = [[reference.split()]] # three brackets in total

    #print('prediction: ', prediction)
    #print('reference: ', reference)
    
    output = {}
    try:
        output = sacrebleu.sentence_bleu(prediction, [reference])
    except Exception as e:
        print('Error -- compute_bleu_single: ', e)
        result_dict['bleu'] = 0
        return result_dict

    result_dict['bleu'] = output.score/100
    return result_dict

def compute_bleu_batch(predictions, references, level = 'corpus'):

    result_dict = {}
    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict

    #predictions = [[x.strip('.') for x in pre.split()] for pre in predictions] # two brackets in total
    #references = [[[x.strip('.') for x in ref.split()] for ref in references]] # three brackets in total
    
    output = {}
    score = []
    try:
        if (level == 'corpus'):
            output = sacrebleu.corpus_bleu(predictions, [references])
        else:
            for pred, ref in zip(predictions, references):
                score.append(compute_bleu_single(pred, ref)['bleu'])
                
    except Exception as e:
        print('Error -- compute_bleu_batch: ', e)
        result_dict['bleu'] = 0
        return result_dict

    if (level == 'sentence'):
        score = sum(score)/len(score)
        result_dict['bleu'] = score
    else:
        result_dict['bleu'] = output.score/100
        
    return result_dict


def compute_meteor_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict

    output = meteor.compute(predictions=[prediction], references=[reference])
    result_dict['meteor'] = output['meteor']
    
    return result_dict


def compute_meteor_batch(predictions, references):

    result_dict = {}
    
    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict

    output = []
    for pred, ref in zip(predictions, references):
        output.append(compute_meteor_single(pred, ref)['meteor'])

    output = sum(output)/len(output)
    print(output)

    # memory error
    #output = meteor.compute(predictions=[predictions], references=[references])
    #print(output)
    
    result_dict['meteor'] = output
    return result_dict


def compute_bertscore_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict
    
    # microsoft/deberta-xlarge-mnli (best model), model_type=roberta-large (default)
    output = bertscore.compute(predictions=[prediction], references=[reference], lang='en')
        
    result_dict['bertscore_precision'] = output['precision'][0]
    result_dict['bertscore_recall'] = output['recall'][0]
    result_dict['bertscore_f1'] = output['f1'][0]

    return result_dict

def compute_bertscore_batch(predictions, references):

    result_dict = {}
    prel, re, f1 = [], [], []
    
    for pre, ref in zip(predictions, references):
        output = bertscore.compute(predictions=[pre], references=[ref], lang='en')       
        prel.append(output['precision'][0])
        re.append(output['recall'][0])
        f1.append(output['f1'][0])

    result_dict = {}
    result_dict['bertscore_precision'] = sum(prel)/len(prel)
    result_dict['bertscore_recall'] = sum(re)/len(re)
    result_dict['bertscore_f1'] = sum(f1)/len(f1)

    return result_dict


#.....................................

#pred = 'Hoima is a town and the capital of Hoima District in the KwaZulu-Natal Region of Uganda.'
#source = 'Hoima is a city in the Western Region of Uganda.", "It is the main municipal, administrative, and commercial center of Hoima District.'

'''pred1 = ' '.join(token.text for token in nlp(pred))
source1 = ' '.join(token.text for token in nlp(source))
print(compute_bleu_single(pred1, source1))'''

#print(compute_bleu_batch([pred], [source]))
#print('sacrebleu: ', sacrebleu.sentence_bleu(pred, [source]))

#from sacrebleu.metrics import BLEU, CHRF, TER
#bleu = BLEU()
#print(bleu.corpus_score([pred], [[source]]))
 

#print(compute_bertscore_single('the quick brown fox', 'the fast brown fox'))
#print(compute_bertscore_batch(['the quick brown fox'], ['the fast brown fox']))

'''predictions = ['The', 'film', 'Death', 'on', 'a', 'Factory', 'Farm', 'was', 'shown', 'on', 'HBO', 'and', \
               'it', 'was', 'written', 'and', 'directed', 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', \
               'editor', 'of', 'the', 'film']
references =  [['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'Film', 'directed', 'and', 'produced', \
                 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', 'editor'],
                ['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'film', 'directed,', 'produced,', 'and', \
                 'edited', 'by', 'Geof', 'Bartz', 'and', 'Tom', 'Simon'],
                ['Tom', 'Simon', 'directed', 'and', 'produced', 'the', 'film', '‘Death', 'on', 'a', 'Factory', 'Farm’,', \
                 'which', 'was', 'broadcasted', 'by', 'HBO', 'Geof', 'Bartz', 'on', 'the', 'other', 'hand', 'edited', 'the', \
                 'film']]

print(compute_meteor_batch(predictions, references))'''

'''from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']

reference_str = ' '.join(x for x in reference[0])
candidate_str = ' '.join(x for x in candidate)

print('nltk: ', sentence_bleu(reference, candidate))

print('datasets: ', bleu.compute(predictions=[candidate], references=[reference]))

print('func: ', compute_bleu_single(candidate1, reference1))'''

'''from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
bleu = BLEU()


predictions = ['The', 'film', 'Death', 'on', 'a', 'Factory', 'Farm', 'was', 'shown', 'on', 'HBO', 'and', \
               'it', 'was', 'written', 'and', 'directed', 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', \
               'editor', 'of', 'the', 'film']
references =  [['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'Film', 'directed', 'and', 'produced', \
                 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', 'editor'],
                ['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'film', 'directed,', 'produced,', 'and', \
                 'edited', 'by', 'Geof', 'Bartz', 'and', 'Tom', 'Simon'],
                ['Tom', 'Simon', 'directed', 'and', 'produced', 'the', 'film', '‘Death', 'on', 'a', 'Factory', 'Farm’,', \
                 'which', 'was', 'broadcasted', 'by', 'HBO', 'Geof', 'Bartz', 'on', 'the', 'other', 'hand', 'edited', 'the', \
                 'film']]
      
'''

'''references = ["label : Cheongju KB Stars | league : Women's Korean Basketball League | headquarters location : Cheongju | country : South Korea | sport : basketball"]

predictions = ["""label : Cheongju KB Stars | league : Women's Korean Basketball League | headquarters location : Cheongju | country : South Korea | sport : basketball"""]

#references1 = [' '.join(x for x in reference) for reference in references]
#predictions1 = [' '.join(x for x in predictions)]

#print('references1: ', references1)
#print('predictions1: ', predictions1)

from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
bleu1 = BLEU()
print('bleu: ', bleu1.corpus_score(predictions, [references]))
print('datasets: ', compute_bleu_batch(predictions, references))   
print('sacrebleu: ', sacrebleu.corpus_bleu(predictions, [references]))'''




