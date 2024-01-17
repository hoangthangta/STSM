from read_write_file import *
from metrics import *
from utils import *
import argparse


def test_dataset(input_file = 'dataset/test_random.json', predict_field = 'target_pred_template', output_dir = 'output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    bleu, meteor = [], []
    rouge1_precision, rouge2_precision, rougeL_precision = [], [], []
    rouge1_recall, rouge2_recall, rougeL_recall = [], [], []
    rouge1_f1, rouge2_f1, rougeL_f1 = [], [], []
    bertscore_f1, bertscore_recall, bertscore_precision = [], [], []

    bleu_2, meteor_2 = [], []
    rouge1_precision_2, rouge2_precision_2, rougeL_precision_2 = [], [], []
    rouge1_recall_2, rouge2_recall_2, rougeL_recall_2 = [], [], []
    rouge1_f1_2, rouge2_f1_2, rougeL_f1_2 = [], [], []
    bertscore_f1_2, bertscore_recall_2, bertscore_precision_2 = [], [], []

    spm = []
    rep = []

    for i, item in enumerate(dataset):
        sys.stdout.write('Checking item: %d \t Model: %s \r' % (i + 1, len(dataset)))
        sys.stdout.flush()

        source = item['source']
        source_list = item['source_list']
        source_values = []

        for x in source_list:
            if (len(x) == 4): source_values.append(x[3]) # add qualifiers
            source_values.append(x[1])      

        target = item['target']
        best_pred = item[predict_field]

        # split tokens by spaCy 
        '''doc = nlp(source)
        source_tokenized = ' '.join(token.text for token in doc if token.text.strip() != '')
        doc = nlp(best_pred)
        best_pred_tokenized = ' '.join(token.text for token in doc if token.text.strip() != '')
        doc = nlp(target)
        target_tokenized = ' '.join(token.text for token in doc if token.text.strip() != '')'''
        
        output = compute_bleu_single(best_pred, target)
        bleu.append(output['bleu'])

        output = compute_meteor_single(best_pred, target)
        meteor.append(output['meteor'])

        output = compute_rouge_single(best_pred, target)
        rouge1_precision.append(output['rouge1_precision'])
        rouge2_precision.append(output['rouge2_precision'])
        rougeL_precision.append(output['rougeL_precision'])

        rouge1_recall.append(output['rouge1_recall'])
        rouge2_recall.append(output['rouge2_recall'])
        rougeL_recall.append(output['rougeL_recall'])

        rouge1_f1.append(output['rouge1_fmeasure'])
        rouge2_f1.append(output['rouge2_fmeasure'])
        rougeL_f1.append(output['rougeL_fmeasure'])

        output = compute_bertscore_single(best_pred, target)
        bertscore_f1.append(output['bertscore_f1'])
        bertscore_recall.append(output['bertscore_recall'])
        bertscore_precision.append(output['bertscore_precision'])

        output = compute_bleu_single(best_pred, source)
        bleu_2.append(output['bleu'])

        output = compute_meteor_single(best_pred, source)
        meteor_2.append(output['meteor'])

        output = compute_rouge_single(best_pred, source)
        rouge1_precision_2.append(output['rouge1_precision'])
        rouge2_precision_2.append(output['rouge2_precision'])
        rougeL_precision_2.append(output['rougeL_precision'])
        
        rouge1_recall_2.append(output['rouge1_recall'])
        rouge2_recall_2.append(output['rouge2_recall'])
        rougeL_recall_2.append(output['rougeL_recall'])

        rouge1_f1_2.append(output['rouge1_fmeasure'])
        rouge2_f1_2.append(output['rouge2_fmeasure'])
        rougeL_f1_2.append(output['rougeL_fmeasure'])

        output = compute_bertscore_single(best_pred, source)
        bertscore_f1_2.append(output['bertscore_f1'])
        bertscore_recall_2.append(output['bertscore_recall'])
        bertscore_precision_2.append(output['bertscore_precision'])
        output = compute_spm_single(best_pred, source_values)
        spm.append(output['spm'])
        
        print('source: ', source)
        print('source_values: ', source_values)
        print('target: ', target)
        print('best_pred: ', best_pred)
        print('spm: ', output)
        print('------------------')

        output = compute_repetition_single(best_pred)
        rep.append(output['rep'])
        

    result_dict = {}
    result_dict['file'] = input_file

    can_tar_dict = {}
    can_tar_dict['bleu'] = get_average(bleu)
    can_tar_dict['meteor'] = get_average(meteor)

    can_tar_dict['rouge1_precision'] = get_average(rouge1_precision)
    can_tar_dict['rouge2_precision'] = get_average(rouge2_precision)
    can_tar_dict['rougeL_precision'] = get_average(rougeL_precision)

    can_tar_dict['rouge1_recall'] = get_average(rouge1_recall)
    can_tar_dict['rouge2_recall'] = get_average(rouge2_recall)
    can_tar_dict['rougeL_recall'] = get_average(rougeL_recall)

    can_tar_dict['rouge1_fmeasure'] = get_average(rouge1_f1)
    can_tar_dict['rouge2_fmeasure'] = get_average(rouge2_f1)
    can_tar_dict['rougeL_fmeasure'] = get_average(rougeL_f1)

    can_tar_dict['bertscore_f1'] = get_average(bertscore_f1)
    can_tar_dict['bertscore_recall'] = get_average(bertscore_recall)
    can_tar_dict['bertscore_precision'] = get_average(bertscore_precision)
    result_dict['prediction_vs_target'] = can_tar_dict

    can_source_dict = {}
    can_source_dict['bleu'] = get_average(bleu_2)
    can_source_dict['meteor'] = get_average(meteor_2)
    
    can_source_dict['rouge1_precision'] = get_average(rouge1_precision_2)
    can_source_dict['rouge2_precision'] = get_average(rouge2_precision_2)
    can_source_dict['rougeL_precision'] = get_average(rougeL_precision_2)

    can_source_dict['rouge1_recall'] = get_average(rouge1_recall_2)
    can_source_dict['rouge2_recall'] = get_average(rouge2_recall_2)
    can_source_dict['rougeL_recall'] = get_average(rougeL_recall_2)

    can_source_dict['rouge1_fmeasure'] = get_average(rouge1_f1_2)
    can_source_dict['rouge2_fmeasure'] = get_average(rouge2_f1_2)
    can_source_dict['rougeL_fmeasure'] = get_average(rougeL_f1_2)

    can_source_dict['bertscore_f1'] = get_average(bertscore_f1_2)
    can_source_dict['bertscore_recall'] = get_average(bertscore_recall_2)
    can_source_dict['bertscore_precision'] = get_average(bertscore_precision_2)
    result_dict['prediction_vs_source'] = can_source_dict

    result_dict['spm'] = get_average(spm)
    result_dict['rouge1_precision'] = get_average(rouge1_precision)
    result_dict['rep'] = get_average(rep)
    result_dict['non_rep'] = 1 - get_average(rep)

    # fuse metric
    scores = [result_dict['spm'], result_dict['rouge1_precision'], result_dict['non_rep']]
    result_dict['fused_metric'] = fuse_score(scores)
    result_dict['predict_field'] = predict_field

    output_dir += '/test_result.json'
    output_dir = output_dir.replace('//','/')

    write_single_dict_to_json_file(output_dir, result_dict, file_access = 'a')
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--input_file', type=str, default='dataset/test_random.json')
    parser.add_argument('--predict_field', type=str, default='target_pred_template')
    parser.add_argument('--output_dir', type=str, default='output/')
    args = parser.parse_args()

    test_dataset(input_file = args.input_file, predict_field = args.predict_field, output_dir = args.output_dir)



