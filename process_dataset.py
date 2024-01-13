import datasets

from read_write_file import *

def convert_dataset(ds_name = 'dart'):

    ds = datasets.load_dataset(ds_name)
    train = ds['train']
    val = ds['validation']
    test = ds['test']

    train_list, val_list, test_list = [], [], []
    if (ds_name == 'dart'):

        # flat each pair of (source, target)
        for item in train:
            texts = [x for x in item['annotations']['text']]
            for text in texts:
                source = ' | '.join(' : '.join(x for x in trip) for trip in item['tripleset'])
                target = text
                train_list.append({'source':source, 'target':target})

        # flat each pair of (source, target)
        for item in val:
            texts = [x for x in item['annotations']['text']]
            for text in texts:
                source = ' | '.join(' : '.join(x for x in trip) for trip in item['tripleset'])
                target = text 
                val_list.append({'source':source, 'target':target}) 

        # multiple target
        for item in test:
            #texts = [x for x in item['annotations']['text']]

            #for text in texts:
            source = ' | '.join(' : '.join(x for x in trip) for trip in item['tripleset'])
            target = [x for x in item['annotations']['text']]
            #target = text
            test_list.append({'source':source, 'target':target})

    if (ds_name == 'e2e_nlg'):
        for item in train:
            source = ' | '.join(x.replace('[', ' : ').strip(']') for x in item['meaning_representation'].split(', '))
            target  = item['human_reference']
            train_list.append({'source':source, 'target':target})

        for item in val:
            source = ' | '.join(x.replace('[', ' : ').strip(']') for x in item['meaning_representation'].split(', '))
            target  = item['human_reference']
            val_list.append({'source':source, 'target':target})

        test_dict = {}
        for item in test:

            source = ' | '.join(x.replace('[', ' : ').strip(']') for x in item['meaning_representation'].split(', '))
            target  = item['human_reference']

            #test_list.append({'source':source, 'target':target})
            
            if (source not in test_dict):
                test_dict[source] = {'target': [target]}
            else:
                test_dict[source]['target'].append(target)

        for k, v in test_dict.items():
            test_list.append({'source':k, 'target':v['target']})
            
            

    output = 'dataset/' + ds_name + '/'
    import os
    if not os.path.exists(output): os.makedirs(output)
        
    write_list_to_jsonl_file(output + 'train.json', train_list, file_access = 'w')
    write_list_to_jsonl_file(output + 'val.json', val_list, file_access = 'w')
    write_list_to_jsonl_file(output + 'test.json', test_list, file_access = 'w')

    return train_list, val_list, test_list

if __name__ == "__main__":
    #convert_dataset('dart')
    convert_dataset('e2e_nlg')
