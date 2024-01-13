from datetime import *
import statistics


def get_average(list):
    if (len(list) != 0): return sum(list)/len(list)
    return 0

def fuse_score(scores):

    """
        calculate harmonic mean
    """
    filtered_scores = []
    for score in scores:
        if (type(score) is list): score = score[0]
        if (score > 0): filtered_scores.append(score)
   
    return statistics.harmonic_mean(filtered_scores)


def convert_datetime_to_string(label, date_format = 'ISO8601'):


    str_label = ''
    try:
        if (date_format == 'ISO8601'):
            dt = convert_datetime_ISO8601(label)
        else:
            dt = convert_datetime(label)
    except:
        return label
    
    if (dt != ''):
        if (dt.hour == 0 and dt.minute == 0 and dt.second == 0):
            if (dt.day == 1 and dt.month == 1):
                str_label = str(dt.year)
            else:
                str_label = str(dt.day) + ' ' + dt.strftime("%B") + ', ' + str(dt.year)
        else:
            if (dt.day == 1 and dt.month == 1):
                str_label = str(dt.year) + ' ' + dt.hour  + ':' + dt.minute + ':' + dt.second
            else:
                str_label = str(dt.day) + ' ' + dt.strftime("%B") + ', ' + str(dt.year) \
                            + ' ' + dt.hour  + ':' + dt.minute + ':' + dt.second

    if (dt == ''): return label

    
    return str_label
    

def convert_datetime_ISO8601(label):

    """
        convert label to datetime (ISO8601)
    """

    try:
        if ('+' in label): label = label.replace('+', '')
        dt = datetime.strptime(label, '%Y-%m-%dT%H:%M:%SZ')
        #print(dt.year)
        return dt
    except Exception as e:
        try:
            temp = label.split('T')
            temp1 = temp[0].split('-')
            if (temp1[1] == '00'):
                temp1[1] = '01'
            if (temp1[2] == '00'):
                temp1[2] = '01'
            label = '-'.join(e for e in temp1)
            label = label + 'T' + temp[1]
            dt = datetime.strptime(label, '%Y-%m-%dT%H:%M:%SZ')
            return dt       
        except Exception as e:
            #print('Error -- convert_datetime: ', e)
            pass
    
    return ''

def convert_datetime(label):

    """
        convert label to datetime (ISO8601)
    """

    #"Jun 28 2018 at 7:40AM" -> "%b %d %Y at %I:%M%p"
    #"September 18, 2017, 22:19:55" -> "%B %d, %Y, %H:%M:%S"
    #"Sun,05/12/99,12:30PM" -> "%a,%d/%m/%y,%I:%M%p"
    #"Mon, 21 March, 2015" -> "%a, %d %B, %Y"
    #"2018-03-12T10:12:45Z" -> "%Y-%m-%dT%H:%M:%SZ"

    try:
        if ('+' in label): label = label.replace('+', '')
        dt = datetime.strptime(label, '%d %B, %Y')
        return dt
    except Exception as e:
        try:
            dt = datetime.strptime(label, '%d %B %Y')
            return dt       
        except Exception as e:
            #print('Error -- convert_datetime: ', e)
            pass
    
    return ''


def match_multi_value(value_list, sen_list):

    flag = False
    sen_id = 0

    for value in value_list:
        flag, sen_id = match_single_value(value, sen_list)
        if (flag == True): break

    return flag, sen_id


def match_single_value(value, sen_list):

    value = str(value)

    flag = False
    sen_id = 0
    for sen in sen_list:
        # should not use lower
        if (' ' + value + ' ' in sen):
            flag = True
            break

        try: 
            head = value + ' '
            if (sen.index(head) == 0):
                flag = True
                break
        except:
            pass

        try:
            tail = ' ' + value + '.'
            if (sen.index(tail) + len(tail)  == len(sen)):
                flag = True
                break
        except:
            pass
        
        sen_id = sen_id + 1

    return flag, sen_id
    
