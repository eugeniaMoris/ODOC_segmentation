
def list_2_string(input_l):
    'convert the lint into a single string separate by ,'
    if (input_l == None) or (len(input_l) == 0):
        return ''
    else:
        return ','.join(list(input_l))
