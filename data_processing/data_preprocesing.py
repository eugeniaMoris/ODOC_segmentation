from configparser import ConfigParser
from os import path


def list_2_string(input_l):
    'convert the lint into a single string separate by ,'
    if (input_l == None) or (len(input_l) == 0):
        return ''
    else:
        return ','.join(list(input_l))

def save_split_file(dst, filename, dataset, train, validation, test):
    '''
    Guardo la separacion entre training, validation y test en archivo .ini
    ----- 
    Input
    dst: path donde guardar el archivo
    filename: nombre del proyecto
    dataset: dataset de donde provienen las imagenes
    train: lista de nombres de las imagenes separadas para training
    validation: lista de nombre de las imagenes separadas para validacion
    test: lista de nombres de las imagenes separadas para el test

    ---
    Output
    split file
    '''

    if '' in train:
        train.remove('')
    if '' in validation:
        validation.remove('')
    if '' in test:
        test.remove('')

    split = ConfigParser()
    split.add_section('split')
    split.set('split','type','holdout')
    split.set('split','training', list_2_string(train))
    split.set('split', 'validation', list_2_string(validation))
    split.set('split','test',list_2_string(test))

    split_file = open(path.join(dst,filename + '_' + dataset + '.ini'),'w')
    split.write(split_file)
    split_file.close()

    print(' - Training: {} images'.format(len(train)))
    print(' - Validation: {} images'.format(len(validation)))
    print(' - Test: {} images'.format(len(test)))

    return split
