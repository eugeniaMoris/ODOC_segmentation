from configparser import ConfigParser
import numpy as np
import copy

fuerza = [10, 25, 50, 75]
probabilidad= [0.10, 0.25, 0.50, 0.75]

original = ConfigParser()
original.read( 'DRISHTI_configuration.ini')

for f in fuerza:
    for p in probabilidad:
        new_config = copy.deepcopy(original)
        new_config['augmentation']['flip'] = str(p)
        new_config['augmentation']['blur'] = str(p)
        new_config['augmentation']['color'] = str(p)
        new_config['augmentation']['affine'] = str(p)

        new_config['blur']['sigma'] = str(f/100)

        new_config['color']['brigtness'] = str(f/100)
        new_config['color']['contrast'] = str(f/100)
        new_config['color']['saturation'] = str(f/100)
        new_config['color']['hue'] = str(f/100)

        new_config['affine']['degrees']= '-' + str(f) + ',' + str(f)
        new_config['affine']['translate']= str(int(f/10)) + ',' + str(int(f/10))
        scale_min = 1.0 - f/100
        scale_max = 1.0 + f/100
        new_config['affine']['scale'] = str(scale_min) + ','+str(scale_max)

        with open('DRISHTI_augm_f-' + str(f) + '_p-' + str(int(p*100)) + '.ini', 'w') as configfile:
            new_config.write(configfile)



