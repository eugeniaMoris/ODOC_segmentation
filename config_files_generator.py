from configparser import ConfigParser
import numpy as np
import copy

fuerza = [10, 25, 50]

probabilidad= [0.10, 0.25, 0.50]

original = ConfigParser()
original.read( 'config_files/DC_DRISHTI_augm_f-10_p-10.ini')

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
        new_config['affine']['translate']= str(0) + ',' + str(0)
        scale_min = 1.0 - f/100
        scale_max = 1.0 + f/100
        new_config['affine']['scale'] = str(scale_min) + ','+str(scale_max)

        with open('REFUGE_STEP2_f-' + str(f) + '_p-' + str(int(p*100)) + '.ini', 'w') as configfile:
            new_config.write(configfile)



