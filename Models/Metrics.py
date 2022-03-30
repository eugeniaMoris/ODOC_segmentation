import numpy as np
    

class Dice(Metric):
    '''
    Dice = 2 * the Area of Overlap divided by the total number of pixels in both images 
    '''

    def get_name(self):
        return 'dice'

    def compute(self, gt, pred):
        return self.dice_metric(gt, pred)

    def dice_metric(gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = (gt > 0).flatten()

        if np.any(gt):

            pred = (pred > 0).flatten()

            numerator = np.sum(np.multiply(pred, gt))

            if numerator == 0.0:
                return 0.0
            else:
                denominator = np.sum(pred) + np.sum(gt)
                return (2.0 * numerator) / denominator

        else:

            return np.nan