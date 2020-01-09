from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):

    #Dice = (2*|X & Y|)/ (|X|+ |Y|)
    #     =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


'''
# Test
y_true = np.array([[0,0,1,0],[0,0,1,0],[0,0,1.,0.]])
y_pred = np.array([[0,0,0.9,0],[0,0,0.1,0],[1,1,0.1,1.]])

r = dice_coef_loss(y_true, y_pred)
print('dice_coef_loss', r)

pre = precision(y_true, y_pred)
print('precision = ', pre)
re = recall(y_true, y_pred)
print('recall = ', re)

# https://github.com/keras-team/keras/issues/5400
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
'''
