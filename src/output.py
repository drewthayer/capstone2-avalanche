import pickle
import numpy as np
import matplotlib.pyplot as plt

output_raw = pickle.load(open('output.p', 'rb'))

thresholds = np.linspace(0.1,6,100)
TP_rates = []
FP_rates = []
accuracies = []
precisions = []
recalls = []
threshold_list = []
for threshold in thresholds:
    output = np.where(output_raw >= threshold, 1, 0)
    true = output[:,0]
    predicted = output[:,1]
    TP = np.sum(np.logical_and(predicted==1, true==1))
    TN = np.sum(np.logical_and(predicted==0, true==0))
    FP = np.sum(np.logical_and(predicted==1, true==0))
    FN = np.sum(np.logical_and(predicted==0, true==1))
    TP_rates.append(TP/len(true))
    FP_rates.append(FP/len(true))

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    threshold_list.append(threshold)
    accuracies.append(accuracy)
    recalls.append(recalls)
    precisions.append(precisions)
    #print('threshold = {}'.format(threshold))
    #print('accuracy = {:0.3f}'.format(accuracy))
    #print('precision = {:0.3f}'.format(precision))
    #print('recall= {:0.3f}'.format(recall))

plt.plot(FP_rates,TP_rates,'k')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('Receiver Operating Characteristic')
plt.savefig('ROC.png',dpi=250)
plt.close()

# fig, ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].plot(thresholds,accuracies,'k')
# ax[0].set_title('accuracy curve')
# ax[0].set_xlabel('probability threshold')
# ax[0].set_ylabel('accuracy')
# ax[1].plot(thresholds,recalls,'k')
# ax[1].set_title('recall curve')
# ax[1].set_xlabel('probability threshold')
# ax[1].set_ylabel('recall')
# plt.show()


#plt.plot(thresholds,recalls)
